
import torch
import torch.nn.functional as F

class NSADetector:
    def __init__(self, embedding_dim=64, num_detectors=100, threshold=0.12):
        self.embedding_dim = embedding_dim
        self.num_detectors = num_detectors
        self.threshold = threshold
        self.detectors = None

    def generate_detectors(self, self_embeddings, forbidden_embeddings, max_attempts=10000):
        detectors = []
        attempts = 0
        n_forbidden = len(forbidden_embeddings)

        while len(detectors) < self.num_detectors and attempts < max_attempts:
            idx = torch.randint(0, n_forbidden, (1,)).item()
            center = forbidden_embeddings[idx]
            noise = torch.randn(self.embedding_dim) * 0.02
            candidate = F.normalize(center + noise, p=2, dim=0)

            dists = torch.cdist(
                candidate.unsqueeze(0),
                self_embeddings
            ).squeeze(0)

            if dists.min().item() > self.threshold:
                detectors.append(candidate)

            attempts += 1

        self.detectors = torch.stack(detectors) if detectors else None
        print(f"NSA 유효 탐지기: {len(detectors)}개 / 시도: {attempts}회", flush=True)
        return self.detectors

    def detect(self, embeddings):
        if self.detectors is None:
            return None, None
        dists = torch.cdist(embeddings, self.detectors)
        min_dists = dists.min(dim=1).values
        violations = min_dists < self.threshold
        return violations, min_dists

    def affinity(self, embeddings, labels):
        if self.detectors is None:
            return None

        n_detectors = len(self.detectors)
        affinities = []

        for i in range(n_detectors):
            det = self.detectors[i].unsqueeze(0)
            dists = torch.cdist(embeddings, det).squeeze(1)
            detected = dists < self.threshold

            forbidden_mask = labels == 0
            possible_mask  = labels == 1

            dr  = detected[forbidden_mask].float().mean().item() if forbidden_mask.sum() > 0 else 0.0
            far = detected[possible_mask].float().mean().item()  if possible_mask.sum() > 0 else 0.0

            affinities.append(dr - 0.3 * far)

        return torch.tensor(affinities)


class CSAMaturator:
    def __init__(self, clone_ratio=0.5, gamma=2.0, mutation_sigma=0.01):
        self.clone_ratio = clone_ratio
        self.gamma = gamma
        self.mutation_sigma = mutation_sigma

    def clone_and_mutate(self, detectors, affinities, forbidden_mask_detectors):
        n = len(detectors)
        new_detectors = []
        sorted_idx = torch.argsort(affinities, descending=True)

        for rank, idx in enumerate(sorted_idx):
            clone_num = max(1, round(self.clone_ratio * n / (rank + 1)))
            if forbidden_mask_detectors[idx]:
                clone_num = int(clone_num * self.gamma)

            for _ in range(clone_num):
                noise = torch.randn_like(detectors[idx]) * self.mutation_sigma
                mutant = F.normalize(detectors[idx] + noise, p=2, dim=0)
                new_detectors.append(mutant)

        if new_detectors:
            return torch.stack(new_detectors[:n])
        return detectors

    def update_gat_weights(self, model, affinities, eta=0.01):
        """AIN 수식: α_ij(t+1) = α_ij(t) + η × (f(d_ij) - α_ij(t))"""
        mean_affinity = affinities.mean().item()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'att' in name or 'weight' in name:
                    target = param.data + eta * (mean_affinity - param.data)
                    param.data = target

        return mean_affinity


def run_nsa_csa_pipeline(model, rs_labeled, device, num_seq=30):
    """
    전체 NSA + CSA 파이프라인 실행
    에피소드 종료 후 호출
    """
    from gat_model import build_graph, check_violation

    model.eval()
    self_embeddings = []
    forbidden_embeddings = []
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for seq in rs_labeled[:num_seq]:
            graph = build_graph(seq)
            x = graph.x.to(device)
            edge_index = graph.edge_index.to(device)
            emb = model(x, edge_index)

            for i in range(len(seq)):
                for j in range(i+1, len(seq)):
                    cat_i = seq[i]["category"]
                    cat_j = seq[j]["category"]
                    emb_pair = ((emb[i] + emb[j]) / 2).cpu()
                    label = 0 if check_violation(cat_i, cat_j) else 1

                    all_embeddings.append(emb_pair)
                    all_labels.append(label)

                    if label == 0:
                        forbidden_embeddings.append(emb_pair)
                    else:
                        self_embeddings.append(emb_pair)

    self_embeddings      = torch.stack(self_embeddings)
    forbidden_embeddings = torch.stack(forbidden_embeddings)
    all_embeddings       = torch.stack(all_embeddings)
    all_labels           = torch.tensor(all_labels, dtype=torch.float)

    # NSA 탐지기 생성
    nsa = NSADetector(embedding_dim=64, num_detectors=100, threshold=0.12)
    nsa.generate_detectors(self_embeddings, forbidden_embeddings)

    # CSA 친화도 계산
    affinities = nsa.affinity(all_embeddings, all_labels)
    print(f"친화도 평균: {affinities.mean():.4f}", flush=True)

    # CSA 복제·변이
    csa = CSAMaturator(clone_ratio=0.5, gamma=2.0, mutation_sigma=0.01)
    forbidden_mask = affinities > affinities.median()
    new_detectors = csa.clone_and_mutate(nsa.detectors, affinities, forbidden_mask)

    # GAT 가중치 업데이트
    mean_affinity = csa.update_gat_weights(model, affinities)
    nsa.detectors = new_detectors

    print(f"GAT 업데이트 완료 (친화도: {mean_affinity:.4f})", flush=True)
    return nsa, csa
