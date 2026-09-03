
import torch
import torch.nn.functional as F
import torch.optim as optim
import pickle

def train_resume(work_dir, epochs=150, lr=0.001, margin=1.0):
    from gat_model import MixedLoadGAT, ContrastiveLoss, build_graph, make_pairs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"디바이스: {device}", flush=True)

    with open(f"{work_dir}/data/rs_labeled.pkl", "rb") as f:
        rs_labeled = pickle.load(f)

    # Best Model 불러오기
    model = MixedLoadGAT().to(device)
    model.load_state_dict(
        torch.load(f"{work_dir}/checkpoints/gat_best.pt",
                   map_location=device)
    )
    print("Best Model 로드 완료", flush=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = ContrastiveLoss(margin=margin)

    best_loss = float("inf")
    print(f"이어서 학습 시작 (lr={lr}, margin={margin})...", flush=True)
    print("="*50, flush=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_forbidden_dist = 0
        epoch_possible_dist = 0
        n_batches = 0

        for seq in rs_labeled[:100]:
            optimizer.zero_grad()
            graph = build_graph(seq)
            x = graph.x.to(device)
            edge_index = graph.edge_index.to(device)
            embeddings = model(x, edge_index)
            emb_i, emb_j, labels = make_pairs(seq, embeddings, device)
            loss, _ = criterion(emb_i, emb_j, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            with torch.no_grad():
                dist = F.pairwise_distance(emb_i, emb_j)
                forbidden_mask = labels == 0
                possible_mask  = labels == 1
                if forbidden_mask.sum() > 0:
                    epoch_forbidden_dist += dist[forbidden_mask].mean().item()
                if possible_mask.sum() > 0:
                    epoch_possible_dist += dist[possible_mask].mean().item()
            n_batches += 1

        avg_loss      = epoch_loss / n_batches
        avg_forbidden = epoch_forbidden_dist / n_batches
        avg_possible  = epoch_possible_dist / n_batches

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(),
                       f"{work_dir}/checkpoints/gat_best.pt")

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"금지: {avg_forbidden:.4f} | "
                  f"가능: {avg_possible:.4f} | "
                  f"차이: {avg_forbidden - avg_possible:.4f}",
                  flush=True)

    print("="*50, flush=True)
    print(f"학습 완료 | Best Loss: {best_loss:.4f}", flush=True)
    return model
