
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from torch_geometric.data import Data
import torch
import torch.nn.functional as F
from gat_model import check_violation, MIXED_LOAD_RULES
from nsa_csa import NSADetector, CSAMaturator


def build_graph_safe(sequence):
    n = len(sequence)
    node_feats = []
    for item in sequence:
        feat = [
            item["size"][0], item["size"][1], item["size"][2],
            item["category_id"], item["hazard_level"],
        ]
        node_feats.append(feat)

    edge_index = []
    edge_attr = []
    for i in range(n):
        for j in range(i+1, n):
            cat_i = sequence[i]["category"]
            cat_j = sequence[j]["category"]
            is_forbidden = 1 if check_violation(cat_i, cat_j) else 0
            edge_index.append([i, j])
            edge_index.append([j, i])
            edge_attr.append([is_forbidden])
            edge_attr.append([is_forbidden])

    if len(edge_index) == 0:
        edge_index = [[0, 0]]
        edge_attr  = [[0]]

    return Data(
        x=torch.tensor(node_feats, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float)
    )


class EMSManager:
    """
    GOPT 방식 Extreme Point Set 관리
    아이템 배치 후 극단점 업데이트
    """
    def __init__(self, bin_W, bin_H, bin_D):
        self.bin_W = bin_W
        self.bin_H = bin_H
        self.bin_D = bin_D
        self.reset()

    def reset(self):
        # 초기 극단점: 박스 원점
        self.ems = [(0, 0, 0)]
        self.placed_items = []  # (x, y, z, w, h, d)

    def _is_valid_point(self, x, y, z):
        """박스 범위 내 유효한 점인지 확인"""
        return (0 <= x < self.bin_W and
                0 <= y < self.bin_H and
                0 <= z < self.bin_D)

    def _get_height_at(self, x, z, w, d):
        """특정 위치에서 현재 높이 계산"""
        max_h = 0
        for px, py, pz, pw, ph, pd in self.placed_items:
            # x축 겹침
            if px < x + w and px + pw > x:
                # z축 겹침
                if pz < z + d and pz + pd > z:
                    max_h = max(max_h, py + ph)
        return max_h

    def can_place(self, item):
        """
        GOPT EMS 방식: 극단점에서 배치 가능 여부 확인
        가장 낮은 극단점 반환
        """
        w, h, d = item["size"]
        best_ep = None
        best_score = float("inf")

        for ex, ey, ez in self.ems:
            # 박스 범위 체크
            if ex + w > self.bin_W or ez + d > self.bin_D:
                continue

            # 실제 배치 높이 계산
            floor_h = self._get_height_at(ex, ez, w, d)
            place_y = floor_h

            if place_y + h > self.bin_H:
                continue

            # 안정성 체크 (바닥 또는 기존 아이템 위)
            score = place_y * 100 + ex + ez  # 낮고 왼쪽 앞이 우선
            if score < best_score:
                best_score = score
                best_ep = (ex, place_y, ez)

        return best_ep

    def place(self, item, pos):
        """아이템 배치 후 EMS 업데이트"""
        w, h, d = item["size"]
        x, y, z = pos

        self.placed_items.append((x, y, z, w, h, d))

        # 새 극단점 생성
        new_eps = [
            (x + w, y, z),   # 오른쪽
            (x, y + h, z),   # 위쪽
            (x, y, z + d),   # 앞쪽
        ]

        # 유효한 극단점만 추가
        for ep in new_eps:
            if self._is_valid_point(*ep):
                self.ems.append(ep)

        # 중복 제거
        self.ems = list(set(self.ems))

        item_vol = w * h * d
        return item_vol

    def get_heightmap(self, bin_W, bin_D, bin_H):
        """Height Map 반환 (관찰 공간용)"""
        hm = np.zeros((bin_W, bin_D), dtype=np.float32)
        for px, py, pz, pw, ph, pd in self.placed_items:
            hm[px:px+pw, pz:pz+pd] = np.maximum(
                hm[px:px+pw, pz:pz+pd], (py + ph) / bin_H
            )
        return hm


class PackingEnvWithHold(gym.Env):
    """
    GOPT EMS + Transformer + HEN Hold 통합 환경
    Layer 1: GAT 임베딩 → 관찰 공간
    Layer 2: NSA Constraint Mask + CSA → GAT 업데이트
    Layer 3: Transformer + PPO → Pack/Hold/Close
    배치: GOPT EMS 방식
    Hold: HEN 논문 방식 (K=3)
    """
    def __init__(self, gat_model, rs_labeled, device,
                 bin_size=(10,10,10), hold_buffer_size=3,
                 nsa_detectors=100, nsa_threshold=0.12):
        super().__init__()
        self.gat_model        = gat_model
        self.rs_labeled       = rs_labeled
        self.device           = device
        self.bin_W            = bin_size[0]
        self.bin_H            = bin_size[1]
        self.bin_D            = bin_size[2]
        self.bin_volume       = bin_size[0] * bin_size[1] * bin_size[2]
        self.hold_buffer_size = hold_buffer_size

        # GOPT EMS 관리자
        self.ems_manager = EMSManager(bin_size[0], bin_size[1], bin_size[2])

        # Layer 2: NSA + CSA
        self.nsa = NSADetector(
            embedding_dim=64,
            num_detectors=nsa_detectors,
            threshold=nsa_threshold
        )
        self.csa = CSAMaturator(
            clone_ratio=0.5,
            gamma=2.0,
            mutation_sigma=0.01
        )
        self.nsa_initialized = False
        self.episode_count   = 0
        self.all_embeddings  = []
        self.all_labels      = []

        self.action_space = spaces.Discrete(3 + hold_buffer_size)
        obs_dim = (self.bin_W * self.bin_D) + 5 + 64 + (5 * hold_buffer_size)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(obs_dim,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.episode_count > 0 and len(self.all_embeddings) > 10:
            self._run_immune_update()

        self.episode_count  += 1
        idx = np.random.randint(0, len(self.rs_labeled))
        self.sequence        = self.rs_labeled[idx]
        self.item_idx        = 0
        self.packed_items    = []
        self.hold_buffer     = []
        self.used_volume     = 0.0
        self.violations      = 0
        self.total_packed    = 0
        self.all_embeddings  = []
        self.all_labels      = []

        # EMS 초기화
        self.ems_manager.reset()

        return self._get_obs(), {}

    def _run_immune_update(self):
        try:
            all_emb = torch.stack(self.all_embeddings)
            all_lbl = torch.tensor(self.all_labels, dtype=torch.float)
            forbidden_emb = all_emb[all_lbl == 0]
            self_emb      = all_emb[all_lbl == 1]

            if len(forbidden_emb) < 5 or len(self_emb) < 5:
                return

            self.nsa.generate_detectors(self_emb, forbidden_emb)
            self.nsa_initialized = True

            affinities = self.nsa.affinity(all_emb, all_lbl)
            forbidden_mask = affinities > affinities.median()
            new_detectors = self.csa.clone_and_mutate(
                self.nsa.detectors, affinities, forbidden_mask
            )
            self.nsa.detectors = new_detectors
            self.csa.update_gat_weights(self.gat_model, affinities)
        except:
            pass

    def _get_gat_embedding(self):
        if self.item_idx < len(self.sequence):
            seq_slice = self.sequence[:max(2, self.item_idx+1)]
            graph = build_graph_safe(seq_slice)
            x = graph.x.to(self.device)
            edge_index = graph.edge_index.to(self.device)
            with torch.no_grad():
                emb = self.gat_model(x, edge_index)
            return emb[-1].cpu()
        return torch.zeros(64)

    def _get_constraint_mask(self, item):
        if not self.nsa_initialized or self.nsa.detectors is None:
            return False
        emb = self._get_gat_embedding().unsqueeze(0)
        violations, _ = self.nsa.detect(emb)
        if violations is not None and violations[0].item():
            return True
        return False

    def _check_violation(self, item):
        for packed in self.packed_items:
            if check_violation(item["category"], packed["category"]):
                return True
        return False

    def _collect_embedding(self, label):
        try:
            emb = self._get_gat_embedding()
            self.all_embeddings.append(emb)
            self.all_labels.append(label)
        except:
            pass

    def _get_obs(self):
        # EMS Height Map
        hm = self.ems_manager.get_heightmap(
            self.bin_W, self.bin_D, self.bin_H
        ).flatten()

        if self.item_idx < len(self.sequence):
            item = self.sequence[self.item_idx]
            item_feat = np.array([
                item["size"][0] / self.bin_W,
                item["size"][1] / self.bin_H,
                item["size"][2] / self.bin_D,
                item["category_id"] / 3.0,
                item["hazard_level"]
            ], dtype=np.float32)
        else:
            item_feat = np.zeros(5, dtype=np.float32)

        gat_feat = self._get_gat_embedding().numpy()

        buf_feat = np.zeros(5 * self.hold_buffer_size, dtype=np.float32)
        for i, buf_item in enumerate(self.hold_buffer[:self.hold_buffer_size]):
            buf_feat[i*5:(i+1)*5] = [
                buf_item["size"][0] / self.bin_W,
                buf_item["size"][1] / self.bin_H,
                buf_item["size"][2] / self.bin_D,
                buf_item["category_id"] / 3.0,
                buf_item["hazard_level"]
            ]

        return np.concatenate([hm, item_feat, gat_feat, buf_feat]).astype(np.float32)

    def _try_place(self, item):
        """GOPT EMS 방식으로 배치 시도"""
        pos = self.ems_manager.can_place(item)
        if pos is None:
            return False, 0.0
        item_vol = self.ems_manager.place(item, pos)
        self.used_volume  += item_vol
        self.total_packed += 1
        self.packed_items.append(item)
        return True, item_vol / self.bin_volume

    def step(self, action):
        terminated = False
        truncated  = False
        reward     = 0.0
        info       = {}

        if self.item_idx >= len(self.sequence):
            terminated = True
            return self._get_obs(), reward, terminated, truncated, info

        current_item = self.sequence[self.item_idx]
        nsa_violation = self._get_constraint_mask(current_item)

        if action == 0:  # Pack
            if nsa_violation or self._check_violation(current_item):
                reward = -1.0
                self.violations += 1
                self._collect_embedding(0)
            else:
                placed, uti_gain = self._try_place(current_item)
                if not placed:
                    reward = -0.2  # 공간 없음
                else:
                    reward = 0.6 * uti_gain + 0.04
                self._collect_embedding(1)
            self.item_idx += 1

        elif action == 1:  # Hold (HEN 방식)
            if len(self.hold_buffer) < self.hold_buffer_size:
                self.hold_buffer.append(current_item)
                reward = -0.05
            else:
                if nsa_violation or self._check_violation(current_item):
                    reward = -1.0
                    self.violations += 1
                    self._collect_embedding(0)
                else:
                    placed, uti_gain = self._try_place(current_item)
                    if not placed:
                        reward = -0.2
                    else:
                        reward = 0.6 * uti_gain
                    self._collect_embedding(1)
            self.item_idx += 1

        elif action == 2:  # Close (새 박스)
            reward = -0.1
            self.ems_manager.reset()
            self.packed_items = []

        elif action >= 3:  # Unhold (HEN 방식)
            buf_idx = action - 3
            if buf_idx < len(self.hold_buffer):
                buf_item = self.hold_buffer[buf_idx]
                if self._check_violation(buf_item):
                    reward = -1.0
                    self.violations += 1
                else:
                    placed, uti_gain = self._try_place(buf_item)
                    if not placed:
                        reward = -0.2
                    else:
                        reward = 0.6 * uti_gain + 0.08
                self.hold_buffer.pop(buf_idx)
            else:
                reward = -0.1

        if self.item_idx >= len(self.sequence):
            terminated = True
            final_uti = self.used_volume / self.bin_volume
            violation_rate = self.violations / max(1, self.total_packed)
            compliance_rate = 1.0 - violation_rate
            reward += 0.6 * final_uti + 0.4 * compliance_rate
            info["uti"]             = round(final_uti, 4)
            info["violations"]      = self.violations
            info["compliance_rate"] = round(compliance_rate, 4)
            info["total_packed"]    = self.total_packed

        return self._get_obs(), reward, terminated, truncated, info
