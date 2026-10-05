
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
    겹침 체크 + 중력 적용
    """
    def __init__(self, bin_W, bin_H, bin_D):
        self.bin_W = bin_W
        self.bin_H = bin_H
        self.bin_D = bin_D
        self.reset()

    def reset(self):
        self.ems = [(0, 0, 0)]
        self.placed_items = []
        self.used_volume  = 0

    def _is_overlapping(self, x, y, z, w, h, d):
        for px, py, pz, pw, ph, pd in self.placed_items:
            if (x < px + pw and x + w > px and
                y < py + ph and y + h > py and
                z < pz + pd and z + d > pz):
                return True
        return False

    def _get_support_height(self, x, z, w, d):
        max_h = 0
        for px, py, pz, pw, ph, pd in self.placed_items:
            if (px < x + w and px + pw > x and
                pz < z + d and pz + pd > z):
                max_h = max(max_h, py + ph)
        return max_h

    def can_place(self, item):
        w, h, d = item["size"]
        if w > self.bin_W or h > self.bin_H or d > self.bin_D:
            return None

        best_ep    = None
        best_score = float("inf")

        for ex, ey, ez in self.ems:
            if ex + w > self.bin_W or ez + d > self.bin_D:
                continue

            floor_h = self._get_support_height(ex, ez, w, d)
            place_y = floor_h

            if place_y + h > self.bin_H:
                continue

            if self._is_overlapping(ex, place_y, ez, w, h, d):
                continue

            score = place_y * 1000 + ex * 10 + ez
            if score < best_score:
                best_score = score
                best_ep = (ex, place_y, ez)

        return best_ep

    def place(self, item, pos):
        w, h, d = item["size"]
        x, y, z = pos
        self.placed_items.append((x, y, z, w, h, d))
        item_vol = w * h * d
        self.used_volume += item_vol

        new_eps = [
            (x + w, y, z),
            (x, y + h, z),
            (x, y, z + d),
        ]
        for ep in new_eps:
            ex, ey, ez = ep
            if (0 <= ex <= self.bin_W and
                0 <= ey <= self.bin_H and
                0 <= ez <= self.bin_D):
                self.ems.append(ep)

        self.ems = list(set(self.ems))
        return item_vol

    def get_heightmap(self, bin_W, bin_D, bin_H):
        hm = np.zeros((bin_W, bin_D), dtype=np.float32)
        for px, py, pz, pw, ph, pd in self.placed_items:
            hm[px:px+pw, pz:pz+pd] = np.maximum(
                hm[px:px+pw, pz:pz+pd],
                (py + ph) / bin_H
            )
        return hm


class PackingEnvWithHold(gym.Env):
    """
    GOPT EMS + Transformer + HEN Hold 통합 환경
    Layer 1: GAT 임베딩 → 관찰 공간
    Layer 2: NSA Constraint Mask + CSA → GAT 업데이트
    Layer 3: Transformer + PPO → Pack/Hold/Close
    배치: GOPT EMS 방식 (겹침 체크 + 중력)
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

        self.ems_manager = EMSManager(bin_size[0], bin_size[1], bin_size[2])

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
        self.violations      = 0
        self.total_packed    = 0
        self.all_embeddings  = []
        self.all_labels      = []
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
                pos = self.ems_manager.can_place(current_item)
                if pos is None:
                    reward = -0.2
                else:
                    item_vol = self.ems_manager.place(current_item, pos)
                    uti_gain = item_vol / self.bin_volume
                    reward = 0.6 * uti_gain + 0.04
                    self.total_packed += 1
                    self.packed_items.append(current_item)
                self._collect_embedding(1)
            self.item_idx += 1

        elif action == 1:  # Hold
            if len(self.hold_buffer) < self.hold_buffer_size:
                self.hold_buffer.append(current_item)
                reward = -0.05
            else:
                if nsa_violation or self._check_violation(current_item):
                    reward = -1.0
                    self.violations += 1
                    self._collect_embedding(0)
                else:
                    pos = self.ems_manager.can_place(current_item)
                    if pos is None:
                        reward = -0.2
                    else:
                        item_vol = self.ems_manager.place(current_item, pos)
                        uti_gain = item_vol / self.bin_volume
                        reward = 0.6 * uti_gain
                        self.total_packed += 1
                        self.packed_items.append(current_item)
                    self._collect_embedding(1)
            self.item_idx += 1

        elif action == 2:  # Close
            reward = -0.1
            self.ems_manager.reset()
            self.packed_items = []

        elif action >= 3:  # Unhold
            buf_idx = action - 3
            if buf_idx < len(self.hold_buffer):
                buf_item = self.hold_buffer[buf_idx]
                if self._check_violation(buf_item):
                    reward = -1.0
                    self.violations += 1
                else:
                    pos = self.ems_manager.can_place(buf_item)
                    if pos is None:
                        reward = -0.2
                    else:
                        item_vol = self.ems_manager.place(buf_item, pos)
                        uti_gain = item_vol / self.bin_volume
                        reward = 0.6 * uti_gain + 0.08
                        self.total_packed += 1
                        self.packed_items.append(buf_item)
                self.hold_buffer.pop(buf_idx)
            else:
                reward = -0.1

        if self.item_idx >= len(self.sequence):
            terminated = True
            final_uti = self.ems_manager.used_volume / self.bin_volume
            final_uti = min(final_uti, 1.0)  # 안전 클리핑
            violation_rate = self.violations / max(1, self.total_packed)
            compliance_rate = 1.0 - violation_rate
            reward += 0.6 * final_uti + 0.4 * compliance_rate
            info["uti"]             = round(final_uti, 4)
            info["violations"]      = self.violations
            info["compliance_rate"] = round(compliance_rate, 4)
            info["total_packed"]    = self.total_packed

        return self._get_obs(), reward, terminated, truncated, info
