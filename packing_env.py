
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from torch_geometric.data import Data
import torch
from gat_model import check_violation, MIXED_LOAD_RULES

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


class PackingEnvWithHold(gym.Env):
    def __init__(self, gat_model, rs_labeled, device,
                 bin_size=(10,10,10), max_items=100, hold_buffer_size=3):
        super().__init__()
        self.gat_model        = gat_model
        self.rs_labeled       = rs_labeled
        self.device           = device
        self.bin_size         = bin_size
        self.max_items        = max_items
        self.hold_buffer_size = hold_buffer_size

        self.action_space = spaces.Discrete(3 + hold_buffer_size)
        obs_dim = 100 + 5 + 64 + (5 * hold_buffer_size)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        idx = np.random.randint(0, len(self.rs_labeled))
        self.sequence      = self.rs_labeled[idx]
        self.item_idx      = 0
        self.packed_items  = []
        self.hold_buffer   = []
        self.height_map    = np.zeros((10, 10), dtype=np.float32)
        self.bin_volume    = self.bin_size[0] * self.bin_size[1] * self.bin_size[2]
        self.used_volume   = 0
        self.violations    = 0
        self.total_packed  = 0
        return self._get_obs(), {}

    def _get_obs(self):
        hm = self.height_map.flatten()

        if self.item_idx < len(self.sequence):
            item = self.sequence[self.item_idx]
            item_feat = np.array([
                item["size"][0], item["size"][1], item["size"][2],
                item["category_id"], item["hazard_level"]
            ], dtype=np.float32)
        else:
            item_feat = np.zeros(5, dtype=np.float32)

        if self.item_idx < len(self.sequence):
            seq_slice = self.sequence[:max(2, self.item_idx+1)]
            graph = build_graph_safe(seq_slice)
            x = graph.x.to(self.device)
            edge_index = graph.edge_index.to(self.device)
            with torch.no_grad():
                emb = self.gat_model(x, edge_index)
            gat_feat = emb[-1].cpu().numpy()
        else:
            gat_feat = np.zeros(64, dtype=np.float32)

        buf_feat = np.zeros(5 * self.hold_buffer_size, dtype=np.float32)
        for i, buf_item in enumerate(self.hold_buffer[:self.hold_buffer_size]):
            buf_feat[i*5:(i+1)*5] = [
                buf_item["size"][0], buf_item["size"][1], buf_item["size"][2],
                buf_item["category_id"], buf_item["hazard_level"]
            ]

        return np.concatenate([hm, item_feat, gat_feat, buf_feat]).astype(np.float32)

    def _check_violation(self, item):
        for packed in self.packed_items:
            if check_violation(item["category"], packed["category"]):
                return True
        return False

    def _pack_item(self, item):
        w, h, d = item["size"]
        item_vol = w * h * d
        self.used_volume  += item_vol
        self.total_packed += 1
        self.height_map[:min(w,10), :min(d,10)] += h / 10.0
        self.height_map = np.clip(self.height_map, 0, 1)
        self.packed_items.append(item)
        return item_vol / self.bin_volume

    def step(self, action):
        terminated = False
        truncated  = False
        reward     = 0.0
        info       = {}

        if self.item_idx >= len(self.sequence):
            terminated = True
            return self._get_obs(), reward, terminated, truncated, info

        current_item = self.sequence[self.item_idx]

        if action == 0:
            if self._check_violation(current_item):
                reward = -1.0
                self.violations += 1
            else:
                uti_gain = self._pack_item(current_item)
                reward = 0.6 * uti_gain + 0.04
            self.item_idx += 1

        elif action == 1:
            if len(self.hold_buffer) < self.hold_buffer_size:
                self.hold_buffer.append(current_item)
                reward = -0.05
            else:
                if self._check_violation(current_item):
                    reward = -1.0
                    self.violations += 1
                else:
                    uti_gain = self._pack_item(current_item)
                    reward = 0.6 * uti_gain
            self.item_idx += 1

        elif action == 2:
            reward = -0.2
            self.height_map   = np.zeros((10, 10), dtype=np.float32)
            self.packed_items = []

        elif action >= 3:
            buf_idx = action - 3
            if buf_idx < len(self.hold_buffer):
                buf_item = self.hold_buffer[buf_idx]
                if self._check_violation(buf_item):
                    reward = -1.0
                    self.violations += 1
                else:
                    uti_gain = self._pack_item(buf_item)
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
