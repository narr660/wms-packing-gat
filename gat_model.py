
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

# 혼적 설정
CATEGORIES = ["위험물", "식품", "화학품", "일반"]
CATEGORY_MAP = {"위험물": 0, "식품": 1, "화학품": 2, "일반": 3}
HAZARD_LEVEL = {"위험물": 0.95, "화학품": 0.80, "식품": 0.70, "일반": 0.10}
MIXED_LOAD_RULES = {
    ("위험물", "식품"):   "금지",
    ("위험물", "화학품"): "금지",
    ("화학품", "식품"):   "금지",
}

def check_violation(cat1, cat2):
    pair = (cat1, cat2)
    reverse = (cat2, cat1)
    return pair in MIXED_LOAD_RULES or reverse in MIXED_LOAD_RULES

def build_graph(sequence):
    n = len(sequence)
    node_feats = []
    for item in sequence:
        feat = [
            item["size"][0],
            item["size"][1],
            item["size"][2],
            item["category_id"],
            item["hazard_level"],
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

    return Data(
        x=torch.tensor(node_feats, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float)
    )

class MixedLoadGAT(nn.Module):
    def __init__(self, in_channels=5, hidden_channels=128,
                 out_channels=64, heads=4, dropout=0.1):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels,
                              heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels,
                              heads=heads, dropout=dropout)
        self.conv3 = GATConv(hidden_channels * heads, out_channels,
                              heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.normalize(x, p=2, dim=1)
        return x

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, pos_weight=3.0):
        super().__init__()
        self.margin = margin
        self.pos_weight = pos_weight

    def forward(self, emb_i, emb_j, label):
        dist = F.pairwise_distance(emb_i, emb_j)
        weight = torch.where(
            label == 0,
            torch.tensor(self.pos_weight).to(label.device),
            torch.tensor(1.0).to(label.device)
        )
        loss_pos = label * dist.pow(2)
        loss_neg = (1 - label) * F.relu(self.margin - dist).pow(2)
        return (weight * (loss_pos + loss_neg)).mean(), dist.mean().item()

def make_pairs(sequence, embeddings, device):
    pairs_i = []
    pairs_j = []
    labels = []
    n = len(sequence)
    for i in range(n):
        for j in range(i+1, n):
            cat_i = sequence[i]["category"]
            cat_j = sequence[j]["category"]
            label = 0 if check_violation(cat_i, cat_j) else 1
            pairs_i.append(embeddings[i])
            pairs_j.append(embeddings[j])
            labels.append(label)
    emb_i = torch.stack(pairs_i)
    emb_j = torch.stack(pairs_j)
    labels = torch.tensor(labels, dtype=torch.float).to(device)
    return emb_i, emb_j, labels
