import torch
from torch import nn


class AttentionHead(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        weights = torch.softmax((q @ k.transpose(-2, -1)) / (q.size(-1) ** 0.5), dim=-1)
        return weights @ v
