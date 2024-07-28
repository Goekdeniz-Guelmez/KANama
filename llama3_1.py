from typing import Optional, Tuple
from dataclasses import dataclass

import math

import torch
import torch.nn as nn


@dataclass
class ModelArgs:
    vocab_size: int = -1
    dim: int = 12
    num_layers: int = 1
    num_heads: int = 6
    num_kv_heads: Optional[int] = None



class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    

class Llama3_1(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.embeddings = nn.Embedding()

        self.layers = nn.ModuleList

        self.norm = nn.BatchNorm1d(args.dim)
        self.lm_head = nn.Linear(in_features=args.dim, out_features=args.vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor, start_pos: int) -> torch.Tensor:
        B, L = tokens.shape
        embedds = self.embeddings(tokens)

        freqs_cis = torch.Tensor

        mask = None
        if L >= 1:
            print("Implementieren")

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)

        h = self.norm(h)
        return self.lm_head(h).float()
