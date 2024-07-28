from typing import Optional, Tuple
from dataclasses import dataclass

import torch.nn as nn
import torch


@dataclass
class ModelArgs:
    vocab_size: int = -1
    dim: int = 12
    num_layers: int = 1
    num_heads: int = 6
    num_kv_heads: Optional[int] = None




class Llama3_1(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.embeddings = nn.Embedding()

        self.layers = nn.ModuleList

        self.norm = nn.BatchNorm1d(args.dim)
        self.lm_head = nn.Linear(in_features=args.dim, out_features=args.vocab_size, bias=False)

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        B, L = x.shape

        embedds = self.embeddings(x)

        freqs_cis = torch.Tensor

        mask = None
        if L >= 1:
            print("Implementieren")

        for layer in enumerate()
