from typing import Optional, Tuple
from dataclasses import dataclass

import torch.nn as nn
import torch


@dataclass
class ModelArgs:
    dim: int = 12
    num_layers: int = 1
    num_heads: int = 6
    num_kv_heads: Optional[int] = None




class Llama3_1(nn.Module):