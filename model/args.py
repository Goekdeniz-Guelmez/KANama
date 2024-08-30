from typing import Optional
from dataclasses import dataclass

@dataclass
class ModelArgs:
    vocab_size: int = -1
    pad_id: int = -1
    eos_id: int = pad_id

    dim: int = 32
    n_layers: int = 8

    n_heads: int = 8
    n_kv_heads: Optional[int] = None

    use_kan: bool = True
    use_softmax_temp_proj: bool = True
    train_softmax_temp: bool = True
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None

    rms_norm_eps: float = 1e-5

    rope_theta: float = 500000
    use_scaled_rope: bool = False

    max_batch_size: int = 32
    max_seq_len: int = 2048

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        assert self.n_kv_heads <= self.n_heads
        assert self.n_heads % self.n_kv_heads == 0
        assert self.dim % self.n_heads == 0



@dataclass
class MOEModelArgs:
    vocab_size: int = -1
    pad_id: int = -1
    eos_id: int = pad_id

    dim: int = 32
    n_layers: int = 8

    n_heads: int = 8
    n_kv_heads: Optional[int] = None

    use_kan: bool = True
    train_softmax_temp: bool = True
    use_softmax_temp_proj: bool = True
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None

    rms_norm_eps: float = 1e-5

    rope_theta: float = 500000
    use_scaled_rope: bool = False

    max_batch_size: int = 32
    max_seq_len: int = 2048

    num_experts: int = 2
    num_experts_per_tok: int = 1

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        assert self.n_kv_heads <= self.n_heads
        assert self.n_heads % self.n_kv_heads == 0
        assert self.dim % self.n_heads == 0