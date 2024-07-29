from typing import List

import torch
import torch.nn.functional as F


@torch.inference_mode()
def generate(model, prompt_tokens: torch.Tensor, max_gen_len: int = 10) -> List[List[int]]:
    args = model.args
    bsz = len(prompt_tokens)
    assert bsz <= args.max_batch_size, (bsz, args.max_batch_size)

    max_prompt_len = max(len(t) for t in prompt_tokens)
    assert max_prompt_len <= args.max_seq_len
    total_len = min(args.max_seq_len, max_gen_len + max_prompt_len)

    pad_id = args.pad_id
    tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long)
    for k, t in enumerate(prompt_tokens):
        tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long)

    prev_pos = 0
    eos_reached = torch.tensor([False] * bsz)

    for cur_pos in range(max_prompt_len, total_len):
        logits, _ = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        probs = torch.softmax(logits[:, -1], dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze()
        tokens[:, cur_pos] = next_token
        eos_reached |= next_token == args.eos_id
        prev_pos = cur_pos
        if all(eos_reached):
            break

    return [t.tolist() for t in tokens]