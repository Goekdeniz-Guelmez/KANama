from typing import Tuple
import json

import torch
import math

from transformers import AutoTokenizer

from trainer.SFTTrainer import train
from model.args import MOEModelArgs
from model.KANamav5 import KANamav5

from utils import load_model


def lr_lambda(current_step: int, max_steps: int=50000, warmup_steps: int=40, lr_scheduler_type: str="cosine"):
    if current_step < warmup_steps:
        return current_step / warmup_steps

    annealing_steps = max_steps - warmup_steps

    if annealing_steps <= 0:
        annealing_steps = 1

    progress = (current_step - warmup_steps) / annealing_steps
    if lr_scheduler_type == "cosine":
        new_learning_rate = 0.5 * (1.0 + math.cos(math.pi * progress))
    elif lr_scheduler_type == "sinus":
        new_learning_rate = 0.5 * (1.0 + math.sin(math.pi * progress))
    else:
        new_learning_rate = 1.0
    return new_learning_rate


print("[LOADING TOKENIZER]")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


MOEModelArgs.vocab_size = tokenizer.vocab_size
MOEModelArgs.pad_id = tokenizer.pad_token_id
MOEModelArgs.max_batch_size = 4
MOEModelArgs.max_seq_len = 20
MOEModelArgs.n_layers = 12
MOEModelArgs.dim = 64
# MOEModelArgs.use_kan = False
# MOEModelArgs.use_softmax_temp_proj = False


# List to hold all text data
texts = []

# Read the JSONL file
with open("datasets/tiny-fineweb-100_128k.jsonl", 'r') as file:
    for line in file:
        # Parse the JSON line and extract the text
        data = json.loads(line)
        texts.append(data['text'])

tokens = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=MOEModelArgs.max_seq_len + 1)
data = tokens['input_ids'].squeeze()

n = int(0.9 * data.size(0))
train_data = data[:n]
val_data = data[n:]



print("[LOADING MODEL]")
model = KANamav5(MOEModelArgs, device="cpu")
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

new_model = train(
    model=model,
    optimizer=optimizer,
    train_data=train_data,
    val_data=val_data,
    scheduler=scheduler,
    save=True,
    max_steps=50000,
    loss_interval=10,
    eval_interval=2000,
    device="cpu"
)


# line_19826 = """ROMEO:\nI pay thy poverty, """
# first_tokens = tokenizer.encode(line_19826)
# input_tokens = torch.LongTensor(first_tokens.ids).unsqueeze(0)

# def inference(model: torch.nn.Module, tokens, max_new_tokens: int):
#     for _ in range(max_new_tokens):
#         tokens_conditioned = tokens[:, -MOEModelArgs.max_seq_len:]
#         logits, _ = model(tokens_conditioned)
#         probabilities = torch.softmax(logits[:, -1], dim=-1)
#         next_token = torch.multinomial(probabilities, num_samples=1)
#         tokens = torch.cat((tokens, next_token), dim=1)
#         print(tokenizer.decode(next_token.squeeze(dim=1).tolist(), skip_special_tokens=True), end="", flush=False)


# inference(model, input_tokens, 100)