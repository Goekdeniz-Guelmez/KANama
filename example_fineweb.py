from tqdm import tqdm
import json

import torch
import math

from transformers import AutoTokenizer

from trainer.SFTTrainer import train
from model.args import MOEModelArgs
from model.KANaMoEv1 import KANamav5

from utils import load_model, quick_inference

# device = 'cuda' if torch.cuda.is_available() else 'mps' if ( torch.backends.mps.is_available() and torch.backends.mps.is_built() ) else 'cpu'

device = "cpu"

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
tokenizer = AutoTokenizer.from_pretrained("Doctor-Shotgun/TinyLlama-1.1B-32k")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


MOEModelArgs.vocab_size = tokenizer.vocab_size
MOEModelArgs.pad_id = tokenizer.pad_token_id
MOEModelArgs.num_experts_per_tok = 2
MOEModelArgs.max_batch_size = 100
MOEModelArgs.max_seq_len = 128
MOEModelArgs.num_experts = 8
MOEModelArgs.n_layers = 16
MOEModelArgs.dim = 256
# MOEModelArgs.use_kan = False
# MOEModelArgs.use_softmax_temp_proj = False


# List to hold all tokenized data
tokenized_data = []

# Estimating the total number of lines
total_lines = sum(1 for line in open("datasets/tiny-fineweb-100_128k.jsonl", 'r'))

with open("datasets/tiny-fineweb-100_128k.jsonl", 'r') as file:
    for line in tqdm(file, desc="Processing JSON lines", total=total_lines):
        data = json.loads(line)
        # Tokenize the current line without truncation or padding
        tokens = tokenizer(data['text'], truncation=False, padding=False)['input_ids']
        # Add the tokenized IDs to the list
        tokenized_data.append(torch.tensor(tokens))

# Concatenate all tokenized sequences into a single tensor
data = torch.cat(tokenized_data, dim=0).unsqueeze(0)  # unsqueeze to add a batch dimension

# Define train/val split based on the size of the dataset
n = int(0.9 * data.size(1))  # data.size(1) because sequences are concatenated along dimension 1
train_data = data[:, :n].to(device)
val_data = data[:, n:].to(device)

print("\n[LOADING MODEL]\n")
model = KANamav5(MOEModelArgs, device=device)

# Starting sequence (as tokens)
initial_text = "Once upon a time"
initial_tokens = tokenizer(initial_text, return_tensors="pt").input_ids.to(device)

# Perform inference
generated_tokens, generated_text = quick_inference(model, initial_tokens, max_new_tokens=50, tokenizer=tokenizer)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

print("\n[TRAINING MODEL]\n")
new_model = train(
    model=model,
    optimizer=optimizer,
    train_data=train_data,
    val_data=val_data,
    scheduler=scheduler,
    save_model_name="KANama-medium",
    max_steps=100,
    loss_interval=5,
    eval_interval=50,
    device=device
)


generated_tokens, generated_text = quick_inference(model, initial_tokens, max_new_tokens=50, tokenizer=tokenizer)