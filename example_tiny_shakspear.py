from typing import Tuple

import torch

from tokenizers import Tokenizer

from trainer.SFTTrainer import train
from model.args import MOEModelArgs
from model.KANamav5 import KANamav5

from utils import load_model


print("[LOADING TOKENIZER]")
tokenizer = Tokenizer.from_file("custom_tokenizer.json")


MOEModelArgs.vocab_size = tokenizer.get_vocab_size()
MOEModelArgs.pad_id = tokenizer.token_to_id("[PAD]") if "[PAD]" in tokenizer.get_vocab() else None
MOEModelArgs.max_batch_size = 4
MOEModelArgs.max_seq_len = 20
MOEModelArgs.dim = 64
# MOEModelArgs.use_kan = False
# MOEModelArgs.use_softmax_temp_proj = False


print("[LOADING DATASET]")
with open("tiny-shakespear.txt", "r") as file:
    dataset = file.read()

dataset = tokenizer.encode(dataset)
data = torch.LongTensor(dataset.ids).unsqueeze(0)

n = int(0.9 * len(data[0]))
train_data = data[:, :n]
val_data = data[:, n:]


print("[LOADING MODEL]")
model = KANamav5(MOEModelArgs)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
new_model = train(model=model, optimizer=optimizer, train_data=train_data, val_data=val_data, save=True, max_steps=50000, loss_interval=10, eval_interval=2000)


# line_19826 = """ROMEO:\nI pay thy poverty, """
# first_tokens = tokenizer.encode(line_19826)
# input_tokens = torch.LongTensor(first_tokens.ids).unsqueeze(0)

# logits, loss = model(input_tokens)

# print(logits[:, -1])

# probabilities = torch.softmax(logits[:, -1], dim=-1)
# print(probabilities)

# next_token = torch.multinomial(probabilities, num_samples=1).squeeze()
# print(next_token)

# next_token = [next_token.tolist()]

# print(next_token)
# print(detokenize(next_token))

# def inference(model: torch.nn.Module, tokens, max_new_tokens: int):
#     for _ in range(max_new_tokens):
#         tokens_conditioned = tokens[:, -MOEModelArgs.max_seq_len:]
#         logits, _ = model(tokens_conditioned)
#         probabilities = torch.softmax(logits[:, -1], dim=-1)
#         next_token = torch.multinomial(probabilities, num_samples=1)
#         tokens = torch.cat((tokens, next_token), dim=1)
#         print(tokenizer.decode(next_token.squeeze(dim=1).tolist(), skip_special_tokens=True), end="", flush=False)


# inference(model, input_tokens, 100)