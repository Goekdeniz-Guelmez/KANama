import torch

from trainer.SFTTrainer import train
from model.args import MOEModelArgs
from model.KANamav5 import KANamav5

from utils import load_model


with open("tiny-shakespear.txt", "r") as file:
    dataset = file.read()
chars = sorted(list(set(dataset)))
chars_dataset = sorted(list(set(dataset)))
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
tokenize = lambda s: [stoi[c] for c in s]
detokenize = lambda l: ''.join([itos[i] for i in l])


MOEModelArgs.vocab_size = len(chars_dataset)
MOEModelArgs.pad_id = tokenize("#")[0]
MOEModelArgs.max_batch_size = 4
MOEModelArgs.max_seq_len = 20
MOEModelArgs.dim = 64
# MOEModelArgs.use_kan = False
# MOEModelArgs.use_softmax_temp_proj = False


data = torch.tensor(tokenize(dataset), dtype=torch.long)
n = int(0.9 * len(data))


train_data = data[:n].unsqueeze(dim=-2)
val_data = data[n:].unsqueeze(dim=-2)


model = KANamav5(MOEModelArgs)
load_model(model, "/Users/gokdenizgulmez/Desktop/meine-repos/KANama/trained_KANamev3_model.pth")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
new_model = train(model=model, optimizer=optimizer, train_data=train_data, val_data=val_data, save=True, max_steps=10000, loss_interval=100, eval_interval=2000)
