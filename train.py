import torch

from tokenizers import Tokenizer

from trainer import train
from KANamav2 import Llama3_2Transformer, ModelArgs


print("... Loading Tokenizer")
tokenizer = Tokenizer.from_file("custom_tokenizer.json")

ModelArgs.vocab_size = tokenizer.get_vocab_size()
ModelArgs.padding_idx = tokenizer.token_to_id("[PAD]") if "[PAD]" in tokenizer.get_vocab() else None
ModelArgs.max_batch_size = 2
ModelArgs.max_seq_len = 64

print("... Loading Model")
model = Llama3_2Transformer(ModelArgs)
print(model)


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("... Loading Dataset")
with open("/Users/gokdenizgulmez/Desktop/KANama/tiny-shakespear.txt", "r") as file:
    dataset = file.read()

# Tokenizing the dataset
dataset = tokenizer.encode(dataset)
# Converting tokens to tensor and adding batch dimension
data = torch.LongTensor(dataset.ids).unsqueeze(0)

# Splitting data into training and validation sets
n = int(0.9 * len(data[0]))
train_data = data[:, :n]  # Ensure correct slicing
val_data = data[:, n:]    # Ensure correct slicing

print("Training")
train(model=model, optimizer=optimizer, train_data=train_data, val_data=val_data, save=True, max_steps=10000, loss_intervall=100, eval_interval=1000)