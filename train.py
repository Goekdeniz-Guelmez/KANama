import torch

from tokenizers import Tokenizer

from trainer import train
from llama3_2 import Llama3_2Transformer, ModelArgs


print("... Loading Tokenizer")
tokenizer = Tokenizer.from_file("custom_tokenizer.json")

ModelArgs.vocab_size = tokenizer.get_vocab_size()
ModelArgs.padding_idx = tokenizer.token_to_id("[PAD]") if "[PAD]" in tokenizer.get_vocab() else None

print("... Loading Model")
model = Llama3_2Transformer(ModelArgs)
print(model)


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("... Loading Dataset")
with open("/Users/gokdenizgulmez/Desktop/Inside-Llama/tiny-shakespear.txt", "r") as file:
    dataset = file.read()

# Tokenizing the dataset
dataset = tokenizer.encode(dataset)
# Converting tokens to tensor and adding batch dimension
data = torch.LongTensor(dataset.ids).unsqueeze(0)

# Splitting data into training and validation sets
n = int(0.9 * len(data[0]))
train_data = data[:, :n]  # Ensure correct slicing
val_data = data[:, n:]    # Ensure correct slicing
print("Dataset loaded and split into training and validation sets.")

print("Training")
train(model=model, optimizer=optimizer, train_data=train_data, val_data=val_data)