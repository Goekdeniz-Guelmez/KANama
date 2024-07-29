import torch

from tokenizers import Tokenizer

from trainer import train
from inference import generate
from KANamav3 import KANamev3, ModelArgs


print("... Loading Tokenizer")
tokenizer = Tokenizer.from_file("custom_tokenizer.json")

ModelArgs.vocab_size = tokenizer.get_vocab_size()
ModelArgs.pad_id = tokenizer.token_to_id("[PAD]") if "[PAD]" in tokenizer.get_vocab() else None
ModelArgs.max_batch_size = 4
ModelArgs.max_seq_len = 64

print("... Loading Model")
model = KANamev3(ModelArgs)
print(model)








print("FIRST TEST")
line_19826 = """ROMEO:\nI pay thy poverty, """
first_tokens = tokenizer.encode(line_19826)
first_input = torch.LongTensor(first_tokens.ids).unsqueeze(0)
output = generate(model=model, prompt_tokens=first_input)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)






optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

with open("/Users/gokdenizgulmez/Desktop/KANama/tiny-shakespear.txt", "r") as file:
    dataset = file.read()

dataset = tokenizer.encode(dataset)
data = torch.LongTensor(dataset.ids).unsqueeze(0)

n = int(0.9 * len(data[0]))
train_data = data[:, :n]
val_data = data[:, n:]

model = train(model=model, optimizer=optimizer, train_data=train_data, val_data=val_data, save=True, max_steps=10000, loss_intervall=100, eval_interval=2000)







print("LAST TEST")
line_19826 = """ROMEO:\nI pay thy poverty, """
first_tokens = tokenizer.encode(line_19826)
first_input = torch.LongTensor(first_tokens.ids).unsqueeze(0)
output = generate(model=model, prompt_tokens=first_input)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)