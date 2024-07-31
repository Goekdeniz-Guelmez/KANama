import torch
import torch.nn as nn

from tokenizers import Tokenizer

from trainer.trainer import train
from model.args import ModelArgs
from model.KANamav4 import KANamav4
from model.KANamav3 import KANamav3
from model.KANamav2 import KANamav2
from model.KANamav1 import KANamav1


ModelArgs.vocab_size = 30
ModelArgs.pad_id = 0
ModelArgs.max_batch_size = 4
ModelArgs.max_seq_len = 64


input_data = torch.tensor([[0, 1, 4, 12, 9]], dtype=torch.long)


print("Creating models")
try:
    KANamav4 = KANamav4(ModelArgs)
    KANamav3 = KANamav3(ModelArgs)
    KANamav2 = KANamav2(ModelArgs)
    KANamav1 = KANamav1(ModelArgs)

    if KANamav1 != nn.Module and KANamav2 != nn.Module and KANamav3 != nn.Module and KANamav4 != nn.Module:
        print("Succesfully created models!")
    else:
        print("Created models are not nn.Modules!")
except Exception as e:
    print(f"Erro while creating model: {e}")









# print("FIRST TEST")

# output = generate(model=model, prompt_tokens=first_input)
# generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
# print(generated_text)






# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# with open("/Users/gokdenizgulmez/Desktop/KANama/tiny-shakespear.txt", "r") as file:
#     dataset = file.read()

# dataset = tokenizer.encode(dataset)
# data = torch.LongTensor(dataset.ids).unsqueeze(0)

# n = int(0.9 * len(data[0]))
# train_data = data[:, :n]
# val_data = data[:, n:]

# model = train(model=model, optimizer=optimizer, train_data=train_data, val_data=val_data, save=True, max_steps=100000, loss_interval=100, eval_interval=2000)







# print("LAST TEST")
# line_19826 = """ROMEO:\nI pay thy poverty, """
# first_tokens = tokenizer.encode(line_19826)
# first_input = torch.LongTensor(first_tokens.ids).unsqueeze(0)
# output = generate(model=model, prompt_tokens=first_input)
# generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
# print(generated_text)