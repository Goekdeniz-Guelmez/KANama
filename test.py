import torch

from tokenizers import Tokenizer

from trainer import train
from inference import generate
from model.KANamav2 import KANamev2, ModelArgs


print("... Loading Tokenizer")
tokenizer = Tokenizer.from_file("custom_tokenizer.json")

ModelArgs.vocab_size = tokenizer.get_vocab_size()
ModelArgs.pad_id = tokenizer.token_to_id("[PAD]") if "[PAD]" in tokenizer.get_vocab() else None
ModelArgs.max_batch_size = 4
ModelArgs.max_seq_len = 64

print("... Loading Model")
model = KANamev2(ModelArgs)
print(model)

line_19826 = """ROMEO:\nI pay thy poverty, """
first_tokens = tokenizer.encode(line_19826)
first_input = torch.LongTensor(first_tokens.ids).unsqueeze(0)
output = generate(model=model, prompt_tokens=first_input)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)