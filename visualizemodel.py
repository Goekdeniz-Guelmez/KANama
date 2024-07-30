from tokenizers import Tokenizer

from KANamav3 import KANamav3, ModelArgs

from utils import load_model, visualize_KANama, save_model_parameters_to_file


print("... Loading Tokenizer")
tokenizer = Tokenizer.from_file("custom_tokenizer.json")

ModelArgs.vocab_size = tokenizer.get_vocab_size()
ModelArgs.pad_id = tokenizer.token_to_id("[PAD]") if "[PAD]" in tokenizer.get_vocab() else None
ModelArgs.max_batch_size = 4
ModelArgs.max_seq_len = 64


model = KANamav3(ModelArgs)
model = load_model(model=model, file_name="/Users/gokdenizgulmez/Desktop/KANama/trained_KANamav3_model.pth")

save_model_parameters_to_file(model, "trained_KANamav3_model.txt")
visualize_KANama("trained_KANamav3_model.txt", "trained_KANamav3_model.png", "trained_KANamav3_model/")