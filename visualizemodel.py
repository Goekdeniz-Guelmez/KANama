from tokenizers import Tokenizer

from KANamav2 import KANamev2, ModelArgs

from utils import load_model, visualize_KANama, save_model_parameters_to_file


print("... Loading Tokenizer")
tokenizer = Tokenizer.from_file("custom_tokenizer.json")

ModelArgs.vocab_size = tokenizer.get_vocab_size()
ModelArgs.pad_id = tokenizer.token_to_id("[PAD]") if "[PAD]" in tokenizer.get_vocab() else None
ModelArgs.max_batch_size = 4
ModelArgs.max_seq_len = 64


model = KANamev2(ModelArgs)
model = load_model(model=model)

save_model_parameters_to_file(model, "trained_KANama_model.txt")
visualize_KANama("trained_KANama_model.txt", "trained_KANama_model.png", "trained_KANama_model/")