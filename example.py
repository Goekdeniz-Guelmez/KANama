import torch

# from KANama.trainer.SFTTrainer import train
# from KANama.model.args import ModelArgs
# from KANama.model.KANamav4 import KANamav4

from model.handler import save_pretrained

from trainer.SFTTrainer import train
from model.args import ModelArgs, MOEModelArgs as ModelArgs
from model.KANaMoEv1 import KANaMoEv1
from model.KANamav4 import KANamav4


ModelArgs.vocab_size = 30
ModelArgs.pad_id = 0
ModelArgs.max_batch_size = 4
ModelArgs.max_seq_len = 20
# ModelArgs.use_kan = False
# ModelArgs.use_softmax_temp_proj = False


# train_data = torch.tensor([[25, 1, 4, 12, 9, 7, 1, 4, 12, 9, 4, 1, 4, 22, 9, 13, 26, 24, 12, 9, 0]], dtype=torch.long) # Must be a 3 dimansional Tensor [B, max_seq_len, tokens]
# val_data = torch.tensor([[25, 1, 4, 12, 9, 7, 1, 4, 12, 9, 4, 1, 4, 22, 9, 13, 26, 24, 12, 9, 0]], dtype=torch.long) # Here too


# model = KANamav4(ModelArgs)
model = KANaMoEv1(ModelArgs)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# new_model = train(model=model, optimizer=optimizer, train_data=train_data, val_data=val_data, save=False, max_steps=100, loss_interval=2, eval_interval=50)

save_pretrained(path_to_save="/Users/gokdenizgulmez/Desktop/meine-repos/KANama/example.py", model=model)