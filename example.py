import torch

from KANama.trainer.trainer import train
from KANama.model.args import ModelArgs
from KANama.model.KANamav4 import KANamav4


ModelArgs.vocab_size = 30
ModelArgs.pad_id = 0
ModelArgs.max_batch_size = 4
ModelArgs.max_seq_len = 20


train_data = torch.tensor([[25, 1, 4, 12, 9, 7, 1, 4, 12, 9, 4, 1, 4, 22, 9, 13, 26, 24, 12, 9, 0]], dtype=torch.long)
val_data = torch.tensor([[25, 1, 4, 12, 9, 7, 1, 4, 12, 9, 4, 1, 4, 22, 9, 13, 26, 24, 12, 9, 0]], dtype=torch.long)


model = KANamav4(ModelArgs)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
new_model = train(model=model, optimizer=optimizer, train_data=train_data, val_data=val_data, save=False, max_steps=10, loss_interval=2, eval_interval=5)