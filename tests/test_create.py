import torch
import torch.nn as nn

from model.KANamav4 import KANamav4
from model.KANamav3 import KANamav3
from model.KANamav2 import KANamav2
from model.KANamav1 import KANamav1

from model.args import ModelArgs


ModelArgs.vocab_size = 30
ModelArgs.pad_id = 0
ModelArgs.max_batch_size = 4
ModelArgs.max_seq_len = 64


print("Creating models")
try:
    KANamav4 = KANamav4(ModelArgs)
    KANamav3 = KANamav3(ModelArgs)
    KANamav2 = KANamav2(ModelArgs)
    KANamav1 = KANamav1(ModelArgs)

    print("Succesfull!")

except Exception as e:
    print(f"Error while creating model: {e}")
