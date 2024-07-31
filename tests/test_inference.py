import torch

from inferencee.inference import generate

from model.KANamav4 import KANamav4
from model.KANamav3 import KANamav3
from model.KANamav2 import KANamav2
from model.KANamav1 import KANamav1

from model.args import ModelArgs


ModelArgs.vocab_size = 30
ModelArgs.pad_id = 0
ModelArgs.max_batch_size = 4
ModelArgs.max_seq_len = 64


input_data = torch.tensor([[0, 1, 4, 12, 9]], dtype=torch.long)


print("Starting Inference")
try:
    KANamav4 = KANamav4(ModelArgs)
    KANamav3 = KANamav3(ModelArgs)
    KANamav2 = KANamav2(ModelArgs)
    KANamav1 = KANamav1(ModelArgs)

    output1 = generate(model=KANamav1, prompt_tokens=input_data)
    print(output1)
    output2 = generate(model=KANamav2, prompt_tokens=input_data)
    print(output2)
    output3 = generate(model=KANamav3, prompt_tokens=input_data)
    print(output3)
    output4 = generate(model=KANamav4, prompt_tokens=input_data)
    print(output4)

    print("Succesfull!")
except Exception as e:
    print(f"Error while inferencing: {e}")