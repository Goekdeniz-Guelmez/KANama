# TODO's

- Adding auto are manual device setting ('cpu' or 'cuda').
- Adding more a better training algorythm.
- Adding custom Tokenizer setting.
- Adding custom Dataset with preperty.

```python
from KANama.model.args import ModelArgs
from KANama.model.KANamav4 import KANamav4
from KANama.data.Dataset import Dataset
from KANama.trainer.SFTTrainer import SFTTrainer

ModelArgs.max_seq_len = 32

model = KANamav4(ModelArgs)

toknizer = "meta-llama/Meta-Llama-3.1-8B-Instruct"

dataset_txt = Dataset.load_text_file("path/to/dataset.txt", split={"train": 90, "validation": 5, "test": 5}, max_seq_len=model.args.max_seq_len, batch_size=ModelArgs.max_batch_size, tokenizer=toknizer) # Only "train": 90, "validation": 5, "test": 5 uses % for the int
dataset_json = Dataset.load_json_file("path/to/dataset.json", text_property: "sample", split={"train": 90, "validation": 5, "test": 5}, max_seq_len=model.args.max_seq_len, batch_size=ModelArgs.max_batch_size, tokenizer=toknizer) # text_property is the property where the to learn content is
dataset_jsonl = Dataset.load_jsonl_file("path/to/dataset.json", text_property: "sample", split={"train": 90, "validation": 5, "test": 5}, max_seq_len=model.args.max_seq_len, batch_size=ModelArgs.max_batch_size, tokenizer=toknizer) # text_property is the property where the to learn content is

# Output
# {{"train": train split}, {"validation", validation split}, {"test", test split}}

SFTTrainer(
    dataset=dataset_txt, # Will automaticaly check if the dataset has "validation" and "test" proterties other wise only "train" = dataset_txt["train"]
    max_steps=1000,
    epochs=2,
    max_train_batch_size=ModelArgs.max_batch_size,
    max_val_batch_size=ModelArgs.max_batch_size,
    max_test_batch_size=ModelArgs.max_batch_size,
    max_seq_len=model.args.max_seq_len
)
```