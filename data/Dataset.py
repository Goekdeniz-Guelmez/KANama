from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import json

class TokenizedDataset(Dataset):
    def __init__(self, tokenized_texts):
        self.tokenized_texts = tokenized_texts

    def __len__(self):
        return len(self.tokenized_texts)

    def __getitem__(self, idx):
        return self.tokenized_texts[idx]

class Dataset:
    @staticmethod
    def load_text_file(path, split, max_seq_len, tokenizer_name, batch_size=16):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        with open(path, 'r') as f:
            text = f.readlines()

        tokenized_texts = [tokenizer.encode(t.strip(), truncation=True, max_length=max_seq_len) for t in text]

        return Dataset._split_dataset(tokenized_texts, split, batch_size)

    @staticmethod
    def load_json_file(path, text_property, split, max_seq_len, tokenizer_name, batch_size=16):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        with open(path, 'r') as f:
            data = json.load(f)

        tokenized_texts = [tokenizer.encode(item[text_property].strip(), truncation=True, max_length=max_seq_len) for item in data]

        return Dataset._split_dataset(tokenized_texts, split, batch_size)

    @staticmethod
    def load_jsonl_file(path, text_property, split, max_seq_len, tokenizer_name, batch_size=16):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        with open(path, 'r') as f:
            lines = f.readlines()

        tokenized_texts = [tokenizer.encode(json.loads(line)[text_property].strip(), truncation=True, max_length=max_seq_len) for line in lines]

        return Dataset._split_dataset(tokenized_texts, split, batch_size)

    @staticmethod
    def _split_dataset(tokenized_texts, split, batch_size):
        total_size = len(tokenized_texts)
        splits = {k: int(v / 100 * total_size) for k, v in split.items()}

        train_size = splits.get('train', total_size)
        remaining_size = total_size - train_size
        val_size = splits.get('validation', remaining_size if 'test' not in splits else remaining_size // 2)
        test_size = splits.get('test', remaining_size - val_size)

        train_dataset = TokenizedDataset(tokenized_texts[:train_size])
        val_dataset = TokenizedDataset(tokenized_texts[train_size:train_size + val_size]) if val_size > 0 else None
        test_dataset = TokenizedDataset(tokenized_texts[train_size + val_size:]) if test_size > 0 else None

        data_loaders = {"train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True)}
        if val_dataset:
            data_loaders["validation"] = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        if test_dataset:
            data_loaders["test"] = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return data_loaders