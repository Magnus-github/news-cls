import torch
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from torch.utils.data import Dataset, DataLoader

import pandas as pd
from omegaconf import OmegaConf


class NewsDataset(Dataset):
    def __init__(self,
                 data_folder: str = 'data/',
                 split: str = 'train') -> None:
        super().__init__()
        train_data = pd.read_json(data_folder + 'train_prep.jsonl', lines=True)
        if split == 'train':
            self.data = train_data
        elif split == 'val':
            self.data = pd.read_json(data_folder + 'dev_prep.jsonl', lines=True)
        elif split == 'test':
            self.data = pd.read_json(data_folder + 'test_prep.jsonl', lines=True)
        
        self.label_mapping = dict(OmegaConf.load('config/classes.yaml'))
        self._tokenizer = get_tokenizer('basic_english')
        self._vocab = build_vocab_from_iterator(map(self._tokenizer, train_data['full_text']))
        self.split = split
    
    def _text_pipeline(self, text: str) -> torch.Tensor:
        return self._vocab(self._tokenizer(text))

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> dict:
        text = self.data.iloc[idx]['full_text']
        if self.split == 'test':
            label = 'unknown'
        else:
            label = self.data.iloc[idx]['category']

        return {'text': self._text_pipeline(text),
                'label': self.label_mapping[label]}
    

def collate_fn(batch: list[dict]) -> tuple:
    label_lst , text_lst, offsets = [], [], [0]
    for item in batch:
        label_lst.append(item['label'])
        text = torch.tensor(item['text'], dtype=torch.int64)
        text_lst.append(text)
        offsets.append(text.size(0))
    label_tensor = torch.tensor(label_lst, dtype=torch.int64)
    text_tensor = torch.cat(text_lst)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    return text_tensor, offsets, label_tensor


def get_dataloader(data_folder: str = 'data/',
                   split: str = 'train',
                   batch_size: int = 16) -> DataLoader:
    dataset = NewsDataset(data_folder, split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    

if __name__ == "__main__":
    dataloader = get_dataloader()

    for i, (text, offsets, labels) in enumerate(dataloader):
        print(text)
        print(offsets)
        print(labels)
        if i == 0:
            break
