import torch
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from torch.utils.data import Dataset, DataLoader

import pandas as pd
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig


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
        self.reverse_label_mapping = {v: k for k, v in self.label_mapping.items()}
        self._tokenizer = get_tokenizer('basic_english')
        train_tokens = [self._tokenizer(text) for text in train_data['full_text']]
        self._vocab = build_vocab_from_iterator(train_tokens, specials=['<unk>'], min_freq=1)
        self._vocab.set_default_index(self._vocab['<unk>'])
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


def get_dataloader(cfg: DictConfig,
                   split: str = 'train') -> DataLoader:
    dataset = NewsDataset(cfg.dataset.dir, split)
    if split == 'train':
        if cfg.dataset.sampling.enable:
            class_sample_count = dataset.data['category'].value_counts().to_dict()
            weights = 1 / torch.tensor([class_sample_count[label] for label in dataset.label_mapping.keys() if label in class_sample_count])
            samples_weight = torch.tensor([weights[t] for t in dataset.data['category'].map(dataset.label_mapping)])
            if cfg.dataset.sampling.oversampling:
                num_samples = int(max(class_sample_count.values()))*len(samples_weight)
            else:
                num_samples = len(samples_weight)
            sampler = torch.utils.data.WeightedRandomSampler(samples_weight, num_samples)
            return DataLoader(dataset, batch_size=cfg.hparams.batch_size, sampler=sampler, collate_fn=collate_fn), len(dataset._vocab), dataset.reverse_label_mapping
    return DataLoader(dataset, batch_size=cfg.hparams.batch_size, shuffle=True, collate_fn=collate_fn), len(dataset._vocab), dataset.reverse_label_mapping
    

if __name__ == "__main__":
    cfg = OmegaConf.load('config/config.yaml')
    dataloader = get_dataloader(cfg)

    for i, (text, offsets, labels) in enumerate(dataloader):
        print(text)
        print(offsets)
        print(labels)
        if i == 0:
            break
