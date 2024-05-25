import torch
import pandas as pd

from scripts.model import NewsClassifier
from scripts.dataset import NewsDataset

from omegaconf import OmegaConf, DictConfig
import logging
import coloredlogs
from tqdm import tqdm

import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def classify(cfg: DictConfig):
    test_data = os.path.join(cfg.dataset.dir, cfg.dataset.test_data_file)
    test_dataframe = pd.read_json(test_data, lines=True)

    dataset = NewsDataset(cfg.dataset.dir, split='test')
    reverse_label_mapping = dataset.reverse_label_mapping
    text_pipeline = dataset._text_pipeline
    vocab_len = len(dataset._vocab)

    logger.info(f'Vocab size: {vocab_len}')
    cfg.model.params.vocab_size = vocab_len
    model = NewsClassifier(**cfg.model.params)
    model_file = os.path.join(cfg.output_paths.train, 'model_85train.pth')
    model.load_state_dict(torch.load(model_file))
    model.eval()
    
    preds = []
    for i in tqdm(range(len(test_dataframe))):
        text = test_dataframe.iloc[i]['full_text']
        text_tensor = torch.tensor(text_pipeline(text), dtype=torch.int64)
        text_tensor = text_tensor.unsqueeze(0)
        offsets = None
        with torch.no_grad():
            output = model(text_tensor, offsets)
            preds.append(output.argmax(1))
    
    categorical_preds = [reverse_label_mapping[i.item()] for i in preds]
    test_dataframe = test_dataframe.assign(predicted_category=categorical_preds)
    test_dataframe.to_json(f'{cfg.output_paths.test}/test_results.jsonl', orient='records', lines=True)

    return


if __name__ == '__main__':
    cfg = OmegaConf.load('config/config.yaml')
    classify(cfg)