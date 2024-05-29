import torch

from scripts.model import NewsClassifier
from scripts.dataset import get_dataloader

from omegaconf import OmegaConf, DictConfig
import logging
from tqdm import tqdm

import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def train(cfg: DictConfig, model_file: str = 'model.pth') -> None:
    train_loader, vocab_len = get_dataloader(cfg, split='train')

    logger.info(f'Vocab size: {vocab_len}')
    cfg.model.params.vocab_size = vocab_len
    model = NewsClassifier(**cfg.model.params)
    
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.hparams.learning_rate)
    
    logger.info('Starting training...')
    for epoch in range(cfg.hparams.epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0

        for i, (text, offsets, labels) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            output = model(text, offsets)
            loss = criterion(output, labels)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            running_acc += (output.argmax(1) == labels).sum().item()/len(labels)
            running_loss += loss.item()
            

        logger.info(f'Epoch: {epoch}, Loss: {running_loss/len(train_loader)}, Accuracy: {running_acc/len(train_loader)}')
        
    os.makedirs(os.path.dirname(cfg.output_paths.train), exist_ok=True)
    save_path = os.path.join(cfg.output_paths.train, model_file)
    torch.save(model.state_dict(), save_path)
    
    return


if __name__ == '__main__':
    cfg = OmegaConf.load('config/config.yaml')
    train(cfg)
