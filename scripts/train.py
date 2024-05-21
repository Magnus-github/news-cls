import torch

from model import NewsClassifier
from dataset import get_dataloader

from omegaconf import OmegaConf
import logging
import coloredlogs

import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
coloredlogs.install(level=logging.INFO, fmt="[%(asctime)s] [%(name)s] [%(module)s] [%(levelname)s] %(message)s")


def train():
    cfg = OmegaConf.load('config/config.yaml')
    model = NewsClassifier(**cfg.model.params)
    
    train_loader = get_dataloader(split='train', batch_size=cfg.hparams.batch_size)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.hparams.learning_rate)
    
    logger.info('Starting training...')
    for epoch in range(cfg.hparams.epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0

        for i, (text, offsets, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(text, offsets)
            loss = criterion(output, labels)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            running_acc += (output.argmax(1) == labels).sum().item()/len(labels)
            running_loss += loss.item()
            

        logger.info(f'Epoch: {epoch}, Loss: {running_loss/len(train_loader)}, Accuracy: {running_acc/len(train_loader)}')
        
    os.makedirs(os.path.dirname(cfg.model.save_path), exist_ok=True)
    torch.save(model.state_dict(), f'{cfg.model.save_path}model.pth')
    
    return None


if __name__ == '__main__':
    train()