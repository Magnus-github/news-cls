import torch
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt

from model import NewsClassifier
from dataset import get_dataloader

from omegaconf import OmegaConf
import logging
import coloredlogs
from tqdm import tqdm

import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
coloredlogs.install(level=logging.INFO, fmt="[%(asctime)s] [%(name)s] [%(module)s] [%(levelname)s] %(message)s")


def evaluate(cfg):
    test_loader, vocab_len, reverse_label_map = get_dataloader(cfg, split='val')

    logger.info(f'Vocab size: {vocab_len}')
    cfg.model.params.vocab_size = vocab_len
    model = NewsClassifier(**cfg.model.params)
    model_file = os.path.join(cfg.model.save_path, 'model_85train.pth')
    model.load_state_dict(torch.load(model_file))
    model.eval()
    
    running_acc = 0.0
    preds = []
    labels_lst = []
    for i, (text, offsets, labels) in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            output = model(text, offsets)
            running_acc += (output.argmax(1) == labels).sum().item()/len(labels)
            preds.append(output.argmax(1))
            labels_lst.append(labels)
    
    preds = torch.cat(preds).cpu().numpy()
    all_labels = torch.cat(labels_lst).cpu().numpy()
    f1 = f1_score(all_labels, preds, average=None)
    recall = recall_score(all_labels, preds, average=None)
    precision = precision_score(all_labels, preds, average=None)
    confusion = confusion_matrix(all_labels, preds)

    results = pd.DataFrame({'class': [reverse_label_map[i] for i in range(len(f1))],'f1': f1, 'recall': recall, 'precision': precision})
    results.to_csv('results.csv', index=True)

    disp = ConfusionMatrixDisplay(confusion[:5,:5], display_labels=[reverse_label_map[i] for i in range(len(f1))][:5])
    disp.plot()
    plt.savefig('confusion_matrix.png')
    
    logger.info(f'Accuracy: {running_acc/len(test_loader)}')
    
    return None


if __name__ == '__main__':
    cfg = OmegaConf.load('config/config.yaml')
    evaluate(cfg)