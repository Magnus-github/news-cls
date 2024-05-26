import torch
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt

from scripts.model import NewsClassifier
from scripts.dataset import get_dataloader

from omegaconf import OmegaConf, DictConfig
import logging
from tqdm import tqdm

import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def evaluate(cfg: DictConfig, model_file: str = 'model.pth'):
    val_loader, vocab_len, reverse_label_map = get_dataloader(cfg, split='val')

    logger.info(f'Vocab size: {vocab_len}')
    cfg.model.params.vocab_size = vocab_len
    model = NewsClassifier(**cfg.model.params)
    model_file = os.path.join(cfg.output_paths.train, model_file)
    model.load_state_dict(torch.load(model_file))
    model.eval()
    
    running_acc = 0.0
    preds = []
    labels_lst = []
    for i, (text, offsets, labels) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            output = model(text, offsets)
            running_acc += (output.argmax(1) == labels).sum().item()/len(labels)
            preds.append(output.argmax(1))
            labels_lst.append(labels)
    
    preds = torch.cat(preds).cpu().numpy()
    all_labels = torch.cat(labels_lst).cpu().numpy()
    f1 = f1_score(all_labels, preds, average=None).round(2)
    recall = recall_score(all_labels, preds, average=None).round(2)
    precision = precision_score(all_labels, preds, average=None).round(2)
    confusion = confusion_matrix(all_labels, preds, normalize='all')

    results = pd.DataFrame({'class': [reverse_label_map[i] for i in range(len(f1))],'f1': f1, 'recall': recall, 'precision': precision})
    os.makedirs(cfg.output_paths.eval, exist_ok=True)
    results.to_csv(f'{cfg.output_paths.eval}/results.csv', index=True)

    disp = ConfusionMatrixDisplay(confusion[:10,:10], display_labels=[reverse_label_map[i] for i in range(len(f1))][:10])
    disp.plot(xticks_rotation='vertical', include_values=False)
    disp.ax_.set_title('Confusion matrix (first 10 classes)')
    disp.figure_.tight_layout()
    disp.figure_.savefig(f'{cfg.output_paths.eval}/confusion_matrix.pdf')
    # plt.savefig(f'{cfg.output_paths.eval}/confusion_matrix.pdf')

    disp = ConfusionMatrixDisplay.from_predictions(all_labels, preds,
                                                   display_labels=[reverse_label_map[i] for i in range(len(f1))],
                                                   xticks_rotation='vertical',
                                                   include_values=False,
                                                   normalize='all')
    disp.figure_.tight_layout()
    disp.ax_.set_title('Confusion matrix (all classes)')
    disp.figure_.savefig(f'{cfg.output_paths.eval}/confusion_matrix_all.pdf')
    
    logger.info(f'Accuracy: {running_acc/len(val_loader)}')
    
    return


if __name__ == '__main__':
    cfg = OmegaConf.load('config/config.yaml')
    evaluate(cfg)