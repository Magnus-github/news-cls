from scripts import train, eval, classify

from omegaconf import OmegaConf, DictConfig
import argparse

import logging
import coloredlogs

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main(cfg: DictConfig, mode: str = 'train') -> None:
    if mode == 'train':
        logger.info('Training model...')
        train.train(cfg)
    elif mode == 'eval':
        logger.info('Evaluating model...')
        eval.evaluate(cfg)
    elif mode == 'classify':
        logger.info('Classifying test data...')
        classify.classify(cfg)
    logger.info('Done!')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the training, evaluation or classification pipeline')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to the config file')
    parser.add_argument('--mode', type=str, default='train', help='Mode to run the pipeline in (train, eval, classify)')
    args = parser.parse_args()

    coloredlogs.install(level=logging.INFO, fmt="[%(asctime)s] [%(name)s] [%(module)s] [%(levelname)s] %(message)s")

    cfg = OmegaConf.load(args.config)
    main(cfg, mode=args.mode)
