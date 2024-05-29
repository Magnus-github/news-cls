from scripts import train, eval, classify, preprocessing

from omegaconf import OmegaConf, DictConfig
import argparse

import logging
import coloredlogs

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main(args: argparse.Namespace) -> None:
    cfg = OmegaConf.load(args.config)
    if args.mode == 'preprocess':
        logger.info('Preprocessing data...')
        preprocessing.preprocess_data(cfg.dataset.dir)
    elif args.mode == 'train':
        logger.info('Training model...')
        train.train(cfg, args.model_file)
    elif args.mode == 'eval':
        logger.info('Evaluating model...')
        eval.evaluate(cfg, args.model_file)
    elif args.mode == 'classify':
        logger.info('Classifying test data...')
        classify.classify(cfg, args.model_file)
    else:
        logger.error('Invalid mode. Please choose from preprocess, train, eval, classify.')
    logger.info('Done!')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the training, evaluation or classification pipeline')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to the config file')
    parser.add_argument('--mode', type=str, default='train', help='Mode to run the pipeline in (preprocess, train, eval, classify)')
    parser.add_argument('--model_file', type=str, default='model.pth', help='Model file to use for evaluation or classification')
    args = parser.parse_args()

    coloredlogs.install(level=logging.INFO, fmt="[%(asctime)s] [%(name)s] [%(module)s] [%(levelname)s] %(message)s")

    main(args)
