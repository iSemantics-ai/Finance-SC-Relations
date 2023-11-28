""" Split data set into train&valid splits 
"""
import argparse
import joblib
from glob import glob
import pandas as pd
from matplotlib import pyplot  as plt
from typing import Text
import yaml
from colorama import Fore
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
import random
import os
import sys
src_dir = Path.cwd()
sys.path.append(str(src_dir)) 
from src.utils import get_logger, dict2dot, split_data
from src.utils.preprocess import create_re_data
from src.relation_extraction.preprocessing_funcs import inverse_dict


def map_relation(relation, concepts):
    for label, concepts in concepts.items():
        for concept in concepts:
            if concept in relation:
                return label
    
def data_split(config_path):

    """
    This function performs data splitting for relation extraction.

    :param config_path: (str) path to the configuration file.
    :return: None
    """
    # Load configuration file
    with open(src_dir / config_path) as conf_file:
        config = dict2dot(yaml.safe_load(conf_file))

    # Initialize logger and random seed
    logger = get_logger('data_split', log_level=config.base['log_level'])
    random.seed(config.base.random_state)

    # Load input data
    data = pd.read_json(src_dir / config.load_data.output_dir)
    # Validate input data
    missed_columns = set(['entity_1', 'entity_2', 'sents', 'org_groups', 'relations', config.base.index_col, 'Label']) - set(data.columns)
    if len(missed_columns) > 0:
        raise ValueError(f"Required columns missed <{','.join(x for x in missed_columns)}>")


    train, valid = split_data(data=data,
                   index_col = config.base.index_col,
                   stratify_by = config.data_split.stratify_by,
                    val_size = config.data_split.val_size,
                    random_state = config.base.random_state)

    assert (train['idx'].isin(valid['idx'])).sum() == 0 

    distributions= {}
    for stratify_ele in config.data_split.stratify_by:
        distributions[f"train_{stratify_ele}"] = \
        (train[stratify_ele].value_counts()/len(train)).to_dict()
        distributions[f"valid_{stratify_ele}"] = \
        (valid[stratify_ele].value_counts()/len(valid)).to_dict()

    dist_md = pd.DataFrame(distributions)
    logger.info("Train and valid distributions\n{}".format(dist_md.to_markdown()))
    # Log label distribution for train and valid data
    plt.hist(train.relations, log=True)
    plt.hist(valid.relations, log=True)
    plt.draw()

    # Concatenate weak labels, if any
    task_config = config.data_split

    # Log data distribution for train data
    logger.info(f"Train relations distribution:\n {train.relations.value_counts().to_markdown()}")
    logger.info(f"Valid relations distribution:\n {valid.relations.value_counts().to_markdown()}")

    logger.info("Entity tagging...")

    # Create tagged sentences to be tokenized
    train = create_re_data(train,
                           'sents',
                           'entity_2',
                           'entity_1',
                           'relations',
                           inverse_dict,
                          static_position=config.base.entity_static_position,
                          num_positions=config.data_split.num_positions,)
    valid = create_re_data(valid,
                           'sents',
                           'entity_2',
                           'entity_1',
                           'relations',
                           inverse_dict,
                          static_position=config.base.entity_static_position,
                          num_positions=0)

    # Save training and validation data
    logger.info(f"Saving training set with length={len(train)}")
    train.to_json(src_dir / f"{config.data_split.output_dir}/train.json")
    logger.info(f"Saving validation set with length={len(valid)}")
    valid.to_json(src_dir / f"{config.data_split.output_dir}/valid.json")
    dist_md.to_markdown(src_dir / 'data/train/distributions.md')


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', default='params.yaml')
    args = args_parser.parse_args()
    data_split(config_path=args.config)