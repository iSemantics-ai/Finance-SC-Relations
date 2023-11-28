""" Split data set into train&valid splits 
"""
import argparse
import joblib
import pandas as pd
from matplotlib import pyplot  as plt
from typing import Text
import yaml
from colorama import Fore
from tqdm import tqdm
from pathlib import Path
import sys 
src_dir = Path.cwd()
sys.path.append(str(src_dir))

from src.utils import get_logger, dict2dot
from src.relation_extraction.trainer import train_and_fit
def train_model(config_path):
    """This method tag text to be formated with respect to relation extraction enocder 
    """
    with open(config_path) as conf_file:
        config = dict2dot(yaml.safe_load(conf_file))
    logger = get_logger('train', log_level=config.base.log_level)
    train_and_fit(config.train)

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', default="params.yaml")
    args = args_parser.parse_args()
    train_model(config_path=args.config)