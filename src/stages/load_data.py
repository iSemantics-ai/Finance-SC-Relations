import argparse
import joblib
import numpy as np
import pandas as pd
from typing import Text
import yaml
from colorama import Fore
from tqdm import tqdm
from pathlib import Path
import os
import sys 
src_dir = Path.cwd()
sys.path.append(str(src_dir))
from src.utils import get_logger, dict2dot
from src.language_model.spacy_loader import SpacyLoader
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def load_configuration(config_path):
    """Load configuration from the specified file."""
    with open(config_path) as conf_file:
        return dict2dot(yaml.safe_load(conf_file))

def preprocess_data(data, config):
    """Preprocess raw data according to the provided configuration."""
    # Drop rows with missing values in specified columns
    data = data.dropna(subset=[config.load_data.text_col,
                               config.load_data.entity_1,
                               config.load_data.entity_2])
    
    # Check for missed required columns
    missed_columns = set([config.load_data.text_col,
                          config.load_data.entity_1,
                          config.load_data.entity_2,
                          config.sc_train.target]) - set(data.columns)
    if len(missed_columns) > 0:
        raise ValueError(f"Required columns missed: {', '.join(missed_columns)}")

    # Rename target column to 'relations' if specified
    if config.load_data.target is not None:
        data.rename(columns={config.load_data.target: 'relations'}, inplace=True)

    # Map 'relations' values to 'other' if not in specified main relations
    data['relations'] = data['relations'].apply(lambda x: "other" if x not in config.base.main_relations else x)

    # Rename columns to standardized names
    data.rename(columns={config.load_data.text_col: 'sents', config.load_data.entity_1: 'entity_1', config.load_data.entity_2: 'entity_2'}, inplace=True)

    # Annotate entities if 'spans' and 'org_groups' columns are missing
    if 'spans' not in data.columns or 'org_groups' not in data.columns:
        spacy_loader = SpacyLoader(lm='en_core_web_trf', require_gpu=True, load_matcher=True)
        sents, spans, groups, aliases = spacy_loader.predictor(data['sents'].tolist())
        data['sents'], data['spans'], data['org_groups'], data['aliases'] = sents, spans, groups, aliases

    # Generate unique IDs for unique strings in the 'idx' column
    if 'idx' not in data.columns:
        unique_ids, unique_strings = pd.factorize(data['idx'])
        data['idx'] = unique_ids
        np.save(src_dir / "data/train/row_ids", unique_strings)

    return data

def save_processed_data(data, output_dir):
    """Save the processed data to the specified output directory."""
    data.to_json(src_dir / output_dir, index='index')

def load_data(config_path):
    """Load and validate data."""
    config = load_configuration(src_dir / config_path)
    logger = get_logger('DATA_LOAD', log_level=config.base.log_level)

    logger.info('Get dataset')
    raw_data = pd.read_json(src_dir / config.load_data.dataset)
    processed_data = preprocess_data(raw_data, config)

    logger.info(Fore.GREEN + "Data loaded and processed successfully!!!" + Fore.RESET)
    logger.info(Fore.BLUE + f"Save training data into `{config.load_data.output_dir}`" + Fore.RESET)
    save_processed_data(processed_data, config.load_data.output_dir)

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', default='params.yaml')
    args = args_parser.parse_args()
    load_data(config_path=args.config)