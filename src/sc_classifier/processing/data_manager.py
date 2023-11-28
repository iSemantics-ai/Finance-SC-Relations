from fileinput import filename

import torch
import torch.nn as nn

import logging
import re
import os 
from pathlib import Path 
from typing import Tuple
from click import FileError
from colorama import Fore, Style, Back
from shutil import copyfile
import numpy as np 
import pandas as pd 
import plotly.graph_objs as go
import plotly.offline as pyo
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from src.sc_classifier.processing.dataset import Model_Dataset
from src.sc_classifier.config.core import config, DATASET_DIR, TRAINED_MODEL_DIR

logger = logging.getLogger(__name__)

def read_and_valid(path:Path)->pd.DataFrame:

    if not path.is_file():
        raise FileExistsError("<{path.name}> this file not exist")

    if path.suffix == ".csv":
        dataframe = pd.read_csv(path)
        return dataframe

    elif path.suffix == '.json': 
        dataframe = pd.read_json(path)
        return dataframe

    elif path.suffix == ".tsv": 
        columns = config.ml_model_config.unused_fields + config.ml_model_config.features 
        dataframe = pd.read_table(path)
        if not set(columns).issubset(dataframe.columns):
            assert len(columns) == len(dataframe.columns)
            dataframe.set_axis(columns, axis=1, inplace=True) 
            dataframe.drop_duplicates(config.ml_model_config.features, inplace=True)

    else: 
        raise ValueError("Not supported format {path.suffix} ")

def load_and_valid_dataset(file_name: str) -> pd.DataFrame:
    """Load and validate data file based on particular schema,
    the schema defined on the config.yml .

    Args:
        file_name (str): the file name

    Returns:
        pd.DataFrame
    """
    path = Path(f"{DATASET_DIR}/{file_name}")
    if not path.is_file():
        path = Path(file_name)

    dataframe = read_and_valid(path)

    valid_columns = config.ml_model_config.features # At least the feature columns exist. 
    assert all([col in dataframe.columns  for col in valid_columns])
    #Pre-process
    # TODO : add pre-process pipeline(feature_engineering, text cleaning..ect)
    return dataframe

def save_frame(dataframe:pd.DataFrame , path:Path):
    '''Save dataframe based on it's suffix

    dataframe (pd.DataFrame): frame to be saved 
    path: directory to be saved at. 
    returns: 
    True if successfully saved, False if the suffix not supported.
    '''
    if path.suffix == '.csv':
        dataframe.to_csv(path,  index =0, encoding='utf-8')
        return True 
    if path.suffix == '.json': 
        dataframe.to_json(path)
        return True 
    if path.suffix == '.tsv':
        dataframe.to_csv(path,  index = 0, encoding='utf-8', sep='\t')
        return True
    
    return False 

def train_val_test_split(X, y, test_size, val_size, random_state, stratify):
    '''split a multi-labeled data into train, validation, and test sets using'''
    split_size = test_size = val_size + test_size if \
                config.ml_model_config.dev_test_split \
                else val_size

    X_train, X_val_test, y_train, y_val_test = train_test_split(X, y,
                                                                test_size= split_size ,
                                                                random_state=random_state,
                                                                stratify=stratify)
    
    if config.ml_model_config.dev_test_split : 
        X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test,
                                                        test_size=0.5,
                                                        random_state=random_state,
                                                        stratify=y_val_test)
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    
    return X_train, X_val_test, None , y_train, y_val_test, None 

    
    

def splitter(dataset:pd.DataFrame ,config)->Tuple[pd.DataFrame]:
    """Split dataframe to three splitts (train, val, test)

    Args:
        dataset (pd.DataFrame): The whole training set(labeled)
        config (_type_): application configuration

    Returns:
       Tuple[pd.DataFrame]: tuple of train,val and test dataframes
    """
    return train_val_test_split(
                                dataset[config.ml_model_config.features[0]].values, 
                                dataset[config.ml_model_config.target].values,
                                test_size=config.ml_model_config.test_size,
                                val_size=config.ml_model_config.test_size,
                                random_state=config.ml_model_config.random_state,
                                stratify=dataset[config.ml_model_config.target],
                                ) 


def train_set_pipe_line( file_name:Path,
                    config,
                    tokenizer,
                    feature=None,
                    target=None)->Tuple[pd.DataFrame]: 
    """            
    Pre-processing and data prepration for training

    @params
    -------
    file_name (Path): training data file 
    config (_type_): application configurations 
    tokenizer (_type_): transformer tokeizer

    @returns
    --------
    Tuple[pd.DataFrame]: the splitted dataset for train and evaluation.
    """
    path = DATASET_DIR/ file_name
    if not path.is_file():
        raise FileExistsError(f"<{path.name}> file not exist") 


    main_feature = config.ml_model_config.features[0]
    main_target= config.ml_model_config.target
    # start by loading the dataset and validate the trainig data 
    dataframe = read_and_valid(path)

    if feature is not None: 
        dataframe.rename(columns={feature:main_feature}, 
            inplace=True, errors='raise')

    if target is not None: 
        dataframe.rename(columns={target:main_target}, 
                        inplace=True, errors='raise')

    if  not set([main_target,main_feature]).issubset(dataframe.columns):
        raise ValueError("Cannot fetch features and targets")

    #Split training data to train, test and validation sets
    X_train, X_val, X_test, y_train, y_val, y_test = splitter(dataframe, config)
    # Preprocessing pipeline
    
    # transform inputs using pre-trained transformer tokenizer
    train_ds = tokenizer.batch_encode_plus(docs = X_train.tolist(), labels=y_train.tolist())
    val_ds = tokenizer.batch_encode_plus(docs = X_val.tolist(), labels=y_val.tolist())
  
    if config.ml_model_config.dev_test_split:
        test_ds = tokenizer.batch_encode_plus(docs = X_test.tolist(), labels=y_test.tolist())
        return train_ds, val_ds, test_ds
    return train_ds, val_ds, None


def save_checkpoint(
    model:nn.Module,
    epoch:int=0,
    score:float=0,
    path:Path=Path("./"), 
    tokenizer=None,
    optimizer=None,
    augmentations:pd.DataFrame=[])->None:
    """
    Save checkpoint as model aritfacts,
    include validation matrics score,
    model config, tokenizer data, 
    augmentations if exists 

    """
    
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist 
        os.makedirs(path)
    # save score to configures 
    model.config.score = score 
    # save num of trained epochs
    model.config.trained_epochs = epoch
    #Save model Config
    model.config.to_json_file(path/"config.json") 
    #Save model module(py file)
    model_dict = {
            'model_state_dict': model.cpu().state_dict(),
            'optimizer_state_dict': optimizer.cpu().state_dict() if optimizer else None,
            'config': model.config
                }
    torch.save(model_dict,path/'pytorch_model.pt')

    #Save customized tokenizer
    if tokenizer: 
        tokenizer.save_pretrained(path)
    if len(augmentations) >0 :
        if os.path.isfile(path / 'augmentations.json'):
            pre_aug = pd.read_json(path / 'augmentations.json')
            augmentations = pd.concat([augmentations, pre_aug] ,axis=0).drop_duplicates("text").reset_index(drop=True)
        augmentations.to_json(path / 'augmentations.json')
   
   
def load_checkpoint(model: nn.Module,path:Path,optimizer=None)->Tuple[nn.Module, int, float]:
    """Load pre-trained model 

    Args:
        model (nn.Module): the contruction module that weights loaded on it. 
        path (Path): dirctory of the saved model 
        optimizer (_type_, optional): to load latest gradients 

    Returns:
        Tuple[nn.Module, int, float]: pre-trained model, number of trained epochs, achieved score
    """
    checkpoint_data = torch.load(path/'pytorch_model.pt') 
    model.load_state_dict(checkpoint_data['model_state_dict'])
    model.config = checkpoint_data['config']

    if optimizer: 
        optimizer = optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
    score = model.config.score
    trained_epochs = model.config.trained_epochs
    return model, trained_epochs, score
    

def data_loader(encoded_data:pd.DataFrame,
                batch_size:int,
                num_workers:int=0,
                shuffle: bool=False,
                index_col:str=None)->DataLoader:
    """transform dataset module to batch generator

    Args:
        encoded_data (pd.DataFrame): dataframe caries the tokens ids and labels if exists
        batch_size (int)
        num_workers (int, optional): numper of cpu workers. Defaults to 0.
        shuffle (bool, optional): randomly shuffle the data raws. Defaults to False.

    Returns:
        Dataloader:batch generator. 
    """
    dataset= Model_Dataset(
        encoded_data,
        index_col=index_col
        )
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def load_augmentations(model_name:str): 
    path = TRAINED_MODEL_DIR/f'{model_name}/augmentations.json'
    if path.is_file(): 
        augmentations = pd.read_json(path)
        return augmentations
    else: 
        return None 
    
def save_plots(logs:dict, title, path):

    train_loss = logs['train_loss']
    val_loss = logs['val_loss']
    train_f1 = logs['train_f1']
    val_f1 = logs['val_f1']

    # Create subplot for each line in the same figure
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Train Loss", "Validation Loss", "Train F1", "Validation F1"))

    fig.add_trace(go.Scatter(x=list(range(len(train_loss))),
                             y=train_loss,
                             mode='lines',
                             name='Train Loss'), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(range(len(val_loss))), y=val_loss, mode='lines', name='Validation Loss'), row=1, col=2)
    fig.add_trace(go.Scatter(x=list(range(len(train_f1))), y=train_f1, mode='lines', name='Train F1'), row=2, col=1)
    fig.add_trace(go.Scatter(x=list(range(len(val_f1))), y=val_f1, mode='lines', name='Validation F1'), row=2, col=2)

    # Update layout for the figure
    fig.update_layout(title=title)

    # Save the figure as an HTML file
    pyo.plot(fig, filename=path)
