from pathlib import Path
from typing import Sequence, Dict,Text
from pydantic import BaseModel
from strictyaml import YAML, load
from src import sc_classifier

# Project Directories 
PACKAGE_ROOT  = Path(sc_classifier.__file__).resolve().parent.parent.parent
print("root==>", PACKAGE_ROOT)
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / 'params.yaml'
DATASET_DIR = ROOT / "data/raw"
SENTEVAL = PACKAGE_ROOT / "SentEval/data/downstream"
EMB_DIR = DATASET_DIR / "embeddings"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "artifacts"

class AppConfig(BaseModel): 
    """
    Application-level config.
    """
    package_name:Text 
    train_file:Text
    valid_file:Text
    pipeline_save_file:Text 
    random_state:int =42
    log_level:Text= "INFO"
    nlp:Text= "en_core_web_trf"
    basic_columns:Sequence[Text]
    index_col:Text
class ModelConfig(BaseModel):
    """
    All configuration relevant to model 
    trainig and feature engineering.
    """
    target:Text 
    features:Sequence[Text]
    classes:Sequence[Text]
class TrainArgs(BaseModel):
    base_model: str
    batch_size: int
    cuda: str
    epochs: int
    num_workers: int
    seed: int
    truncate: int
    wandb: bool
    warmup_smooth: float
    weight_decay: float
    learning_rate: float
    load_pretrained: bool
    mutate: bool
    metric_dir:Text='sc_metrics'
    
class Config(BaseModel):
    """Master config object"""
    app_config: AppConfig
    ml_model_config:ModelConfig
    train_args:TrainArgs
    

    
    
def find_config_file()-> Path :
    """Locate the configuration file."""
    if CONFIG_FILE_PATH.is_file(): 
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH}")

def fetch_config_from_yaml(cfg_path:Path=None)-> YAML:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()
    
    if cfg_path: 
        with open(cfg_path,"r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")

def create_and_validate_config(parsed_config:YAML=None)->Config:
    """Run validation on config values."""
    parsed_config = fetch_config_from_yaml()

    # specify the data attribute from the strictyaml YAML type 
    _config = Config(
        app_config= AppConfig(**{**parsed_config.data['sc_train'],
                              **parsed_config.data['base']}),
        ml_model_config = ModelConfig(**parsed_config.data['sc_train']),
        train_args = TrainArgs(**parsed_config.data['sc_train'])
        
    )
    return _config 

config = create_and_validate_config()
