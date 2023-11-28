'''Helper functions form model mangement'''
import os 
import json
from typing import List, Union
import pandas as pd 
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from transformer_model.config.core import TRAINED_MODEL_DIR
from transformer_model.models import constructor
from transformer_model.processing.data_manager import load_checkpoint
from transformer_model.trainer import Trainer              
from glob import glob
from cli import args 

class Model_Manager: 
    

    def trained_models(self,filter_k=''):
        trained_models = list(map(lambda x : filter_key(x,filter_k) , glob(str(TRAINED_MODEL_DIR)+'/*/*.pt')) )
        if len(trained_models) == 0 :
            raise FileExistsError(f"No traiend models exists on {str(TRAINED_MODEL_DIR)}") 
        return list(filter(None,trained_models))
    
    def models_scores(self, filter_k =''):
        scores = {}

        for model_name in self.trained_models(filter_k): 
            model_dir = TRAINED_MODEL_DIR / model_name
            try: 
                with open(model_dir/'config.json','r')as f:
                    config = json.load(f)
                scores[model_name] = config['score']
            except: 
                continue
        return pd.DataFrame(scores.values(), index=scores.keys(), columns=['score'] ).sort_values(by='score', ascending=False)


    def load_model(self, model_name:str=None, classes:list=[0,1] , best:bool = False):
        if model_name is not None: 
            model_dir = TRAINED_MODEL_DIR / model_name 
        elif best: 
            model_dir = TRAINED_MODEL_DIR / self.models_scores().index[0]
        else : 
            raise Exception("")
        with open(model_dir/'config.json','r')as f:
            config = json.load(f)
        base_model = config['_name_or_path']
        model = constructor.construct_model(base_model, classes=classes)
        try : 
            model, trained_epochs, score = load_checkpoint(model= model, path= model_dir)
            return model, trained_epochs, score
        except:
            print(f"Faild to load <{model_name}>")

    @staticmethod
    def load_config(model_name): 
        model_dir = TRAINED_MODEL_DIR / model_name
        with open(model_dir/'config.json','r')as f:
            config = json.load(f)
        return config



    def predict(self, model_name:str, path:str, data:Union[List,str]=None, accumilate=True):
            ml_model_config = self.load_config(model_name)
            trainer = Trainer(
                            base_model = ml_model_config['_name_or_path'],
                            model_name=model_name,
                            classes= [0,1], 
                            loss_function=CrossEntropyLoss() , 
                            optimizer=AdamW,
                            project_name = args['project'], 
                            wandb=None,
                            batch_size = args['batch_size'], 
                            num_workers= args['num_workers'], 
                            load_data = False
                        )
            
            return trainer.predict(data=data,  file_name = path, accumilate=accumilate)

    def aggregate_augmentations(self, models:List[str]=None):
        '''Aggregate the augmentations resulting from contrastive training
        
        Args: 
            models List(str): names of the models to return it's augmentations
        
        Returns: 
            pd.DataFrame : Augmentations with it's pseudo labels
        ''' 
        augmentations = []
        models = self.trained_models if models is None else models 
        for model_name in models: 
            model_dir = TRAINED_MODEL_DIR / model_name
            path = model_dir/'augmentations.json'
            if path.is_file(): 
                aug = pd.read_json(path)
                augmentations.append(aug)
        
        if len(augmentations) == 0 : 
            print(">>>No augmentations founded")
            return None
        
        dataframe = pd.concat(augmentations, axis=0)

        # dataframe.rename(columns={'text': config.ml_model_config.features[0], 
        #                   'label_id': config.ml_model_config.target},errors='raise' , inplace=True)
        
        # return dataframe[[config.ml_model_config.features[0], config.ml_model_config.target]]
        return dataframe[['text','label_id']].drop_duplicates(['text']).reset_index(drop=True)


def displayer(sequences): 
    for seq in sequences:
        display(seq)

def filter_key(x,filter_k): 
        name = x.split('/')[-2]
        if name.find(filter_k) != -1 : 
            return name 
    