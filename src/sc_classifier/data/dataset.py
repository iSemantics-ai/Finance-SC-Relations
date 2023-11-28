import torch 
from torch.utils.data import Dataset, DataLoader
from typing import List, Union
import numpy as np
from collections import UserDict
from pathlib import Path
from sc_classifier.config.core import config

class Model_Dataset(Dataset):
    
    def __init__(self, encoded_data):
        self.transformer_inputs = encoded_data.transformer_ids

        self.additional =False
        if 'additional_ids' in encoded_data.columns: 
            self.additional_inputs = encoded_data.additional_ids 
            self.additional =True
            
        
        self.attn_mask = encoded_data.attention_mask
        self.text = encoded_data.text
        self.labeled = False
        if 'label_id' in encoded_data.columns: 
            self.label_id_list = encoded_data.label_id
            self.labeled = True 
        
        self.data = encoded_data
    def __len__(self):
        return len(self.transformer_inputs)

    def __getitem__(self, item):
        # convert list of token_ids into an array of PyTorch LongTensors
        row = self.data.iloc[item]

        text = self.text[item]
        trans_inputs_ids_tensor = torch.LongTensor(row.transformer_ids)

        attn_mask_tensor = torch.LongTensor(row.attention_mask)

        if self.labeled :
            label_id_tensor = torch.tensor(row.label_id, dtype=torch.long)
            if self.additional:
                additional_inputs_ids_tensor = torch.LongTensor(row.additional_ids)
                return (trans_inputs_ids_tensor, additional_inputs_ids_tensor), attn_mask_tensor, label_id_tensor, text  
            return trans_inputs_ids_tensor, attn_mask_tensor, label_id_tensor, text  

        if self.additional:
            additional_inputs_ids_tensor = torch.LongTensor(row.additional_ids)
            return (trans_inputs_ids_tensor, additional_inputs_ids_tensor), attn_mask_tensor, text
        return trans_inputs_ids_tensor, attn_mask_tensor, text


# class EmbeddingsDict(UserDict):
#     """Embeddings dictionary for saving and loaddings pre-cumputed embeddings
#     """
#     def __init__(self, text:List[str]=None, embeddings:torch.Tensor=None)->None : 
#         super().__init__(self)
#         if text :
#             self[config.ml_model_config.features[0]] = text 
#         if embeddings: 
#             self['embeddings'] = embeddings

#     def save(self, path:Path): 
#         torch.save(self.data, path)

#     def load(self, path:Path):
#         self.data = torch.load(path)

class EmbeddingsDict(UserDict): 
    def __init__(self, text:str , embeddings:str ='embeddings'): 
        super().__init__(self)
        self.text = text 
        self.emb = embeddings
        self[self.text] = []
    

    def set_text(self, sentences:List[str]):
        self[self.text] = sentences

    def set_embeddings(self, embeddings:Union[np.ndarray,torch.Tensor]):
        if embeddings.shape[0] != len(self[self.text]): 
            raise ValueError("text and embeddings must have same length!")
        self[self.emb] = embeddings
    
    def save(self,path:Path):
        try : 
            torch.save(self.data, path)
        except FileExistsError as fe: 
            print(fe)
    def load(self, path:Path):
        try:
            self.data = torch.load(path)
        except FileExistsError as fe: 
            print(fe) 
    
    def keys(self): 
        return self.data.keys()

    def __str__(self): 
        if len(self[self.text]) == 0 : 
            return("Empty")
        return(f"{self.text} = {len(self[self.text])} \t {self.emb} = {self[self.emb].shape}") 
    
    def __repr__(self): 
        if len(self[self.text]) == 0 : 
            return("Empty")
        return(f"{self.text} = {len(self[self.text])} \t {self.emb} = {self[self.emb].shape}") 

           
        

  