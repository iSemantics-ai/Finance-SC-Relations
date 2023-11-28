'''Install needed packeges'''
import pandas as pd
from transformers import AutoTokenizer
# from language_model.nlp import Language_Model



class Transformer_Tokenizer():
    '''Tokeizer for transformers
    '''
    def __init__(self,transformer_name):
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_name) 

    def batch_encode_plus(
        self,
        docs,
        labels=None,
        add_special_tokens=True,
        max_length = None,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True,
        ): 
        data = pd.DataFrame(columns= ['transformer_ids','attention_mask' ,'text'])

            
        batch_encoded = self.tokenizer.batch_encode_plus(
            docs,
            add_special_tokens=True,
            max_length=max_length,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        data.transformer_ids  = batch_encoded.input_ids.tolist() 
        data.attention_mask  = batch_encoded.attention_mask.tolist() 
        data['text'] = docs
        if labels != None : 
            data['label_id'] = labels
        return data
    
    def encode(self, text:str): 
        return self.tokenizer.encode(text ,return_tensors='pt')

    
    def save_pretrained(self,path): 
        self.tokenizer.save_pretrained(path)