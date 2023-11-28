from tqdm import tqdm
import torch 
from torch.utils.data import Dataset, DataLoader



class Model_Dataset(Dataset):
    
    def __init__(self, encoded_data, index_col:str=None):
        self._index_col = index_col
        self.additional =False
        self.labeled = False

        if 'additional_ids' in encoded_data.columns: 
            self.additional =True
        
        if 'label_id' in encoded_data.columns: 
            self.labeled = True 
        self.data = encoded_data.copy()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        # convert list of token_ids into an array of PyTorch LongTensors
        row = self.data.iloc[item]
        text = row.text
        index= row[self._index_col] if self._index_col else item
        trans_inputs_ids_tensor = torch.LongTensor(row.transformer_ids)
        attn_mask_tensor = torch.LongTensor(row.attention_mask)

        if self.labeled :
            label_id_tensor = torch.tensor(row.label_id, dtype=torch.long)
            if self.additional:
                additional_inputs_ids_tensor = torch.LongTensor(row.additional_ids)
                return ((trans_inputs_ids_tensor,
                        additional_inputs_ids_tensor),
                        attn_mask_tensor,
                        label_id_tensor,
                        text,
                        index)
            return (trans_inputs_ids_tensor,
                    attn_mask_tensor,
                    label_id_tensor,
                    text,
                    index)

        if self.additional:
            additional_inputs_ids_tensor = torch.LongTensor(row.additional_ids)
            return ((trans_inputs_ids_tensor, additional_inputs_ids_tensor),
                    attn_mask_tensor, text, index)
        
        return (trans_inputs_ids_tensor,
                attn_mask_tensor,
                text,
                index)