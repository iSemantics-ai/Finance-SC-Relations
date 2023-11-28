import os
from pathlib import Path
from typing import List, Union, Tuple, Dict
import pandas as pd
import numpy as np
import re
from collections import defaultdict
from itertools import permutations
from tqdm import tqdm
import torch
from sklearn.preprocessing import LabelBinarizer
from torch.utils.data import DataLoader
import sys
src_dir = Path(os.path.dirname(os.path.realpath(__file__))).parent
sys.path.append(str(src_dir)) 
from .train_funcs import  evaluate_results, load_state
from .preprocessing_funcs import re_dataset, Pad_Sequence, inverse_relations, inverse_dict
from .misc import  load_pickle
from .re_transfomers.re_trf import RE_Transformers
from transformers import AutoConfig, AutoTokenizer
from utils import get_logger, entity_annotation
tqdm.pandas(desc="prog-bar")
icon = '\U0001F4AB'
inverse_dict = {"supplier":"customer", "customer":"supplier", "other":"other"}

pandas_format = {'.tsv':'csv', '.csv':'csv', '.xlsx':'excel', '.json':'json'}
pandas_params = {'.tsv':{'sep':'\t'}, '.csv':{'index_col':0}}
class infer_from_trained(object):
    """Inference Module to load and predict supply relation between 
    organization entities using Transformer-Base models
    """
    def __init__(self,
                 detect_entities=False,
                 language_model:str= "en_core_web_trf",
                 require_gpu:bool=False,
                 entities_of_interest=['ORG'],
                 basic_targets= ['supplier','customer'],
                 load_matcher = False,
                 entity_matcher='artifacts/matcher_model'):
        """To initialize this module need to have dir of pretrained reltaion extractor
        model, attached with following files:
        """
        self.logger = get_logger(f'{icon} Relations Extractor', 'INFO')
        self.cuda = torch.cuda.is_available()
        self.detect_entities = detect_entities
        self.basic_targets = basic_targets
        # Use the GPU, with memory allocations directed via PyTorch.
        # This prevents out-of-memory errors that would otherwise occur from competing
        # memory pools.
        if self.cuda:
            print("Torch GPU Exists..")
            # require_gpu(0)
            # set_gpu_allocator("pytorch")
        
        if detect_entities:
            from language_model.spacy_loader import SpacyLoader
            self.spacy_loader = SpacyLoader(lm=language_model,
                                            require_gpu=require_gpu,
                                            load_matcher=load_matcher,
                                            entity_matcher = entity_matcher)
            self.entities_of_interest = entities_of_interest
        else:
            self.spacy_loader = None
        self.net = None 
        self.tokenizer = None
        
    def load_model(self, args):
        """
        Load a pre-trained relation extractor model.
        @params
        -------
        args (Dict): A dictionary that contains information about the model.
            The required keys are:
        - model_path: The path to the directory containing the model artifacts.
            This directory should include the following files:
            - added_tokens.json: A JSON file containing the specially added tokens and their IDs.
            - config.json: A JSON file containing the model configuration,
        including the architecture, number of labels, and label mapping.
        - model.pth.tar: A PyTorch checkpoint file containing the model weights.
        - tokenizer_config.json: A JSON file containing the tokenizer configuration.
        - vocab.txt: A file containing the tokenizer vocabulary.

        @return
        --------
        None.
        
        @raises:
        TypeError: If args is None.
        KeyError: If any of the required keys are missing from args.
        FileNotFoundError: If any of the required files are missing from the model directory.

        """
        # Check if args is None
        if args is None:
            self.args = load_pickle("args.pkl")
        else:
            self.args = args

        # Load the tokenizer and model
        self.logger.info("Loading tokenizer and model...")
        self.batch_size = args['batch_size']
        src_dir = Path(args.get("src_dir", './'))
        model_path = src_dir / args['model_path']
        model_artifacts = model_path / 'model.pth.tar'
        self.tokenizer = None
        config = AutoConfig.from_pretrained(model_path)
        model = config.base_model
        model_name = config.model_type
        self.net = RE_Transformers(config)
        self.tokenizer = self.tokenizer or AutoTokenizer.from_pretrained(model_path)
        # Resize the token embeddings
        self.net.model.resize_token_embeddings(len(self.tokenizer))

        # Move the model to the GPU if available
        if self.cuda:
            self.net.cuda()

        # Load the model weights and other information
        self.start_epoch, self.best_pred, amp_checkpoint = load_state(self.net, model_path, None, None, load_best=True)

        # Finish loading the model
        self.logger.info("Done!")
        self.label_binarizer = LabelBinarizer()
        self.label_binarizer.fit(np.arange(self.net.config.num_labels))

        # Get the token IDs for [E1], [E2], and padding
        self.e1_id = self.tokenizer.convert_tokens_to_ids('[E1]')
        self.e2_id = self.tokenizer.convert_tokens_to_ids('[E2]')
        self.pad_id = self.tokenizer.pad_token_id

        # Get the label mapping
        self.label2id = self.net.config.label2id
        self.id2label = {str(k):v for k,v in self.net.config.id2label.items()}
        
    def predict_fn(self,df, reverse:bool=False, mutate:bool=True)->Tuple:
        """ The predict_fn used for direct inference when the text is pre-processed
        @params
        -------
        - df(pd.DataFrame): dataframe with `sents` column containes org-tagged sentences
        
        @returns
        --------
        Tuple[
            score(np.array), # Confident scores of the predicted label
            out_labels(np.array), # The predicted label with highest score
            out_losses(np.array) # The cross-entropy loss ]
        """
        data = df.copy()
        data = self.estimate_(data, mutate=mutate)
        if reverse:
            inver_scores = torch.tensor(self.estimate_(data.copy(), mutate=mutate,
                                                       reverse=True)['scores'])
            inv_dict = inverse_dict.copy()
            _switch_pairs = []
            for k,v in inverse_dict.items(): #s , c
                if inv_dict.get(k): # s
                    if inv_dict.get(v) and v!=k: # c
                        _switch_pairs.append([(self.label2id[k], # s
                                              self.label2id[inv_dict.pop(k)]), # c
                                             (self.label2id[v], # c
                                              self.label2id[inv_dict.pop(v)])]) # s

            for switch in _switch_pairs:
                inver_scores[:, list(switch[0])] = inver_scores[:, list(switch[1])]
            
            scores = (torch.tensor(data['scores']) + inver_scores)/2
            data['scores'] = scores.tolist()
        return data
    
    def estimate_(self, df:pd.DataFrame,
                  mutate=True,
                  reverse=False):
        
        if reverse:
            df.sents = df.sents.apply(inverse_relations)
        df_set = re_dataset(df,
                            tokenizer=self.tokenizer,
                            e1_id=self.e1_id,
                            e2_id=self.e2_id,
                            mutate=mutate)
        ## If labels provided return losses
        if 'relations_id' in df.columns:
            labeled = True
            criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        else: 
            labeled = False

        PS = Pad_Sequence(seq_pad_value=self.tokenizer.pad_token_id,\
                    label_pad_value=self.tokenizer.pad_token_id,\
                    label2_pad_value=-1)
        
        loader = DataLoader(df_set, batch_size=self.args['batch_size'], shuffle=False, \
                                    num_workers=0, collate_fn=PS, pin_memory=True)
        out_scores = torch.tensor([]);   out_losses = []
        self.net.eval()
        with torch.no_grad():
            for i, data in tqdm(enumerate(loader), total=len(loader)):
                x = data[0]
                e1_e2_start = data[1]
                sents_ids = data[3] if labeled else data[2]
                attention_mask = (x != self.pad_id).float()
                token_type_ids = torch.zeros((x.shape[0], x.shape[1])).long()
                if self.cuda:
                    x = x.cuda()
                    attention_mask = attention_mask.cuda()
                    token_type_ids = token_type_ids.cuda()

                classification_logits = self.net(x, token_type_ids=token_type_ids,
                                                 attention_mask=attention_mask,
                                                 Q=None,
                                                 e1_e2_start=e1_e2_start).to('cpu')
                scores = torch.softmax(classification_logits, dim=1)
                out_scores = torch.cat((out_scores, scores), dim=0)

                if labeled:
                    losses = list(map(lambda x,y:criterion(x, y).tolist(), classification_logits, data[2].squeeze(1)))
                else:
                    losses = []
                out_losses+=losses
        if len(out_losses) > 0:
            df_set.df['losses'] = out_losses
        df_set.df['scores'] = out_scores.tolist()
        return df_set.df.drop(columns=['input','e1_e2_start'])
            
    def tag_sentences(self,sentences:List[str],
                          ent:str='ORG',
                          spans:List[Dict]=None,
                          org_groups:List[Dict]=None,
                          aliases=None,
                         num_positions=np.inf):
        # estimate NER tags (e.g. ORGs) in the sentences
        if  isinstance(sentences , list) and isinstance(sentences[0] , str): 
            df = pd.DataFrame({'sents':sentences,
                               "idx":range(len(sentences))})
        else: 
            raise ValueError('Invalid input type, `sentences` must be `List[str]` ! ')
        ## If Spans column not required
        if self.spacy_loader is None and spans is None:
            raise ValueError("No spans provided or NLP model!!!")

        if not isinstance(spans,list) or not isinstance(org_groups,list) : 
            sentences, spans, org_groups, aliases = self.spacy_loader.predictor(df['sents'])
            df.loc[:, 'sents'] = sentences

        # assert len(spans) == df.shape[0] and len(org_groups) == df.shape[0]
        # Check if `ORG` entities exist
        df.loc[:,'spans'] = spans
        df.loc[:, 'org_groups'] = org_groups
        df.loc[:, 'aliases'] = aliases
        df.loc[:, 'num_orgs'] = df.org_groups.apply(lambda x : len(set(x.values())))
        predict_index = df.query("num_orgs > 1").index
        # Start relation extracting
        tagged_frame = []
        for i,row in df.loc[predict_index].iterrows():
            sent = row['sents']
            org_group = row['org_groups']
            num_groups = row['num_orgs']
            # Inverse the org_groups to map each id to refer to it's corresponding names
            id2org = defaultdict(lambda : [])
            for k,v in org_group.items():
                id2org[v].append(k)
            # Creating pair of organization, assuming each pair as group representation
            pairs = [] 
            group_ids = list(set(org_group.values()))
            for idx in range(num_groups) :
                pairs += [(sorted(id2org[group_ids[idx]], key=len)[::-1][0],
                           sorted(id2org[group_ids[n]], key=len)[::-1][0]) for\
                    n in range(idx+1, num_groups) ]
            # Create frame to consist of all the tagged sentences
            all_sents = []
            for i, pair in enumerate(pairs):
                annotated_sents = entity_annotation(sent, pair[0], pair[1],org_group , num_positions=num_positions)
                if len(annotated_sents) > 0:
                    for tagged_sent in annotated_sents:
                        tagged_frame.append({
                            "sents":tagged_sent,
                            "orig_sents":sent,
                            "entity1":pair[0],
                            "entity2":pair[1],
                            "org_groups":org_group, 
                            "idx":row['idx'],
                            'r_id': "{}_{}".format(row['idx'], i)
                        })
        if len(tagged_frame) == 0: 
            self.logger.info(f"No valid rows with at least two `ORG` entities exist!!!\n{df[['sents','spans']].to_dict(orient='records')}")
            return df
        tagged_frame = pd.DataFrame(tagged_frame).reset_index(drop=True)
        return tagged_frame
    
    def predict_relations(self, sentences:List[str],
                          ent:str='ORG',
                          spans:List[Dict]=None,
                          org_groups:List[Dict]=None,
                          aliases=None,
                          reverse:bool=False,
                          mutate:bool=False,
                         num_positions:int= np.inf)->pd.DataFrame:
        """ This functions extract the relations between groups of entities in
        given sentences following the next steps: 
            - Use SpacyLoader module to extract entities spans, org_groups, aliases 
                from each sentence.
            - Mutate the sentence by inserting the `[E1], [E2]` tags to cover 
                all org_groups mentioned in the text.
            - Use the built in method `predict_fn` to predict the relation of all 
                possible relation in each sentence
            - Aggregate the predictions on one dictionary, each sentence aligned with 
                relations dict of possible relations
                
        @params
        -------
        - sentences(List[str]): sequence of string sentenes
        - ent(str): The entity type to detect relations between
        - spans(List[Dict]): NER spans for the entities, default(None)
            if `None` use SpacyLoader to extract the spans and groups
        - org_groups(List[Dict]): Organizations groups each group have unique id, default(None)
            if `None` use SpacyLoader to extract the spans and groups
            
        @returns
        --------
        - pd.DataFrame # This dataframe consist of 4 columns: 
                    - `sents`: the valid sentences
                    - `idx`: sentence order in the input sequence
                    - `org_groups`: the detected organizations with `group id`
                    - `spans`: NER spans for the entities
                    - `num_orgs`: number of groups
                    - `relations`: all the possible relations between 
                        all the `org_groups` entities
        """
        #tagged_frame = pd.DataFrame(tagged_frame)
        tagged_frame = self.tag_sentences(sentences=sentences,
                                          ent=ent,
                                          spans=spans,
                                          org_groups=org_groups,
                                          aliases=aliases,
                                          num_positions=num_positions)
        # estimate softmax scores    
        tagged_frame = self.predict_fn(tagged_frame, reverse=reverse, mutate=mutate)
        # aggregate multi-positioning relations.
        id_scores = tagged_frame.groupby(['r_id'])\
         .apply(lambda x : list(np.mean(x['scores'].tolist(), axis=0))).to_dict()
        # assign aggregated relation for all positions
        tagged_frame['scores'] = tagged_frame['r_id'].apply(lambda x: id_scores[x])
        # drop duplicates from multi-positions
        tagged_frame.drop_duplicates(subset=['r_id'], inplace=True, ignore_index=True)
        # define max scores and its label_ids
        score, labels = torch.tensor(tagged_frame['scores']).max(1)
        # create relations info items to compine relations on each sentence
        tagged_frame.loc[:, 'relation'],\
        tagged_frame.loc[:, 'scores'] =  labels, score
        tagged_frame.loc[:, 'relation'] = tagged_frame['relation']\
        .apply(lambda x : self.id2label[str(x)])
        tagged_frame.loc[:, 'relations_info'] =tagged_frame\
        .apply(lambda x : assign_relation(x['sents'], x['relation'], x['scores']), axis=1)
        # tagged_frame.loc[:, 'r_scores'] = [{self.id2label[str(i)] :round(v.item(),4)\
        # for i,v  in enumerate(score)} for score in scores]        
        # #tagged_frame['sents'] = df['sents']
        # Group the predictions and merge with the input frame
        relation_groups = []
        for i,group in tagged_frame.groupby('idx'):
            relations = list(group['relations_info'])
            group_id = group['idx'].unique()[0]
            relation_groups.append({'relations':relations ,
                                    "idx": group_id})
        merged_df = pd.merge(pd.DataFrame(relation_groups), tagged_frame[['idx','orig_sents', 'org_groups']],
                            on='idx', how='inner').drop_duplicates(['idx'])
        return merged_df.set_index('idx')
    
    def predict_frame(self, path:Union[pd.DataFrame,str],
                         sentence_column:str,
                         spans_column:str='spans',
                         org_groups_column:str='org_groups',
                         reverse:bool=False,
                         mutate:bool=True):
        """Apply `predict_relations` function on data file
        
        Args:
            path(str): Exist dir for data file with any of the following format
                        [tsv, csv, json, xlsx]
            
            sentence_column(str): this column is required
            spans_column(str): entity spans column of exist. Default('spans')
            org_groups_column
        """
        if isinstance(path, pd.DataFrame): 
            df = path
        elif isinstance(path, str) and os.path.exists(path):
            suffix = Path(path).suffix
            if not suffix in ['.tsv','.csv','.json','.xlsx']:
                raise(f'Invalid file type {path} must be in {pandas_format.keys()}')
            # Read the file based on the requirements
            df = pd.__getattribute__(f'read_{pandas_format[suffix]}')\
                                    (path,**pandas_params.get(suffix,{}))
            
        else: 
            raise ValueError("Invalid input `path` must be valid directory or pandas frame")
            
        if not sentence_column in df.columns: 
            raise ValueError(f"Must verify valid `sentence _column` {sentence_column} is not exist!!")
            
        if spans_column not in df.columns or org_groups_column not in df.columns: 
            spans, org_groups, aliases= None, None, None 
        else:
            spans, org_groups, aliases=(df[spans_column].tolist() ,
                                        df[org_groups_column].tolist(), 
                                        df['aliases'].tolist())
        
        output = self.predict_relations(df[sentence_column].tolist(),
                                        spans= spans,
                                        org_groups= org_groups,
                                        aliases=aliases,
                                        reverse=reverse,
                                        mutate=mutate)
        return output
        

    
    
    def create_loader(self, data_or_path): 
        raise NotImplementedError

    def evaluate_results(self, dataloader, pad_id=None, cuda=None, top_losses=False,calc_ruc=False):
        self.logger.info("Start Evaluation...")
        pad_id = pad_id or self.pad_id
        cuda = cuda or self.cuda
        
        cr, roc_plot = evaluate_results(self.net,
                                        dataloader,
                                        self.label_binarizer,
                                        self.id2label,
                                        pad_id,
                                        cuda,
                                        calc_ruc)
        return cr, roc_plot
    




    def get_e1e2_start(self, x):
        e1_e2_start = ([i for i, e in enumerate(x) if e == self.e1_id][0],\
                        [i for i, e in enumerate(x) if e == self.e2_id][0])
        return e1_e2_start
    


def assign_relation(sent, relation, score, main_relations=['supplier', 'customer']): 
    e1_start = '[E1] '
    e1_end = ' [/E1]'
    e2_start = '[E2] '
    e2_end = ' [/E2]'
    e1 = sent[sent.find(e1_start)+len(e1_start):sent.rfind(e1_end)]
    e2 = sent[sent.find(e2_start)+len(e2_start):sent.rfind(e2_end)]
#     if relation == main_relations[1]:
#         e1,e2 = e2,e1
#         relation = main_relations[0]
#     return (e2,relation,round(score,4),e1)

    return {e2:relation, 
            e1:inverse_dict.get(relation, 'other'),
            "score":round(score,4)}