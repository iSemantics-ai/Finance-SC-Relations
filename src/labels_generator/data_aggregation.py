from pathlib import Path
import sys
from typing import Tuple, List, Text, Dict
from collections import defaultdict
from itertools import chain
import yaml
import pandas as pd
import random
from .agg_utils import *
class DataAggregator():
    '''Aggregating the annoated files from the LLMs from multiple files        
    '''
    def __init__(self,
                 dataset_name,
                 output_dir,
                 entity_matcher,
                 relation_direction={"customer":"supplier"},
                 filer_names=['entity_1', 'firstEntity', 'filer' ],
                 relations_key = 'relations',
                 text_col= 'sentence',
                 lm_name="en_core_web_trf",
                 lm_type='spacy',
                 ):

        # Define the basic information about the files which will be used
        self._dataset_name = dataset_name
        self.filer_names = filer_names
        self.text_col= text_col
        self.relations_key=relations_key
        

        self.relation_direction = relation_direction
        self.main_relations = list(self.relation_direction.keys()) + list(self.relation_direction.values())
        self.output_dir = output_dir

        with open(dataset_name) as o:
            self.data_files = yaml.safe_load(o)
        
        # Construct language model for entity extraction and matching
        if lm_type=='spacy':
            from src.language_model.spacy_loader import SpacyLoader
            self.lm = SpacyLoader(lm_name,
                                  entity_matcher= entity_matcher,
                                  load_matcher=True)

    def read_and_prepare_datafiles(self):
        # read file and store each into it's coresponding key_value
        for key in list(self.data_files.keys()):
            print(key,'\n--------------')
            self.data_files[key]['data'] = pd.read_excel(self.data_files[key]['dir'])
            self.data_files[key]['data'] = self.process_labeled_data(self.data_files[key]['data']).reset_index(drop=True)
            self.data_files[key]['data']['idx'] = ["{}_{}".format(key, i)\
                                            for i in range(self.data_files[key]['data'].shape[0])]
            self.data_files[key]['data']['Label'] = self.data_files[key]['data']['relations']\
                                                .apply(lambda x : sc_label_from_relations(x, self.main_relations))        

    def process_labeled_data(self,
                             data:pd.DataFrame,): 
        """
        Process labeled data by creating 'org_groups' if not existent and deserializing data.

        @params:
        - data: DataFrame containing labeled data.
        - text_col: Column containing text data.
        - relation_direction: Dictionary mapping relations.

        @returns:
        - None (modifies 'data' DataFrame in place).
        """
        # Create 'org_groups' if not existent
        if not all([x in data.columns for x in ['spans', 'org_groups']]):
            sents, spans, org_groups, aliases = self.lm.predictor(data[self.text_col])
            data[self.text_col] = sents
            data.loc[:, self.text_col] = sents
            data.loc[:, "spans"] = spans
            data.loc[:, "org_groups"] = org_groups
        # Deserialize data and evaluate relations
        eval_relation_data(data, self.relation_direction)
        return data


    def create_re_dataset(self,
                          data,
                          threshold:float=0.9,
                          max_others:int=3,
                          basic_columns:list=[],
                          only_filer=False,
                          ) -> pd.DataFrame:
        """
        Create a relation extraction dataset.

        @params
        -------
        - data: A pandas dataframe containing the following keys:
            * sentence: A string representing a sentence containing relevant text.
            * relations: A list of tuples representing the relationship between companies.

        @returns
        --------
        - dataset: A pandas dataframe containing the following columns:
        - idx: A unique identifier for the datapoint.
        - sentence: A string representing the sentence containing relevant text.
        - entity_2: A string representing the second entity in the relation.
        - relation: A string representing the relation between the two entities.
        - entity_1: A string representing the first entity in the relation.
        """

        # Apply the `extract_relations_from_llm` function to each datapoint in the data dataframe
        tqdm.pandas(desc="extract relations")
        results = data.progress_apply(lambda x: self.extract_relations_from_llm(datapoint=x,
                                threshold=threshold,
                                max_others=max_others,
                                only_filer = only_filer), axis=1)

        # Create new columns in the output dataframe to store the results of the `extract_relations_from_llm` function
        data['llm_relations']  = results.apply(lambda x : x[0]).tolist()
        data['other_relations']  = results.apply(lambda x : x[1]).tolist()
        relation_columns = ['llm_relations', 'other_relations']
        # Create a list of all possible pair-wise combinations of the values in 'ids2org', and randomly choose 5 of those combinations
        columns = relation_columns + [self.text_col, self.relations_key] + basic_columns
        re_dataset = []
        for _, row in data[columns].iterrows():
            row = row.to_dict()
            for r_column in relation_columns:
                # Iterate over relations and ingest row for each relation
                for relation_tuple in row[r_column]:
                    row['entity_2'] = relation_tuple[0]
                    row['relation'] = relation_tuple[1]
                    row['entity_1'] = relation_tuple[2]
                    re_dataset.append(dict(row))

        # Return a pandas dataframe containing the relation extraction dataset
        dataset = pd.DataFrame(re_dataset)[[ self.text_col,
                                            'entity_2',
                                            'relation',
                                            'entity_1'] + basic_columns]
        return dataset

        
    def extract_relations_from_llm(self,
                                   datapoint,
                                   threshold:float=0.9,
                                   only_filer = False,
                                   max_others=3):
        """
        Create a dataset for relation extraction training.
        @params
        -------
        - datapoint: A dictionary containing the following keys:
             * org_groups: A dictionary of company names associated with an integer identifier.
             * relations: A list of tuples representing the relationship between companies.
        - threshold: The similarity threshold for matching company names.

        @returns
        --------
        - llms_relations: A list of tuples representing the relationships between companies that were successfully matched.
        - other_relations: A list of tuples representing the relationships between companies that were not matched.

        @raises
        -------
        - ValueError: If the relations list in the datapoint is invalid.
        """
        r_others = True
        # establish org_groups
        group2id = datapoint['org_groups']
        id2group = defaultdict(list)
        for k,v in group2id.items():
            id2group[v].append(k)

        # define llms relations
        relations = datapoint[self.relations_key]

        # build index for org_groups
        if len(group2id) > 0:
            self.lm.entity_matcher.build_index(list(group2id.keys()))

        # Assert the relations on the right format
        if not check_relation_tuples(relations):
            raise ValueError("Invlid relations list on the datapoint, must be List[Tuple[Text, Text, Text]]")
        # Collect all companies mentioned in the relations and create a dictionary with each unique company as a key
        llms_companies = set()
        if isinstance(relations, list):
            llms_companies = list(set(chain(*[[x[0], x[2]] for x in relations])))

        # match the llm_companies to assign id according to group2id
        llms_co_matches = self.lm.entity_matcher.search(llms_companies, threshold=threshold, top_k=2)\
                          if len(llms_companies) > 0 else []
        # Create map the merge org_groups with llm_companies
        llms_ids = {}
        for co_match, llm_company in zip(llms_co_matches, llms_companies):
            # If match found

            if len(co_match) > 0:
                llms_ids[llm_company] = group2id[co_match[0][0]]

            # check if llm_company valid and add it to 
            elif llm_company in datapoint[self.text_col]:
                group2id[llm_company] = max(id2group.keys()) + 1 if len(id2group.keys()) > 0 else 1
                id2group[group2id[llm_company]] = [llm_company]
                llms_ids[llm_company] = group2id[llm_company]

        # Create a dictionary mapping IDs to company names
        llms_names = {k: id2group[v][0] for k, v in llms_ids.items()}

        # get all possible paris from the llms_ids
        availabel_relations = return_possible_pairs(sorted(set(llms_ids.values())))

        # Define all the exist relations from LLM with pairs tuples
        exist_relations = []
        llms_relations = []
        if isinstance(relations, list):
            for relation in relations:
                c1, c1_name = relation[0], llms_names.get(relation[0])
                c2, c2_name = relation[2], llms_names.get(relation[2])
                c1_id = llms_ids.get(c1)
                c2_id = llms_ids.get(c2)
                if None in [c1_id, c2_id]:
                    continue
                llms_relations.append((c1_name, relation[1], c2_name))
                exist_relations.append(tuple(sorted([c1_id, c2_id])))
        # Define all possible other relation pairs within the sentence
        other_ids = list(set(availabel_relations) ^ set(exist_relations))
        
        # Create relations tuples for other_relations
        other_relations = [(id2group[pair[0]][0],
                            'other',
                            id2group[pair[1]][0]) \
                           for pair in other_ids]
        # If llm return nothign, get all possible relations as `other`
        if len(llms_relations) == 0 and len(other_relations) == 0:
            other_relations = get_other_relations(id2group)
        # If only filer include only relations with filer
        if only_filer:
            # Find Filer name as mentioned on the sentence
            filer_column = list(set(datapoint.keys()).intersection(self.filer_names))
            filer_column = filer_column[0] if len(filer_column) > 0 else None
            given_filer = datapoint[filer_column] if filer_column else None
            if given_filer:
                filer_name = group2id.get(given_filer)
                if not filer_name and len(group2id) > 0:
                    filer_scope = list(group2id.keys())
                    filer_sim = self.lm.entity_matcher.similarity(given_filer,filer_scope)
                    if filer_sim.max() > threshold:
                        filer_name = filer_scope[filer_sim.argmax()]
            if filer_name:
                llms_relations = list(filter(None, [x if filer_name in [x[0], x[2]]\
                                                    else None  for x in llms_relations] ))
                other_relations = list(filter(None, [x if filer_name in [x[0], x[2]] \
                                                     else None  for x in other_relations] ))
        # Based on max_other return random sample
        other_relations = random.sample(other_relations, min(len(other_relations), max_others))

        return llms_relations, other_relations
