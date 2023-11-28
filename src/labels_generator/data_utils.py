
import re
import json
import time
from typing import Tuple, List, Text, Dict
from collections import defaultdict
from tqdm import tqdm
import traceback
from typing import Text 
from itertools import chain
from copy import copy
import pandas as pd
import numpy as np
import random
from .utils import check_relation_tuples, resort_relation

def process_labeled_data(data,
                         text_col:str,
                         relation_tuple:str,
                        relations_map:dict): 
    # Create org_groups if not exist
    if not all([x in data.columns for x in ['spans', 'org_groups']]):
        from src.language_model.spacy_loader import SpacyLoader
        spacy_loader = SpacyLoader("en_core_web_trf",
                                   entity_matcher="artifacts/matcher_model/",
                                  load_matcher=True)
        sents, spans, org_groups, aliases = spacy_loader.predictor(data[text_col])
        data[text_col]= sents
        data.loc[:, text_col] = sents
        data.loc[:, "spans"] = spans
        data.loc[:, "org_groups"] = org_groups
    # deserialize data
    eval_relation_data(data, relations_map)
def get_other_relations(org_groups, max_others:int):
    """
    Returns a list of other relations between companies based on the dictionary of company groups passed as input.
    The maximum number of other relations is determined by the max_others parameter.

    @params
    -------
    org_groups (dict): A dictionary with company names as keys and groups as values.
    max_others (int): The maximum number of other relations to return.

    @returns
    --------
    list: A list of other relations between companies.
    """
    # Collect all unique company names from the org_groups dictionary
    orgs = list(org_groups.keys())
    orgs = {k:None for k in set(orgs)}

    # Create a dictionary to map each unique company name to an ID
    orgs_ids = {k:i for i,k in enumerate(set(orgs))}
    ids2org = {i:k for i,k in enumerate(set(orgs_ids))}

    # Create a list of all possible pair-wise combinations of the values in 'ids2org', and randomly choose max_others of those combinations
    available_relations = []
    comp_keys = list(ids2org.keys())
    for i in range(len(comp_keys)):
        for j in range(i+1, len(comp_keys)):
            relation_t = tuple(sorted([comp_keys[i], comp_keys[j]]))
            available_relations.append(relation_t)

    other_ids = set(available_relations)

    other_relations = []
    if len(other_ids) > max_others:
        other_ids = random.sample(other_ids, max_others)

    # Map the IDs back to the company names and create a list of other relations
    for pair in other_ids:
        c1 = ids2org[pair[0]]
        c2 = ids2org[pair[1]]
        other_relations.append((c1,'other', c2))

    return other_relations

def extract_relations_from_llm(datapoint,
                                matcher,
                                text_col:str="sentence",
                                relations_key:str='relations',
                                threshold:float=0.9,
                               max_others=3):
    
    """
    Create a dataset for relation extraction training.

    @params
    -------
    datapoint: A dictionary containing the following keys:
        - org_groups: A dictionary of company names associated with an integer identifier.
        - relations: A list of tuples representing the relationship between companies.
    matcher: An instance of the `StringMatcher` class.
    text_col: The key in the datapoint dictionary where the text to match can be found.
    relations_key: The key in the datapoint dictionary where the relations can be found.
    threshold: The similarity threshold for matching company names.

    @returns
    --------
    llms_relations: A list of tuples representing the relationships between companies that were successfully matched.
    other_relations: A list of tuples representing the relationships between companies that were not matched.

    @raises
    -------
    - ValueError: If the relations list in the datapoint is invalid.
    """
    org_groups = datapoint['org_groups']
    relations = datapoint[relations_key]
    # Assert the relations on the right format
    if not check_relation_tuples:
        raise ValueError("Invlid relations list on the datapoint, must be List[Tuple[Text, Text, Text]]")
    matcher_built = False
    # Collect all companies mentioned in the relations and create a dictionary with each unique company as a key
    llms_companies = []
    if isinstance(relations, list):
        for relation in relations:
            llms_companies += [relation[0] , relation[2]]
    llms_companies = {k:None for k in set(llms_companies)}
    llms_ids = {k:i for i,k in enumerate(set(llms_companies))}
    ids_llms = {i:k for i,k in enumerate(set(llms_companies))}
    # Check if each company in the dictionary is mentioned in the sentence, and if not, try to match it with a known organization
    for company in list(llms_companies.keys()):
        if company in datapoint[text_col]:
            llms_companies[company] = company
        elif len(org_groups):
            if matcher_built is False:
                matcher.build_index(list(org_groups.keys()))
                matcher_built = True
            matches = matcher.search(company, threshold=threshold    , top_k = 3)
            if len(matches) > 0:
                llms_companies[company] = matches[0][0]
            else:
                llms_companies.pop(company)
                ids_llms.pop(llms_ids.pop(company))
                
    # Create a dictionary called 'ids2org' that maps each value in 'org_groups' to a list of keys that have that value
    ids2org = defaultdict(lambda : [])
    for key ,val in llms_ids.items():
        ids2org[val].append(key)

    '''
    Create a list of all possible pair-wise combinations of the values in 'ids2org',
    and randomly choose 5 of those combinations
    '''
    availabel_relations = []
    comp_keys = list(ids2org.keys())
    for i in range(len(comp_keys)):
        for j in range(i+1, len(comp_keys)):
            relation_t = tuple(sorted([comp_keys[i], comp_keys[j]]))
            availabel_relations.append(relation_t)
    exist_relations = []
    llms_relations = []
    if isinstance(relations, list):
        for relation in relations:
            c1, c1_name = relation[0], llms_companies.get(relation[0])
            c2, c2_name = relation[2], llms_companies.get(relation[2])
            relation = relation[1]
            if not all([c1 in llms_companies.keys() , c2 in llms_companies.keys()]):
                continue
            if not all([c1,c2,relation]):
                continue 
            llms_relations.append((c1_name, relation, c2_name))

            c1_id = llms_ids.get(c1)
            c2_id = llms_ids.get(c2)    
            exist_relations.append(tuple(sorted([c1_id, c2_id])))
    
    other_ids = set(availabel_relations) ^ set(exist_relations)
    other_relations = []
    for pair in list(other_ids): 
        c1 = llms_companies[ids_llms[pair[0]]]
        c2 = llms_companies[ids_llms[pair[1]]]    
        other_relations.append((c1,'other', c2))
        
    if len(llms_relations) == 0 and len(other_relations) == 0:
        other_relations = get_other_relations(org_groups, max_others)
        
    return llms_relations, other_relations
def create_re_dataset(
                      output,
                      matcher,
                      text_col:str="sentence",
                      relations_key:str='relations',
                      threshold:float=0.9,
                      max_others:int=3,
                      basic_columns:list=[],
                      ) -> pd.DataFrame:
    """
    Create a relation extraction dataset.

    @params
    -------
    - output: A pandas dataframe containing the following keys:
        * sentence: A string representing a sentence containing relevant text.
        * relations: A list of tuples representing the relationship between companies.
    - matcher: An instance of the `StringMatcher` class.
    - text_col: The key in the output dataframe where the text to match can be found.
    - relations_key: The key in the output dataframe where the relations can be found.
    - threshold: The similarity threshold for matching company names.

    @returns
    --------
    - dataset: A pandas dataframe containing the following columns:
    - idx: A unique identifier for the datapoint.
    - sentence: A string representing the sentence containing relevant text.
    - entity_2: A string representing the second entity in the relation.
    - relation: A string representing the relation between the two entities.
    - entity_1: A string representing the first entity in the relation.
    """

    # Apply the `extract_relations_from_llm` function to each datapoint in the output dataframe
    results = output.apply(lambda x: extract_relations_from_llm(datapoint=x,
                            matcher=matcher,
                            text_col=text_col,
                            relations_key=relations_key,
                            threshold=threshold), axis=1)

    # Create new columns in the output dataframe to store the results of the `extract_relations_from_llm` function
    output['llm_relations']  = results.apply(lambda x : x[0])
    output['other_relations']  = results.apply(lambda x : x[1])
    relation_columns = ['llm_relations', 'other_relations']
    # Create a list of all possible pair-wise combinations of the values in 'ids2org', and randomly choose 5 of those combinations
    columns = relation_columns + [text_col] + basic_columns
    re_dataset = []
    for i, row in output[columns].iterrows():
        row = row.to_dict()
        for r_column in relation_columns:
            # Iterate over relations and ingest row for each relation
            for relation_tuple in row[r_column]:
                row['entity_2'] = relation_tuple[0]
                row['relation'] = relation_tuple[1]
                row['entity_1'] = relation_tuple[2]
                re_dataset.append(dict(row))

    # Return a pandas dataframe containing the relation extraction dataset
    dataset = pd.DataFrame(re_dataset)[[ text_col,
                                        'entity_2',
                                        'relation',
                                        'entity_1'] + basic_columns]
    return dataset

# method to add Label based on the relations columns
def sc_label_from_relations(relation_tuples, main_relations):
    if not relation_tuples:
        return 0
    if len(relation_tuples) == 0:
        return 0
    for relation_tuple in relation_tuples:
        if len(relation_tuple) != 3:
            continue
        elif  relation_tuple[1] in main_relations:
            return 1
    return 0


def eval_relations(relation, default=[]):
    try:
        re = eval(relation)
    except:
        re = default
    return re

def eval_relation_data(output, relations_map):
    # Resort the sme_relations to unify the relations directions
    if not isinstance(output['sme_relations'].iloc[0], list):
        tqdm.pandas(desc="Eval sme_relations")
        output['sme_relations'] = output['sme_relations'].progress_apply(eval)
        
        
    if 'relations' in output.columns:
        if not isinstance(output['relations'].iloc[0], list):
            tqdm.pandas(desc="Eval relations")
            output['relations'] = output['relations'].progress_apply(eval_relations)
    else:
        output['relations'] = None
    if not isinstance(output['org_groups'].iloc[0], dict):
        tqdm.pandas(desc="Eval org_groups")
        output['org_groups'] = output['org_groups'].progress_apply(eval_relations, default={})  
    tqdm.pandas(desc="Resort sme relations")
    output['sme_relations'] = output['sme_relations'].progress_apply(lambda x:\
                              resort_relation((x[0], x[1], x[2]),
                                            relations_map))

