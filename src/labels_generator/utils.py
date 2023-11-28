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
import openai 


 
def deserialize_relations(ser_relations):    
    # the string representation of the list of dictionaries
    string_list_of_dicts = ser_relations
    # regular expression to match a dictionary
    dict_regex = r"\{[^{}]+\}"
    # find all dictionaries in the string
    dict_strings = re.findall(dict_regex, string_list_of_dicts)
    # deserialize each dictionary into a Python object
    list_of_dicts = []
    for dict_string in dict_strings:
        try:
            list_of_dicts.append(json.loads(re.sub(r"(?<!\w)'|'(?!\w)", '"', dict_string)))
        except:
            try:
                list_of_dicts.append(json.loads(re.sub(r"(?<!\w)'|'(?!\w)", '"', dict_string.replace('"', '\\"'))))
            except:
                continue
    return list_of_dicts

def get_completion(prompt:Text,
                        temperature:float=0,
                        model="gpt-3.5-turbo")->str:
    messages = [{"role": "user", "content": prompt}]
    response = None
    while not response:
        try:
            response = openai.ChatCompletion.create(
                model= model,
                messages=messages,
                temperature= temperature    #this is the degree of randomness of the model's output
            )
        except:
            time.sleep(0.2)
        
    return response.choices[0].message["content"]


def resort_relation(relation_tuple:Tuple, relations_map:Dict)->Tuple:
    """Resorts a tuple to match the order of the main relation."""
    c1, relation, c2 = relation_tuple
    return [c1, relation, c2]\
            if not relations_map.get(relation)\
            else [c2, relations_map.get(relation), c1]

def relations_tupled(relations_list:dict, relations_map:map):
    """Converts a list of relations into a list of tuples.

    Args:
    relations_list (list): A list of dictionaries containing the relation information.
    relations_map (map): A map containing the inverse relations.

    Returns:
    list: A list of tuples containing the relation information.
    """
    relations_tuples = []

    for relation in relations_list:
        c1 = relation.get('company_1')
        c2 = relation.get('company_2')
        relation = relation.get('relation')
        if not all([c1,c2,relation]):
            continue
        
        relation = 'supplier' if 'supplier' in relation.lower() else relation
        relation_tuple = resort_relation((c1,relation,c2), relations_map)
        relations_tuples.append(relation_tuple)
    return relations_tuples






def generate_relations(data: pd.DataFrame,
                       prompt: Text,
                       replaces: dict,
                       relations_map:dict,
                       )-> pd.DataFrame:
    """
    Generate relations for each row of data using OpenAI's GPT-3 API and a given prompt.

    @params:
    --------
    data (pd.DataFrame): DataFrame containing the data to extract relations from.
    prompt (str): The prompt to use for generating the relation for each row of data.
    replaces (dict): A dictionary containing the keys to replace in the prompt with corresponding values from the data.
    relations_map (dict): Map of relations to unify the relations directions
    
    @returns:
    ---------
    pd.DataFrame: A DataFrame containing the original data with an additional column for the extracted relations.
    """
    output = []
    # Iterate over the frame rows
    for i, row in tqdm(data.iterrows(), total=data.shape[0], desc="Generating relations"):
        row = row.to_dict()
        row['index'] = i
        report_prompt = copy(prompt)
        for k, v in replaces.items():
            report_prompt = report_prompt.replace(v, row.get(k))
        # Generate relations and parse it
        completion = get_completion(prompt=report_prompt)
        try:
            relations_list = deserialize_relations(completion)
            row['relations'] = relations_tupled(relations_list, relations_map)
            row['relation_completion'] = completion

        except:
            row['relations'] = "NotParsed"
            row['completion'] = completion
        # Append the output list with modified row
        output.append(row)
    return pd.DataFrame(output).set_index('index')




def deserialize_json_dict2(json_dict_str):
    json_dict = {}    
    
    try:
        json_dict = json.loads(re.sub(r"(?<!\w)'|'(?!\w)", '"', json_dict_str))
    except:
        traceback.print_exc()
    
    if not json_dict:
        try:
            json_dict = json.loads(re.sub(r"(?<!\w)'|'(?!\w)", '"', json_dict_str.replace('"', '\\"')))
        except:
            traceback.print_exc()           
    
    if not json_dict:
        try:
            json_dict = json.loads(re.sub(r"(?<!\w)'|'(?!\w)", '"', json_dict_str.replace('"', '\\"')))
        except:           
            traceback.print_exc()
    if not json_dict:
        try:
            for i,c in enumerate(json_dict_str[::-1]):
                if c == '}':
                    solve = json_dict_str[:-i] + ']}'
                    break
            json_dict = deserialize_json_dict2(solve)
        except:           
            traceback.print_exc()

    return json_dict


def relations_tupled_2(relations_list:dict, relations_map:map):
    """Converts a list of relations into a list of tuples.

    Args:
    relations_list (list): A list of dictionaries containing the relation information.
    relations_map (map): A map containing the inverse relations.

    Returns:
    list: A list of tuples containing the relation information.
    """
    relations_tuples = []

    for relation in relations_list.get('supplier_and_customer', []):
        c1 = relation.get('customer')
        c2 = relation.get('supplier')
        relation_tuple = [c2, 'supplier', c1]
        if not all(relation_tuple):
            continue
        relations_tuples.append(relation_tuple)

    for relation in relations_list.get('financial_trade', []):
        c1 = relation[0] if len(relation) == 2 else ""
        c2 = relation[1] if len(relation) == 2 else ""
        relation_tuple = [c1, 'financial_trade', c2]
        if not all(relation_tuple):
            continue
        relations_tuples.append(relation_tuple)
    
    for relation in relations_list.get('nothing', []):
        c1 = relation[0] if len(relation) == 2 else ""
        c2 = relation[1] if len(relation) == 2 else ""
        relation_tuple = [c1, 'nothing', c2]
        if not all(relation_tuple):
            continue
        relations_tuples.append(relation_tuple)
        
    return relations_tuples    


def generate_relations_with_explanation(data: pd.DataFrame,
                       explanation_prompt: Text,
                       relation_prompt: Text,
                       explantion_replaces: dict,
                       relation_replaces: dict,
                       relations_map:dict,
                       deserialize_func,
                       tuple_func,
                       do_explain = True,
                       do_relation = True

                       )-> pd.DataFrame:
    """
    Generate relations for each row of data using OpenAI's GPT-3 API and a given prompt.

    Args:
    data (pd.DataFrame): DataFrame containing the data to extract relations from.
    prompt (str): The prompt to use for generating the relation for each row of data.
    replaces (dict): A dictionary containing the keys to replace in the prompt with corresponding values from the data.
    relations_map (dict): Map of relations to unify the relations directions
    Returns:
    pd.DataFrame: A DataFrame containing the original data with an additional column for the extracted relations.
    """
    output = []
    # Iterate over the frame rows
    for i, row in tqdm(data.iterrows(), total=data.shape[0], desc="Generating relations"):
        row = row.to_dict()
        row['index'] = i
        exp_prompt = copy(explanation_prompt)
        rel_prompt = copy(relation_prompt)

        for k, v in explantion_replaces.items():
            exp_prompt = exp_prompt.replace(v, row.get(k))
        
        if do_explain or (not do_explain and not row.get("explanation", False)):
            # Generate relations and parse it
            explanation = get_completion(prompt=exp_prompt)
            row['explanation'] = explanation
        for k, v in relation_replaces.items():
            rel_prompt = rel_prompt.replace(v, row.get(k))
        
        if do_relation or (not do_explain and not row.get("relation_completion", False)):
            relation_completion = get_completion(prompt=rel_prompt)
            row['relation_completion'] = relation_completion

        try:
            relations_list = deserialize_func(row['relation_completion'])
            row['relations'] = tuple_func(relations_list, relations_map)

        except:
            row['relations'] = "NotParsed"
            row['completion'] = relation_completion
        # Append the output list with modified row
        output.append(row)
    return pd.DataFrame(output).set_index('index')


def relation_search(query_relation: Tuple[str, str, str],
                     relations_tuples: List[Tuple[str, str, str]],
                     matcher: object,
                     threshold: float = 0.85,
                     main_relations: List[str] = ['supplier']) -> bool:
    """
    Search for a relation in a given list of relation tuples using, 

    Args:
    - query_relation: A tuple representing the relation to search for, consisting of a source entity,
                      a relation type, and a destination entity.
    - relations_tuples: A list of tuples representing relations, consisting of a source entity,
                        a relation type, and a destination entity.
    - matcher: CosineSim matcher using Attetion encoder and Faiss for quick matching
    - threshold: A float representing the minimum similarity needed to consider a match.
    - main_relations: A list of strings representing the main relation types to search for.

    Returns:
    - A boolean representing whether the relation was found in the list of relation tuples
    
    Example:
        llm_relations
        -------------
        ['ORBCOMM Inc', 'nothing', 'Systems Inc'],
        ['ORBCOMM Inc', 'nothing', 'inthinc Technology Solutions Inc'],
        ['ORBCOMM Inc', 'nothing', 'Value added Solutions Providers'],
        ['Onixsat', 'supplier', 'ORBCOMM Inc'],
        ['Satlink', 'supplier', 'ORBCOMM Inc'],
        ['Sascar', 'supplier', 'ORBCOMM Inc']
        -------------
        source_orgs = [Onixsat, Satlink, Sascar]
        dist_orgs = ['ORBCOMM Inc']
        -------------
        QUERY
        Onixsat, supplier ,ORBCOMM Inc
        -------------
        s2d
        if relation == supplier:
            s2d = 'Onixsat' in `source_orgs` && 'ORBCOMM Inc' in `dist_orgs`
            return s2d
        else: # Means Other
            d2s = 'Onixsat' in `dist_orgs` && 'ORBCOMM Inc' in `source_orgs`
            return TRUE if d2s & s2d both == False
        -------------
    """

    align = False
    source_orgs = defaultdict(lambda: [])
    dist_orgs = defaultdict(lambda: [])
    
    if relations_tuples:
        # Loop over each relation tuple to create lists of source and destination entities
        for relation_tuple in relations_tuples:
            if isinstance(relation_tuple, str):
                continue
            relation = relation_tuple[1]
            if relation in main_relations:
                source_orgs[relation_tuple[0]].append(relation_tuple[2])
                dist_orgs[relation_tuple[2]].append(relation_tuple[0])
    source_list = list(source_orgs.keys())
    dist_list = list(dist_orgs.keys())

    s2d = False
    d2s = False
    # Check for a match in the source entities
    if source_orgs:
        source_sim = matcher.similarity(query_relation[0], source_list)
        max_idx, max_score = source_sim.argmax(), source_sim.max()
        source_match = max_score > threshold or any([query_relation[0].lower() in x.lower() for x in source_list])

        if source_match:
            dist_match = True if matcher.similarity(query_relation[2],
                                                    source_orgs[source_list[max_idx]]).max()\
                                                    > threshold or\
                                                    any([query_relation[2].lower() in x.lower()\
                                                    for x in source_orgs[source_list[max_idx]]])\
                                                    else False
            s2d = all([source_match, dist_match])
    # Check for a match in the destination entities
    if query_relation[1] in main_relations:
        return s2d
    elif dist_orgs:
        dist_sim = matcher.similarity(query_relation[0], dist_list)
        max_idx, max_score = dist_sim.argmax(), dist_sim.max()
        dist_match = max_score > threshold or any([query_relation[0].lower() in x.lower() for x in dist_list])
        if dist_match:
            source_match = True if matcher.similarity(query_relation[2],
                                                    dist_orgs[dist_list[max_idx]]).max()> threshold\
                                                    or any([query_relation[2].lower()\
                                                            in x.lower()\
                                                            for x in dist_orgs[dist_list[max_idx]]])\
                                                    else False
            d2s = all([source_match, dist_match])
    return not any([s2d, d2s])


# create sme_tuple for each example:
def create_sorted_relation(source,
                           relation,
                           dist,
                           label,
                           relations_map,
                           ):
    if label == 0:
        relation = "other"
    
    relation_tuple = (source,
                relation,
                dist)
    return resort_relation(relation_tuple, relations_map=relations_map)



def check_relation_tuples(variable: any) -> bool:
    if isinstance(variable, List):
        if all(isinstance(item, Tuple) \
            and len(item) == 3 \
            and all(isinstance(element, str) for element in item)\
                  for item in variable):
            return True
    return False


def extract_relations_from_llm(datapoint,
                                matcher,
                                feature_key:str="sentence",
                                relations_key:str='relations',
                                threshold:float=0.9):
    
    """
    Create a dataset for relation extraction training.

    Args:
    - datapoint: A dictionary containing the following keys:
        - org_groups: A dictionary of company names associated with an integer identifier.
        - relations: A list of tuples representing the relationship between companies.
    - matcher: An instance of the `StringMatcher` class.
    - feature_key: The key in the datapoint dictionary where the text to match can be found.
    - relations_key: The key in the datapoint dictionary where the relations can be found.
    - threshold: The similarity threshold for matching company names.

    Returns:
    - llms_relations: A list of tuples representing the relationships between companies that were successfully matched.
    - other_relations: A list of tuples representing the relationships between companies that were not matched.

    Raises:
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
    for relation in relations:
        llms_companies += [relation[0] , relation[2]]
    llms_companies = {k:None for k in set(llms_companies)}
    llms_ids = {k:i for i,k in enumerate(set(llms_companies))}
    ids_llms = {i:k for i,k in enumerate(set(llms_companies))}
    # Check if each company in the dictionary is mentioned in the sentence, and if not, try to match it with a known organization
    for company in list(llms_companies.keys()):
        if company in datapoint[feature_key]:
            llms_companies[company] = company
        else:
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

    # Create a list of all possible pair-wise combinations of the values in 'ids2org', and randomly choose 5 of those combinations
    availabel_relations = []
    comp_keys = list(ids2org.keys())
    for i in range(len(comp_keys)):
        for j in range(i+1, len(comp_keys)):
            relation_t = tuple(sorted([comp_keys[i], comp_keys[j]]))
            availabel_relations.append(relation_t)
    exist_relations = []
    llms_relations = []
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
    return llms_relations, other_relations


def create_re_dataset(output, matcher, feature_key:str="sentence", relations_key:str='relations', threshold:float=0.9) -> pd.DataFrame:
    """
    Create a relation extraction dataset.

    Args:
    - output: A pandas dataframe containing the following keys:
        - sentence: A string representing a sentence containing relevant text.
        - relations: A list of tuples representing the relationship between companies.
    - matcher: An instance of the `StringMatcher` class.
    - feature_key: The key in the output dataframe where the text to match can be found.
    - relations_key: The key in the output dataframe where the relations can be found.
    - threshold: The similarity threshold for matching company names.

    Returns:
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
                            feature_key=feature_key,
                            relations_key=relations_key,
                            threshold=threshold), axis=1)

    # Create new columns in the output dataframe to store the results of the `extract_relations_from_llm` function
    output['llm_relations']  = results.apply(lambda x : x[0])
    output['other_relations']  = results.apply(lambda x : x[1])

    # Create a list of all possible pair-wise combinations of the values in 'ids2org', and randomly choose 5 of those combinations
    relation_columns = ['llm_relations', 'other_relations']
    re_dataset = []
    for i, row in output.query("align == True")[relation_columns + [feature_key]].iterrows():
        row = row.to_dict()
        row['idx'] = i

        for r_column in relation_columns:
            # Iterate over relations and ingest row for each relation
            for relation_tuple in row[r_column]:
                row['entity_2'] = relation_tuple[0]
                row['relation'] = relation_tuple[1]
                row['entity_1'] = relation_tuple[2]
                re_dataset.append(dict(row))

    # Return a pandas dataframe containing the relation extraction dataset
    dataset = pd.DataFrame(re_dataset)[[ 'idx', feature_key, 'entity_2', 'relation', 'entity_1']]
    return dataset
