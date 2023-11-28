from typing import Dict, List, Tuple, Set, Text, Dict
from num2words import num2words
import textdistance
import string
import re
from tqdm import tqdm
import random
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from collections import defaultdict
src_dir = Path.cwd().parent
sys.path.append(str(src_dir))
# from src.relation_extraction.preprocessing_funcs import entity_annotation

def initial_char(text):
    return [s[0] for s in text.split()] 

def hamming_search(query:str, text:str, ents):
    '''Search similarities in entities, based on hamming distance between two 
    entities, with more bias toward first and intitial charater simialrites

    @params
    -------
    query(str): the entity to search it's similar in text
    text(str): text that query meant to search inside 
    ent(str): NER tag to match with query
    '''
    query = query.lower().translate(str.maketrans('', '', string.punctuation))
    search = set(filter(None,ents))
    main_ents = list(search)
    clean_search = filter(None,[t.lower().translate(str.maketrans('', '', string.punctuation)) for t in search])
    results = [textdistance.hamming.normalized_similarity(query.split()[0], s.split()[0])+\
               textdistance.hamming.normalized_similarity(query, s)+\
               textdistance.hamming.normalized_similarity(query, initial_char(s))+\
               textdistance.hamming.normalized_similarity(initial_char(query), s)
               for s in clean_search\
            ]
    if len(results) == 0 : 
        return None, 0.0
    if max(results) <= 0.6:
        return None,max(results) 
    index = results.index(max(results))
    e = main_ents[index]
    return e,max(results)

def word_search(word:str, text:str):
    """Extarct start&end char span for specific word in a text.

    @params
    -------
    - word (str): query word to search
    - text (str): Text to be searched

    @returns
    --------
    - List with spans tuple(start,end) 
        
    """
    return [(ele.start(), ele.end()) \
            for ele in re.finditer(re.escape(word.lower()),
            text.lower())] if word is not None else []
def Intersection(lst1:List, lst2:List)->Set:
    """Intersection between two lists

    @params:
    -------
    - lst1 (List)
	- lst2 (List)

    Returns:
        Set of intercetion
    """
    return set(lst1).intersection(lst2)



def word_search(word:str, text:str):
    """Extarct start&end char span for specific word in a text.

    Args:
        word (str): query word to search
        text (str): Text to be searched

    Returns:
        List with spans tuple(start,end) 
        
    """
    pos_pairs =  [(ele.start(), ele.end()) \
            for ele in re.finditer(re.escape(word.lower()),
            text.lower())] if word is not None else []
    random.shuffle(pos_pairs)
    return pos_pairs
def Intersection(lst1:List, lst2:List)->Set:
    """Intersection between two lists

    Args:
        lst1 (List)
        lst2 (List)

    Returns:
        Set of intercetion
    """
    return set(lst1).intersection(lst2)

def entity_annotation(sent:Text,
                      ent1:Text,
                      ent2:Text,
                      org_groups: dict,
                      num_positions= np.inf,
                      matcher=None):
    """Create an entity relation extraction model by inserting entity tags around
    each entity token given in dataframe rows.

    @params
    -------
    - sent (str): The input sentence to annotate.
    - ent1 (str): The first entity.
    - ent2 (str): The second entity.
    - org_groups (dict): A dictionary containing organization groups.
    - num_positions (int): number of positions should be returned between Company X and Company Y.
    - matcher: If matcher model exist use it to correct the entity names in case some changes happened

    @returns
    --------
    - list: A list of annotated sentences.

    """
    if matcher is not None:  
        orgs = list(org_groups.keys())
        # Match with org_groups in case no exact match founded
        if ent2 not in orgs:
            matches = np.argwhere(matcher.similarity(ent2, orgs)>0.9)
            if matches.shape[0] > 0:
                ent2 = orgs[matches[0][0]]
        if ent1 not in orgs:
            matches = np.argwhere(matcher.similarity(ent1, orgs)>0.9)
            if matches.shape[0] > 0:
                ent1 = orgs[matches[0][0]]
    # Transform org_groups to id2org
    id2org = defaultdict(lambda:[])
    for k,v in org_groups.items():
        id2org[v].append(k)

    # Sort and retrieve entity names for ent1
    ent1_names = id2org[org_groups.get(ent1)] if \
    org_groups.get(ent1) is not None else [ent1]
    ent1_names.sort(reverse=True)

    # Sort and retrieve entity names for ent2
    ent2_names = id2org[org_groups.get(ent2)] if \
    org_groups.get(ent2) is not None else [ent2]
    ent2_names.sort(reverse=True)

    # Create a dictionary of entity names and their corresponding tags
    names = {**{k:1 for k in ent1_names },
             **{k:2 for k in ent2_names }}
    sorted_names = dict(sorted(names.items(),  key=lambda x : len(x[0]), reverse=True))
    # Replace entity names with entity tags in the given sentence
    for k, v in sorted_names.items():
        k  = k.replace('’', "'")
        if k.strip().endswith("'s") : # or k.strip().endswith('’s')
            k = k.replace("'s", '')
            sent = sent.replace(k, "<entity_{}> ".format(v))
        else:
            sent = sent.replace(k, "<entity_{}>".format(v))

    sentence, e1, e2 = sent, '<entity_1>', '<entity_2>'
    res1 = word_search(e1, sentence)
    res2 = word_search(e2, sentence)

    sentences = []
    relations_num= 0
    # Iterate through the results of entity1 and entity2 in the sentence
    for j, r1 in enumerate(res1):
        s = sentence[: r1[0]] + "[E1] " + sentence[r1[0] :]
        s = s[0 : (r1[1] + 5)] + " [/E1]" + s[(r1[1] + 5) :]

        res1 = word_search(e1, s)
        r1 = res1[j]
        res2 = word_search(e2, s)

        # Iterate through the results of entity2 in the modified sentence
        for i, r in enumerate(res2):
            
            intersec = Intersection(list(range(r[0], r[1])), list(range(r1[0], r1[1])))

            # Check for intersection between entity1 and entity2
            if len(intersec) > 0:
                continue

            if r[0] < r1[0]:
                r2 = r[0], r[1]
            else:
                r2 = r[0], r[1]

            intersec = Intersection(
                list(range(r2[0], r2[1])), list(range(r1[0], r1[1]))
            )

            # Check for intersection between entity1 and entity2 after modifying entity2
            if len(intersec) > 0:
                continue

            # Replace entity tags with entity names in the modified sentence
            out = s[: r2[0]] + "[E2] " + s[r2[0] :]
            out = out[0 : (r2[1] + 5)] + " [/E2]" + out[(r2[1] + 5) :]

            out = out.replace("<entity_1>", ent1)
            out = out.replace("<entity_2>", ent2)

            sentences.append(out)
            relations_num +=1
            if relations_num >= num_positions :
                return sentences

    return sentences

def create_re_data(dataframe: pd.DataFrame,
                   text: Text,
                   relation_source: Text,
                   relation_target: Text,
                   label: Text,
                   inverse_dict:dict,
                   static_position:bool = False,
                   num_positions:int= np.inf,
                   use_matcher:bool=False) -> pd.DataFrame:
    """
    Extracts sentences with two entities, where one entity is the relation source
    and the other is the relation target. Adds tags to mark the positions of the entities in the
    sentences and creates a new dataframe with the tagged sentences and the corresponding relation labels.

    @params:
    --------
    - dataframe: The input dataframe.
    - text: The column name of the text to be processed in the dataframe.
    - relation_source: The column name of the relation source in the dataframe.
    - relation_target: The column name of the relation target in the dataframe.
    - label: The column name of the relation label in the dataframe.
    - inverse_dict: A dictionary containing the inverse of each relation.
    - static_position: A flag indicating whether to use a static position for entity 1 or not.
    - num_positions: The maximum number of positions to consider.
    - use_matcher: A boolean value indicating whether to use a matcher model to link non-exact matches.

    @returns
    --------
    - A new dataframe with the tagged sentences and corresponding relation labels.
    """
    sentences = []
    orig_sents = []
    labels = []
    org_groups = []
    spans = []
    Labels = []
    concepts = []
    concept_class_remapped = []
    idxs = []
    r_id = []
    relation_tuples = []
    if use_matcher:
        from src.matcher.core import SimCSE_Matcher
        matcher = SimCSE_Matcher(model_name_or_path=str(src_dir / 'artifacts/matcher_model'))
    else: 
        matcher = None

    for i, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="Tagging poles of relations"):
   
        rel_source = row[relation_source]
        rel_target = row[relation_target]
        rel = row[label]
        sentence = row[text]
        relation_tuple = (rel_source, rel, rel_target)
        
        if not static_position:
            sents = entity_annotation(sentence,
                                      rel_target,
                                      rel_source,
                                      row['org_groups'],
                                      num_positions=num_positions,
                                      matcher=matcher)
            l_sents = len(sents)
            sentences+= sents
            orig_sents += [sentence] * l_sents
            labels += [rel] * l_sents
            spans += [row['spans']] * l_sents
            org_groups += [row['org_groups']] * l_sents
            idxs += [row.get('idx')] * l_sents
            r_id += ["{}_{}".format(str(row['idx']),str(i))] * l_sents
            Labels += [row.get('Label')] * l_sents
            concepts += [row.get('concept_class')] * l_sents
            concept_class_remapped += [row.get('concept_class_remapped')] * l_sents
            relation_tuples += [relation_tuple]  * l_sents
        
        else:
            n_relations = 0
            s_positions = set(word_search(rel_source, sentence))
            t_positions = set(word_search(rel_target, sentence))
            for si, s_pos in enumerate(s_positions):
                if n_relations > 0 and n_relations > num_positions:
                    break
                for ti, t_pos in enumerate(t_positions):

                    intersec = Intersection(list(range(s_pos[0], s_pos[1])), list(range(t_pos[0], t_pos[1])))
                    if len(intersec) > 0:
                        continue
                    action = "default"

                    if s_pos[0] > t_pos[0]:
                        ent1 = t_pos
                        relation = inverse_dict[rel]
                        entity2_name = rel_source
                        action = "inverse"
                    else:
                        ent1, entity2_name, relation = s_pos, rel_target, rel
                    s = sentence[:ent1[0]] + '[E1] ' + sentence[ent1[0]:]
                    s = s[0: (ent1[1] + 5)] + ' [/E1]' + s[(ent1[1] + 5):]

                    try:
                        ent2 = word_search(entity2_name, s)[si] if action == 'inverse' \
                            else word_search(entity2_name, s)[ti]
                    except:
                        continue

                    out = s[:ent2[0]] + '[E2] ' + s[ent2[0]:]
                    out = out[0: (ent2[1] + 5)] + ' [/E2]' + out[(ent2[1] + 5):]
                    n_relations += 1
                    # Append the required values
                    sentences.append(out)
                    orig_sents.append(sentence)
                    labels.append(relation)
                    spans.append(row['spans'])
                    org_groups.append(org_groups)
                    idxs.append(row.get('idx'))
                    Labels.append(row.get('Label'))
                    concepts.append(row.get('concept_class'))
                    concept_class_remapped.append(row.get('concept_class_remapped'))
                    relation_tuples.append(relation_tuple)
                    if n_relations > num_positions:
                        break

    return pd.DataFrame({'sents': sentences,
                         "orig_sents": orig_sents,
                         'relations': labels,
                         'org_groups': org_groups,
                         'spans': spans,
                         'Label': Labels,
                         'concept_class': concepts,
                         'concept_class_remapped': concept_class_remapped,
                         'idx': idxs,
                         "r_id": r_id,
                         "relation_tuples": relation_tuples
                         })
def spread_rows(data, indices, index_col):
    """
    This function takes a Pandas DataFrame, a list of indices, and an index column name as input,
    and returns a new DataFrame that contains the rows corresponding to the given indices,
    with each row in the original DataFrame expanded into multiple rows if it contains
    multiple values in one or more columns.

    @params
    -------
    - data (pd.DataFrame): The input DataFrame
    - indices (list): The indices of the rows to select
    - index_col (str): The name of the index column in the original DataFrame

    @return
    -------
    - pd.DataFrame: The new DataFrame with expanded rows
    """
    # Set the input DataFrame as the index of the new DataFrame
    dataset = data.set_index(index_col)

    # Initialize an empty list to store the expanded rows
    spreaded_rows = []

    # Loop over the given indices and select the corresponding rows from the input DataFrame
    for idx in tqdm(indices, total=len(indices), desc='sort dataset by indices'):
        rows = dataset.loc[idx]

        # If the selected rows contain only one row, convert it to a dictionary and add it to the list of expanded rows
        if isinstance(rows, pd.core.series.Series):
            rows = rows.to_dict()
            rows[index_col] = idx
            spreaded_rows += [rows]

        # If the selected rows contain multiple rows, convert them to a list of dictionaries and add them to the list of expanded rows
        else:
            spreaded_rows += rows.reset_index().to_dict('records')

    # Create a new DataFrame from the list of expanded rows and return it
    return pd.DataFrame(spreaded_rows)


def split_data(data:pd.DataFrame,
               index_col:str,
               stratify_by:List[str],
               val_size:int = 0.20,
               random_state:int = 1
              ):
    """
    This function splits the input DataFrame into train and validation sets using
    stratified sampling with the specified stratification columns.

    @params:
    --------
    - data (pd.DataFrame): The input DataFrame to split
    - index_col (str): The name of the index column in the DataFrame
    - stratify_by (List[str]): A list of column names to use for stratification
    - val_size (float): The proportion of the data to use for validation (default: 0.2)
    - random_state (int): The random seed to use for reproducibility (default: None)

    @return
    -------
    - pd.DataFrame, pd.DataFrame: The train and validation DataFrames
    """

    # Group the data by the index column and the stratification columns, and convert to a DataFrame
    meta_data = pd.DataFrame(
        data.groupby(index_col)\
        .apply(lambda x : {**{ index_col:x[index_col].iloc[0]},
                           **{z:x[z].iloc[0] for z in stratify_by}})\
        .to_list()
    )

    # Split the meta data into train and validation sets using stratified sampling
    train, valid = train_test_split(meta_data,
                                     test_size=val_size,
                                     random_state=random_state,
                                     shuffle=True,
                                     stratify=meta_data[stratify_by])

    # Expand the rows in the train and validation sets based on the index column
    train = spread_rows(data=data,
                indices=train[index_col],
                index_col=index_col)

    valid = spread_rows(data=data,
                indices=valid[index_col],
                index_col=index_col)

    return train, valid

def get_source(ids_map, data_sources, idx):
    """
    Retrieve the data source and index of a given ID.

    @params
    -------
    ids_map: A dictionary mapping IDs to data source and index.
    data_sources: A dictionary containing the data sources.
    idx: The ID of the data.

    @return
    -------
    A tuple containing the data source and index of the given ID.
    """
    name = ids_map[idx]
    # Extract the key name and index from the ID.
    key_name, index = '_'.join(name.split("_")[:-1]), int(name.split("_")[-1])
    # Retrieve the data source from the dictionary.
    source = data_sources[key_name]
    return (source, index)


def mutate_sent(sent:str, spans:List[str]=None, org_groups:Dict[str, int]=None) -> str:
    """
    Replace each organization into an ID.

    @params
    -------
    - sent (str): The input sentence.
    - spans (List[str]): A list of spans.
    - org_groups (Dict[str, int]): A dictionary of organization groups.

    @returns
    --------
    - str: The mutated sentence.
    """
    if org_groups:
        org_list = list(org_groups.items())
        org_list.sort(key=lambda x:len(x[0]),reverse=True)
        org_dict = {ele[0] : ele[1]  for ele in org_list}
        ids = set(org_dict.values())
        ids_shift = {k:random.randint(0, 100) for k in ids}
        for org, org_id in  org_dict.items():
            if len(org) > 3:
                sent = re.sub(re.escape(org), f'org-{num2words(ids_shift[org_id]).lower()}', sent)
#                 rand_word = ''.join(random.sample(string.ascii_letters, random.choice(list(range(6)))))
#                 sent = re.sub(re.escape(org), f'ORG-{rand_word.upper()}', sent)
    return sent
