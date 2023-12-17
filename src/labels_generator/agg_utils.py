from tqdm import tqdm
from collections.abc import Iterable as iterable
from typing import List, Text,Tuple, Iterable, Dict
from itertools import chain
import random
from collections.abc import Iterable as iterable

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

def check_relation_tuples(relations: List[Iterable]) -> bool:
    """
    Check if the relations list is in the correct format.
    """
    if not all(isinstance(relation, iterable) and len(relation) == 3 for relation in relations):
        return False
    return True

def return_possible_pairs(ids_set:List):
    return list(zip(
            list(chain(*[[ids_set[x]]*(len(ids_set)-1-x) \
                 for x in range(len(ids_set))])),
            list(chain(*[[ids_set[i] for i in range(x+1, len(ids_set))]\
                 for x in range(len(ids_set))]))))


def get_other_relations(ids2org):
    """
    Returns a list of other relations between companies based on the dictionary of company groups passed as input.
    The maximum number of other relations is determined by the max_others parameter.

    @params
    ids2org
    org_groups (dict): A dictionary with company ids to map each group of ents.
    max_others (int): The maximum number of other relations to return.

    @returns
    --------
    list: A list of other relations between companies.
    """
    # Sort company keys (IDs) in ascending order
    comp_keys = sorted(ids2org.keys())

    # Generate all possible pairs of IDs
    other_ids = set(return_possible_pairs(comp_keys))

    # Create 'other_relations' tuples for each pair of IDs and return
    return [(ids2org[pair[0]][0] , 'other', ids2org[pair[1]][0]) for pair in other_ids]



def eval_relations(relation, default=[]):
    try:
        re = eval(relation)
    except:
        re = default
    return re

def eval_relation_data(output, relations_map):
    """
    Evaluate and process relation data in the output DataFrame.

    @params:
    - output: DataFrame containing relation data.
    - relations_map: Dictionary mapping relations.

    @returns:
    - None (modifies 'output' DataFrame in place).
    """
    # Resort sme_relations to unify the relations directions
    if not isinstance(output['sme_relations'].iloc[0], list):
        tqdm.pandas(desc="Eval sme_relations")
        output['sme_relations'] = output['sme_relations'].progress_apply(eval)

    # Evaluate and process 'relations' column if it exists
    if 'relations' in output.columns:
        if not isinstance(output['relations'].iloc[0], list):
            tqdm.pandas(desc="Eval relations")
            output['relations'] = output['relations'].progress_apply(eval_relations)
    else:
        # If 'relations' column doesn't exist, create it with None values
        output['relations'] = None
    
    # Evaluate and process 'org_groups' column
    if not isinstance(output['org_groups'].iloc[0], dict):
        tqdm.pandas(desc="Eval org_groups")
        output['org_groups'] = output['org_groups'].progress_apply(eval_relations, default={})
    
    # Resort sme_relations based on the provided 'relations_map'
    tqdm.pandas(desc="Resort sme relations")
    output['sme_relations'] = output['sme_relations'].progress_apply(lambda x: \
                              resort_relation((x[0], x[1], x[2]),
                                            relations_map))

def resort_relation(relation_tuple:Tuple, relations_map:Dict)->Tuple:
    """Resorts a tuple to match the order of the main relation."""
    c1, relation, c2 = relation_tuple
    return [c1, relation, c2]\
            if not relations_map.get(relation)\
            else [c2, relations_map.get(relation), c1]

