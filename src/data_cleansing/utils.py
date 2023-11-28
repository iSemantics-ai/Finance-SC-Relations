from collections import defaultdict
import random
import pandas as pd

def extract_llms_relations(data: pd.DataFrame, task_config: dict = {"llms_samples": 2, "other_samples": 2}) -> pd.DataFrame:
    """
    Extracts low-level market structure (LLMS) relations from a DataFrame of data.

    Args:
        data (pd.DataFrame): A pandas DataFrame containing the data from which to extract LLMS relations.
        task_config (dict): A dictionary containing the parameters for the task, including the number of samples to take for LLMS and other relations.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the extracted LLMS relations.
    """
    all_relations = []
    for i, datapoint in data.iterrows():
        if datapoint['relations'] == 'undefined' or len(datapoint['llms_relations']) == 0 and datapoint['Label'] == 0:
            org_groups = datapoint['org_groups']
            # Create a dictionary called 'ids2org' that maps each value in 'org_groups' to a list of keys that have that value
            ids2org = defaultdict(lambda : [])
            for key ,val in org_groups.items():
                ids2org[val].append(key)
            # Create a list of all possible pair-wise combinations of the values in 'ids2org', and randomly choose 5 of those combinations
            availabel_relations = []
            comp_keys = list(ids2org.keys())
            for i in range(len(comp_keys)):
                for j in range(i+1, len(comp_keys)):
                    relation_t = comp_keys[i], comp_keys[j]
                    availabel_relations.append(relation_t)


            # For each of the 5 chosen combinations, randomly choose one key from each group of 'org_groups' that corresponds to the two values in the combination
            if len(availabel_relations) > 0:
                req_samples = task_config.get('llms_samples', 2)
                number_of_sample =  req_samples if len(availabel_relations) > req_samples else len(availabel_relations)
                for relation in random.sample(availabel_relations, k = number_of_sample):
                    company1 = random.choice(ids2org[relation[0]])
                    company2 = random.choice(ids2org[relation[1]])
                    all_relations.append({"sents":datapoint['sentence'],
                                         "entity_2":company2,
                                         "relations":"other",
                                         "entity_1":company1,
                                         "org_groups": datapoint['org_groups'],
                                          "spans":datapoint['spans'],
                                         "sentence_id": 11111+i
                                        })
        else:
            req_samples = task_config.get('other_samples', 2)
            number_of_sample =  req_samples if len(datapoint['other_relations']) > req_samples \
                                            else len(datapoint['other_relations'])
            other_sample = random.sample(datapoint['other_relations'], k =number_of_sample)\
                            if number_of_sample > 0 else []

            req_samples = task_config.get('llms_samples', 2)
            number_of_sample =  req_samples if len(datapoint['llms_relations']) > req_samples \
                                            else len(datapoint['llms_relations'])
            llms_sample = random.sample(datapoint['llms_relations'], k =number_of_sample)\
                            if number_of_sample > 0 else []

            for relation in llms_sample + other_sample:
                mapped_rel = map_relation(relation[1], concepts)
                if datapoint['Label'] == 0 and mapped_rel != 'other':
                    continue
                all_relations.append({"sents":datapoint['sentence'],
                                      "entity_2": relation[0],
                                      "relations": mapped_rel,
                                      "entity_1": relation[2],
                                      "org_groups": datapoint['org_groups'],
                                      "spans": datapoint['spans'],
                                      "sentence_id": 11111+i
                })
    # Remove duplicates and return DataFrame
    weaklabels_frame = pd.DataFrame(all_relations)\
                        .drop_duplicates(['sents','entity_2', 'entity_1' , 'relations'])\
                        .reset_index(drop=True)
    return weaklabels_frame

def rel_from_text(sent):
    e1_start = '[E1] '
    e1_end = ' [/E1]'
    e2_start = '[E2] '
    e2_end = ' [/E2]'
    e1 = sent[sent.find(e1_start)+len(e1_start):sent.rfind(e1_end)]
    e2 = sent[sent.find(e2_start)+len(e2_start):sent.rfind(e2_end)]
    for tag in [e1_start,e1_end,e2_start,e2_end]:
        sent = sent.replace(tag, '')
    return pd.Series({"entity_1":e1, 'entity_2':e2})

def map_relation(relation, concepts):
    for label, concepts in concepts.items():
        for concept in concepts:
            if concept in relation:
                return label

def process_weaklabels(task_config, concepts):
    """
    Process the weak labels for a given task configuration and concept set.

    Args:
        task_config (dict): The configuration for the task, including weak labels.
        concepts (set): The set of concepts to map relations to.

    Returns:
        pandas.DataFrame: A DataFrame of processed weak labels, with duplicates removed.

    Example:
        >>> task_config = {'weaklables': ['version1', 'version2'], 'other_samples': 2, 'llms_samples': 2}
        >>> weaklabels_frame = process_weaklabels(task_config, concepts)
    """
    all_versions = [pd.concat([pd.read_json(os.path.join(dir_path, f)) for f in files], axis=0)
                    for version in task_config.get('weaklables')
                    for dir_path, _, files in os.walk(f'data/llms_datasets/templates/{version}/labels')]

    all_weaklabels = pd.concat(all_versions, axis=0).reset_index(drop=True)

    all_relations = []
    other_samples = task_config.get('other_samples', 2)
    llms_samples = task_config.get('llms_samples', 2)
    for row in all_weaklabels.itertuples(index=False):
        if row.relations == 'undefined' or (not row.llms_relations and row.Label == 0):
            org_groups = row.org_groups
            ids2org = defaultdict(list)
            for key, val in org_groups.items():
                ids2org[val].append(key)

            comp_keys = list(ids2org.keys())
            availabel_relations = [(comp_keys[i], comp_keys[j]) for i in range(len(comp_keys)) for j in range(i+1, len(comp_keys))]
            if availabel_relations:
                number_of_sample = min(other_samples, len(availabel_relations))
                for relation in random.sample(availabel_relations, k=number_of_sample):
                    company1 = random.choice(ids2org[relation[0]])
                    company2 = random.choice(ids2org[relation[1]])
                    all_relations.append({"sents":row.sentence,
                                         "entity_2":company2,
                                         "relations":"other",
                                         "entity_1":company1,
                                         "org_groups": row.org_groups,
                                          "spans":row.spans,
                                         "sentence_id": 11111+row.index
                                        })
        else:
            llms_sample = random.sample(row.llms_relations, k=min(llms_samples,
                                                            len(row.llms_relations)))\
                                                            if row.llms_relations else []
            other_sample = random.sample(row.other_relations, k=min(other_samples,
                                                                len(row.other_relations)))\
                                                                if row.other_relations else []
            for relation in llms_sample + other_sample:
                mapped_rel = map_relation(relation[1], concepts)
                if row.Label == 0 and mapped_rel != 'other':
                    continue
                all_relations.append({"sents":row.sentence,
                                      "entity_2": relation[0],
                                      "relations": mapped_rel,
                                      "entity_1": relation[2],
                                      "org_groups": row.org_groups,
                                      "spans": row.spans,
                                      "sentence_id": 11111+row.index
                })

    weaklabels_frame = pd.DataFrame(all_relations)\
                        .drop_duplicates(['sents','entity_2', 'entity_1' , 'relations'])\
                        .reset_index(drop=True)

    return weaklabels_frame
   
