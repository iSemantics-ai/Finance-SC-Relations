# llm_annotator.py
import os
import openai
from glob import glob
from typing import List, Tuple, Union
from typing import Dict, Text
import json
import random
from collections import defaultdict
import pandas as pd
import time
import numpy as np
import yaml
import re
from colorama import Fore
from tqdm import tqdm
from pathlib import Path
import sys 
src_dir = Path.cwd().parent
sys.path.append(str(src_dir)) 
from src.matcher.core import SimCSE_Matcher
from src.utils import get_logger, dotdict



dataset_columns = ['sentence', 'org_groups']
valid_models =["gpt-3.5-turbo"]

main_relations = ['supplier', 'customer']
inverse = {"customer":"supplier", "supplier":"customer", "other":"other"}
explanation_tags = ["{sentence}", "{instructions}"]
labeling_tags = ["{explanation}"]
confirm_tags = ["{company1}", "{company2}", "{relation}" , "{explanation}"]

class LLMAnnotator(object):
    """This module contains code that generates prompt templates for Language Model
        APIs, such as OpenAI's GPT-3. The prompts are designed to help generate labeled
        datasets for training Relation Extraction models.

        The template consists of three prompts:

        1. Explanation prompt: This prompt is used to explain certain aspects of a given
        sentence. It is designed to help the user identify the entities and relations
        in the sentence.

        2. Label generation prompt: This prompt is responsible for generating a JSON
        object that contains the label for the given sentence. The label includes the
        type of relation between the entities and any additional information that may
        be relevant.

        3. Confirmation prompt: This prompt is used to curate the final label generated
        by the previous prompt. It allows the user to review and modify the label as
        needed.

    """
    def __init__(self, version, matcher_device='cpu'):
        self.logger = get_logger('\U0001F300 LLMAnnotator', log_level="INFO")
        versions_dirs = set(filter(None, [v if not '.dvc' in v else None for v in glob(str(src_dir / 'data/llms_datasets/templates/v*'))]))
        print("versions_dirs", versions_dirs, str(src_dir / 'data/llms_datasets/templates/v*'))
        
        self._versions = list(sorted([float(v.split('/')[-1][1:]) for v in versions_dirs]))
        # Validate the version
        if version not in self._versions:
            raise (f"Invalid template, The available _versions are {self._versions}")
        self.version = version
        self.logger.info("Loading template card...")
        # Reading template card
        with open(src_dir / f"data/llms_datasets/templates/v{str(version)}/card.yaml") as ob:
            self.card = dotdict(yaml.safe_load(ob))
        # Define conflict directory
        self._conflict_path = src_dir /f"data/llms_datasets/templates/v{self.version}/reports/conflict_sme_llm_trainset.json"
        # Load entity matcher
        self.matcher  = SimCSE_Matcher('sentence-transformers/all-MiniLM-L6-v2', device=matcher_device)

    @property
    def unlabeled(self):
        suffix = (src_dir / self.card['dataset']).suffix
        if  suffix == '.json':
            pd.read_json(src_dir / self.card['dataset'])
        elif suffix == '.xlsx':
            return pd.read_excel(src_dir / self.card['dataset'], index_col="index" )



    @property
    def conflicts(self):
        if os.path.isfile(self._conflict_path):
            return pd.read_json(self._conflict_path)
        else:
            self.logger.info("No conflicts had been detected the conflict file might be not exist or target dataset not labeled!!!")

    def get_completion(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        response = None
        while not response:
            try:
                response = openai.ChatCompletion.create(
                    model=self.card['model'],
                    messages=messages,
                    temperature=self.card['temperature'] # this is the degree of randomness of the model's output
                )
            except:
                time.sleep(0.2)
            
        return response.choices[0].message["content"]
    
    def update_template(self):
        new_template = self.card.copy()
        
        for item in new_template.keys():
            text = f"Insert {Fore.BLUE}`{item}`{Fore.RESET}\nPrevious one:\n{Fore.LIGHTCYAN_EX}{new_template[item]}\n"
            print(text)
            input_item = input()
            print(f"{'*'*50}\n")
            if input_item:
                new_template[item] = input_item
        if self.card == new_template:
            print("No changes founded")
        else:
            # TODO: validate the new template
            # If Vaild 
            # pprint the new template and ask  for confirmation
            #print("The new template\n", Fore.CYAN, json.dumps(new_template, indent=4),'\n', "*"*50)
            print("#### New Template ####\n######################\n")
            for k,v in new_template.items():
                print(f'{k}\n--------------\n{Fore.GREEN}{v}{Fore.RESET}\n{"*"*50}\n')
            confirm  = input("Confirm the changes?(y|n)")
            if confirm == 'y':
                new_version = round(max(self._versions)+0.1, 2)
                # Validate the data
                if self.card['dataset'] != new_template['dataset']:
                    # Read file
                    new_data = pd.read_json(new_template['dataset'])
                    # Check required columns
                    if not all([x in new_data.columns for x in dataset_columns]):
                        raise(f"Invalid dataset must contains {dataset_columns}")

                # Validate templates 
                if new_template['model'] not in valid_models:
                    raise Exception(f"invalid open-ai model, Valid Models: {valid_models}")
                    
                if self.card['explanation_prompt'] != new_template["explanation_prompt"]:
                    if not all([x in new_template["explanation_prompt"] for x in explanation_tags]):
                        raise Exception(f"Invalid prompt for explanation, must include [{explanation_tags}]")
                        
                if self.card['labeling_prompt'] != new_template["labeling_prompt"]:
                    if not all([x in new_template["labeling_prompt"] for x in labeling_tags]):
                        raise Exception(f"Invalid prompt for labeling, must include [{labeling_tags}]")
                        
                if self.card['confirmation_prompt'] != new_template["confirmation_prompt"]:
                    if not all([x in new_template["confirmation_prompt"] for x in confirm_tags]):
                        raise Exception(f"Invalid prompt for confirmation, must include [{confirm_tags}]")
                
                # Create the directory and the card of the template
                version_dir = str(src_dir / f"data/llms_datasets/templates/v{new_version}") 
                os.mkdir(version_dir)
                os.mkdir(version_dir+'/data')
                with open (version_dir+'/card.yaml', 'w')as obj:
                    yaml.safe_dump(new_template, obj)
                
                print(f"Create new template with version {new_version}")
    
    def overwrite_card(self):
        """
        Write the current state of the `card` dictionary to the `card.yaml` file.
        """
        with open(src_dir / f"data/llms_datasets/templates/v{str(self.version)}/card.yaml", 'w') as ob:
            yaml.safe_dump(self.card, ob)

    def update_instructions(self, command: Text, instruction: Text, overwrite: bool = True) -> bool:
        """
        Update the `instructions` set in the `card` dictionary using the provided `command` and `instruction`.

        @params:
        -------
        - command (Text): The name of the method to call on the `instructions` set.
        - instruction (Text): The argument to the method specified in `command`.
        - overwrite (bool, optional): Whether to overwrite the `card.yaml` file. Defaults to True.

        @returns:
        --------
        - bool: Whether the update was successful.
        """
        try:
            self.card['instructions'].__getattribute__(command)(instruction)
            if overwrite:
                self.overwrite_card()
            return True
        except KeyError:
            return False


    @staticmethod
    def mask_terms(sentence:Text, mask:Dict, mask_word:Text, demask=False):
        if demask:
            for v,k in dict(sorted(mask.items(),
                                    key=lambda item:item[1],
                                    reverse=True)).items():
                sentence = sentence.replace(f"{mask_word}{k}",v)
        else:
            for k,v in dict(sorted(mask.items(),
                                    key=lambda item:item[1],
                                    reverse=True)).items():
                sentence = sentence.replace(k,f"{mask_word}{v}")
        return sentence
        
    def generate_explanation_prompt(self, sentence, org_groups=None):
        prompt = self.card['explanation_prompt']
        if org_groups: 
            sentence = LLMAnnotator.mask_terms(sentence=sentence,
                                                mask=org_groups,
                                                mask_word="Company")
        # create_explanation_prompt
        explanation_prompt = str(self.card.explanation_prompt)
        # Create definitions
        definitions = ''
        output_rules = ''
        for k, v in self.card.instructions.items():
            definitions += "\n## Definitions for {}:\n".format(k)
            for defintion in v:
                definitions += '- {}\n'.format(defintion)
        for output_rule  in self.card.explanation_output_rules:
            output_rules += "- {}\n".format(output_rule)

        explanation_prompt = explanation_prompt.replace('{sentence}', sentence)    
        explanation_prompt = explanation_prompt.replace('{definitions}', definitions)
        explanation_prompt = explanation_prompt.replace('{explanation_output_rules}', output_rules)
        return explanation_prompt

    def generate_relation_prompt(self, explanation):
        prompt = self.card['labeling_prompt']
        prompt = prompt.replace('{explanation}',  explanation)
        return prompt

    def generate_confirmation(self, company1, company2, relation, explanation):
        prompt = self.card['confirmation_prompt']
        prompt = prompt.replace('{company1}',  company1)
        prompt = prompt.replace('{company2}',  company2)
        prompt = prompt.replace('{explanation}',  explanation)
        prompt = prompt.replace('{relation}',  relation)
        return prompt
        
    def annotate(self, datapoint):
        """
        Annotates a datapoint with explanations, relations, and confirmations if they do not already exist.

        @params:
        -------
        - datapoint (dict): A dictionary representing a datapoint.

        @returns:
        --------
        - dict: The annotated datapoint.
        """
        if not datapoint.get('explanation'): 
            # If the datapoint does not have an explanation, generate an explanation prompt and get the completion.
            datapoint['explanation_prompt'] = self.generate_explanation_prompt(sentence=datapoint['sentence'],
                                                                               org_groups=datapoint.get('org_groups'))
            explanation = self.get_completion(datapoint['explanation_prompt'])
            datapoint['explanation'] = LLMAnnotator.mask_terms(sentence = explanation,
                            mask=datapoint.get('org_groups'),
                            mask_word="Company", 
                            demask=True)
        if not datapoint.get('ser_relations'):
            # If the datapoint does not have a relation, generate a relation prompt and get the completion.
            datapoint['relation_prompt'] = self.generate_relation_prompt(datapoint['explanation'])
            datapoint['ser_relations'] = self.get_completion(datapoint['relation_prompt'])
            try:
                datapoint['relations'] = deserialize_relations(datapoint['ser_relations'])
                llms_relations, other_relations = establish_company_relations(datapoint, self.matcher)
                datapoint['llms_relations'] = llms_relations
                datapoint['other_relations'] = other_relations

            except:
                datapoint['relations'] = 'undefined'
                datapoint['llms_relations'] = 'undefined'
                datapoint['other_relations'] = 'undefined'

        if not datapoint.get('confirmation') and self.card.get('confirm'):
            # If the datapoint does not have a confirmation, generate a confirmation prompt and get the completion.
            if datapoint['relations'] == 'undefined':
                datapoint['confirmation_prompt'] = 'undefined'
                datapoint['confirmation'] = 'undefined'
            else:
                datapoint['confirmation_prompt'] = self.generate_confirmation(datapoint['company1'],
                                                        datapoint['company2'],
                                                        datapoint['relations'],
                                                        datapoint['explanation'])
                datapoint['confirmation'] = self.get_completion(datapoint['confirmation_prompt'])
        return datapoint
    
     

    def generate_labels(self, 
                        batch:Tuple[int,int]):
        batch_name = f'batch_{batch[0]}_{batch[1]}.json'
        file_name = src_dir / f'data/llms_datasets/templates/v{str(self.version)}/data/{batch_name}'
        if os.path.exists(file_name):
            self.logger.info("This batch is already exist")
            data = pd.read_json(file_name)
            data = pd.concat([data ,
                              self.unlabeled_data[len(data)+batch[0]:batch[1]]], axis = 0)                
        else:   
            data = self.unlabeled_data[batch[0]:batch[1]]
        annotations = []
        count  = 0
        for i, datapoint in tqdm(data.iterrows(), total= data.shape[0]):
            datapoint = datapoint.to_dict()
            datapoint['index'] = i
            if self.card.get('tagged'):
                annotations.append(self.annotate(datapoint))                 
            else: 
                # Annotate all possible pairs
                datapoint_pairs= LLMAnnotator.get_random_company_pairs(datapoint['org_groups'],
                                                                      self.card.get("max_rel", 7))
                for pair in datapoint_pairs:
                    db_pair = datapoint.copy()
                    db_pair['company1'] = pair[0]
                    db_pair['company2'] = pair[1]
                    # self.logger.info(f"For {pair}, we have datapoint: \n{datapoint}")
                    annotations.append(self.annotate(db_pair))
            count += 1
                
            if (count%10) == 0:
                pd.DataFrame(annotations).to_json(file_name)
        pd.DataFrame(annotations).to_json(file_name)
        return pd.DataFrame(annotations)

    
    def is_conflict(self, row: Dict, threshold: float = 0.85) -> Tuple[bool, Tuple[str, str, str]]:
        """
        Check if there is a conflict between two entities based on their expected relation and their actual relations.

        Args:
        - row (Dict): A dictionary containing the information about the entities and their relations.
        - threshold (float): A float value between 0 and 1 that determines the minimum similarity score for the relations to be considered aligned.

        Returns:
        - A tuple containing a boolean value indicating whether the relations are aligned or not, and a tuple of the two entities and their expected relation.
        """
        
        # Initialize the value of align to False until we prove otherwise.
        align = False
        
        # Define the two entities to search for.
        c1 = row.get('entity_1')
        c2 = row.get('entity_2')
        
        # Determine the expected relation between c2 and c1.
        expected_relation = row.get('inf_relations')
        
        # Get the organization groups and the SME relation.
        org_groups  = row.get('org_groups')
        sme_relation = (c2, expected_relation, c1)
        
        # Initialize defaultdict to carry org_ids as keys with values carrying all companies and their aliases.
        id2c = defaultdict(lambda: [])
            
        # Group the companies by their organization ID.
        for k,v in org_groups.items():
            id2c[v].append(k)
        
        # Set the expected relation to "other" if the label is 0.
        if row.get('Label') == 0:
            expected_relation = "other"
        
        # Set the SME relation based on the main relations.
        elif main_relations[0] == expected_relation:
            sme_relation = (c2,main_relations[0], c1)
        elif main_relations[1] == expected_relation:
            sme_relation = (c1,main_relations[0], c2)
            
        # Initialize defaultdict to carry supplier names as keys with values carrying customer names.
        llm_relations= defaultdict(lambda : [])
        
        # Get all the supplier-customer relations.
        if isinstance(row['llms_relations'], list):
            for llm_relation in row['llms_relations']:
                if llm_relation[1] == 'supplier':
                    supplier = llm_relation[0]
                    supplier_id = org_groups.get(supplier)
                    supplier_names = id2c[supplier_id] if supplier_id else [supplier]
                    
                    customer = llm_relation[2]
                    customer_id = org_groups.get(customer)
                    customer_names = id2c[customer_id] if customer_id else [customer]
                    
                    for supplier_name in supplier_names:
                        llm_relations[supplier_name] += customer_names
        
        # Get the supplier names.
        llm_suppliers = list(llm_relations.keys())
        
        # Check if the relation had been detected
        expected_supplier = sme_relation[0]
        expected_customer = sme_relation[2]
        
        # If the expected relation is "other".
        if expected_relation == "other":
            # If there is no relation between the reporter and other companies.
            if len(llm_relations) == 0:
                align = True
            # If there are relations between the reporter and other companies.
            else:
                # Check if the supplier is found in the llm_relations.
                align = not self.matcher.similarity(expected_supplier,list(llm_relations.keys())).max() > threshold
        
        # If the expected relation is not "other".
        else:
            # If there are relations between the reporter and other companies.
            if len(llm_relations) > 0:
                # Get the similarity scores between the expected supplier and the llm_suppliers.
                sim_scores = self.matcher.similarity(expected_supplier, llm_suppliers)
                max_score = sim_scores.max()
                max_idx  = sim_scores.argmax()
                
                # If the maximum similarity score is greater than the threshold or the expected supplier is found in the llm_suppliers.
                if max_score > threshold  or any([expected_supplier in x for x in llm_suppliers  ]):
                    # Check if the expected customer is found in the llm_relations.
                    align = self.matcher.similarity(expected_customer,llm_relations[llm_suppliers[max_idx]] ).max() > threshold \
                            or any([[expected_customer.lower() in x.lower() for x in y] for y in llm_relations[llm_suppliers[max_idx]]  ])
        
        # Return the align and sme_relation values.
        return align, sme_relation
    
    def detect_conflicts(self, threshold:float=0.85,save:bool = True)->pd.DataFrame:
        '''Search conflicts within llms annotator compared with ground truth labels
        '''
        # Read generated labels
        llm_labels = pd.read_json(f"data/llms_datasets/templates/v{self.version}/labels/labels.json")
        llm_labels = llm_labels.query("inf_relations.notnull()")

        # Determine the basic two lists: sme_relations (expert annotations) and align_bool (True if llms align with ground truth).
        sme_relations = []
        align_bool = []
        for i, row in tqdm(llm_labels.iterrows(), total=llm_labels.shape[0]):
            align, sme_relation = self.is_conflict(row.to_dict(), threshold=threshold)
            align_bool.append(align)
            sme_relations.append(sme_relation)

        llm_labels['sme_relations'] = sme_relations
        llm_labels['align'] = align_bool

        true_ratio = llm_labels.query("align == True").shape[0] / llm_labels.shape[0]
        self.logger.info("Alignment percentage: {:.2f}%".format(true_ratio*100))
        conflicts = llm_labels.query("align == False")

        if save:
            if not os.path.exists(f"data/llms_datasets/templates/v{self.version}/reports"):
                os.mkdir(f"data/llms_datasets/templates/v{self.version}/reports")
            conflicts.to_json(self._conflict_path, index=True)
        return conflicts

    @staticmethod
    def get_companies_and_relation(relation: dict) -> tuple:
        """
        Extracts the companies and relation from a dictionary.

        Parameters:
        relation (dict): A dictionary containing the relation between two companies.

        Returns:
        tuple: A tuple containing the first company, the relation, and the second company in the relation.

        Example:
        >>> relation = {'company_1': 'Health Net Inc.', 'relation': 'supplier', 'company_2': 'LA Care'}
        >>> get_companies_and_relation(relation)
        ('Health Net Inc.', 'supplier', 'LA Care')
        """
        keys = np.array(list(relation.keys()))

        # Get the index of the 'relation' key in the dictionary
        relation_idx = np.where(keys == 'relation')[0][0]

        # Get the first and second company names based on the 'relation' key index
        company_1 = relation[keys[relation_idx-1] if relation_idx > 0 else keys[1]]
        company_2 = relation[keys[relation_idx+1] if relation_idx<len(keys) else keys[0]]

        return company_1, relation['relation'], company_2
    @staticmethod
    def get_random_company_pairs(org_groups, max_relation=5):
        """
        Returns a list of randomly-selected pairs of companies from a dictionary of company groups.

        Parameters:
            org_groups (dict): A dictionary mapping company keys to group values.
            max_relation (int): The maximum number of pairs to return.

        Returns:
            A list of randomly-selected pairs of companies.
        """
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

        # For each of the 5 chosen combinations, randomly choose one key from each group of 'org_groups'
        #  that corresponds to the two values in the combination
        n_relations = max_relation if len(availabel_relations) > max_relation else len(availabel_relations)
        random_pairs = random.sample(availabel_relations, n_relations)
        company_pairs = []
        for pair in random_pairs:
            company1 = random.choice(ids2org[pair[0]])
            company2 = random.choice(ids2org[pair[1]])
            company_pairs.append((company1, company2))
        return company_pairs

    
    
    
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
            list_of_dicts.append(json.loads(re.sub(r"(?<!\w)'|'(?!\w)", '"', dict_string.replace('"', '\\"'))))
    return list_of_dicts

def establish_company_relations(datapoint, matcher):
    """
    Assigns relationships between companies based on certain criteria.

    Args:
        datapoint (dict): A dictionary containing the sentence to be processed.
        matcher (object): An instance of the fuzzywuzzy string matching class.

    Returns:
        tuple: A tuple containing the LLMS relations and other relations.

    """
    global main_relations
    global inverse
    org_groups = datapoint['org_groups']
    relations = datapoint['relations']
    matcher_built = False
    # Collect all companies mentioned in the relations and create a dictionary with each unique company as a key
    llms_companies = []
    for relation in relations:
        llms_companies += [relation.get('company_1'), relation.get('company_2')]
    llms_companies = {k:None for k in set(llms_companies)}
    llms_ids = {k:i for i,k in enumerate(set(llms_companies))}
    ids_llms = {i:k for i,k in enumerate(set(llms_companies))}
    # Check if each company in the dictionary is mentioned in the sentence, and if not, try to match it with a known organization
    for company in list(llms_companies.keys()):
        if company in datapoint['sentence']:
            llms_companies[company] = company
        else:
            if matcher_built is False: 
                matcher.build_index(list(org_groups.keys()))
                matcher_built = True
            
            matches = matcher.search(company, threshold=0.95, top_k = 3)
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
        c1 = relation.get('company_1')
        c2 = relation.get('company_2')
        if not all([c1 in llms_companies.keys() , c2 in llms_companies.keys()]):
            continue
        relation = relation.get('relation')
        if main_relations[0] == relation:
            llms_relations.append((c1,main_relations[0], c2))
        elif main_relations[1] == relation:
            llms_relations.append((c2,main_relations[0], c1))
        else:
            llms_relations.append((c1, relation, c2))
        
        if not all([c1,c2,relation]):
            continue 
        c1_id = llms_ids.get(c1)
        c2_id = llms_ids.get(c2)    
        exist_relations.append(tuple(sorted([c1_id, c2_id])))
        
    other_ids = set(availabel_relations) ^ set(exist_relations)
    other_relations = []
    for pair in other_ids: 
        c1 = llms_companies[ids_llms[pair[0]]]
        c2 = llms_companies[ids_llms[pair[1]]]    
        other_relations.append((c1,'other', c2))
    return llms_relations, other_relations