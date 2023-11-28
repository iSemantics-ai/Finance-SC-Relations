"""This script meant to use LLM relations to create ready for train dataset

"""

import os
import sys
import argparse
import pandas as pd
import openai
import pandas as pd
from glob import glob
from tqdm import tqdm
import json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KE")

current_path = Path.cwd()
src_dir = current_path
print("src_dir", src_dir)
sys.path.append(str(src_dir))
# import annotation methods
from src.labels_generator import (generate_relations,relation_search,resort_relation, get_completion,
                                  deserialize_relations,
                                  generate_relations_with_explanation,
                                  relations_tupled,
                                 create_sorted_relation)
from src.labels_generator.utils import create_re_dataset
# Load matcher
from src.matcher.core import SimCSE_Matcher
matcher = SimCSE_Matcher(str(src_dir/ 'artifacts/matcher_model'))

replaces = {"sentence": "{sentence}"}
# Replace the keys with values for unified relation direction
relations_map = {"customer": "supplier"}





if __name__ == '__main__':
    # Create an argument parser
    parser = argparse.ArgumentParser(description='GPT Annotation Job arguments')
    parser.add_argument('--version', type=str, help='Prompt version')
    # Define the arguments you want to accept
    # Parse the command-line arguments
    args = parser.parse_args()
    version = args.version
    
    
#     files =glob(f"data/tasks/finetune_llm_on_label_1/llm_relations_with_explained_v{version}_*.xlsx")
#     output  = pd.concat([pd.read_excel(file, index_col="index") for file in files], axis=0)
    
    output = pd.read_excel("data/raw/llm_relations_v2_3.xlsx", index_col="index")
    
    print("output.shape",output.shape)
    print("output.columns",output.columns)
    def eval_relations(relation):
        try:
            re = eval(relation)
        except:
            re = []
        return re
    # Resort the sme_relations to unify the relations directions
    if not isinstance(output['sme_relations'].iloc[0], list):
        tqdm.pandas(desc="Eval sme_relations")
        output['sme_relations'] = output['sme_relations'].progress_apply(eval)
    if not isinstance(output['relations'].iloc[0], list):
        tqdm.pandas(desc="Eval relations")
        output['relations'] = output['relations'].progress_apply(eval_relations)
    if not isinstance(output['org_groups'].iloc[0], list):
        tqdm.pandas(desc="Eval org_groups")
        output['org_groups'] = output['org_groups'].progress_apply(eval_relations)  
    tqdm.pandas(desc="Resort sme relations")
    output['sme_relations'] = output['sme_relations'].progress_apply(lambda x:\
                              resort_relation((x[0], x[1], x[2]),
                                            relations_map))
    if "align" not in output.columns:
        # Search relations and return mask
        tqdm.pandas(desc="Search relations")
        output['align'] =\
        output[['sme_relations', 'relations']]\
        .progress_apply(lambda x:
        relation_search(
        query_relation=x[0],
        relations_tuples=x[1],
        matcher=matcher,
        threshold=0.8,
        main_relations=list(relations_map.values()) ),axis=1).to_list()

    print("Errors count, Errors rate",output[output['align'] == False].shape[0],\
        output[output['align'] == False].shape[0] / len(output))
    dataset = create_re_dataset(output,
                            matcher,
                            feature_key="sentence",
                            relations_key='relations',
                            threshold=0.9)
    print("dataset shape", dataset.shape)
    dataset.to_json(f"data/raw/llm_v{version}_align.json", index='idx')
    print(dataset.relation.value_counts())