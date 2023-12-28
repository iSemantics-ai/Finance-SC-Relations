import argparse
import json
import pandas as pd
import numpy as np
from typing import Text
import yaml
from colorama import Fore
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
src_dir = Path.cwd()
sys.path.append(str(src_dir)) 
from src.utils.logs import get_logger
from src.utils.data import dict2dot
from sklearn.metrics import classification_report, confusion_matrix
from src.relation_extraction.infer import infer_from_trained
from src.relation_extraction.preprocessing_funcs import (inverse_relations,
                                                        inverse_dict)
from src.utils.preprocess import create_re_data, get_source, entity_annotation
from src.relation_extraction.misc import evaluation_report, create_org_groups
from src.language_model.spacy_loader import SpacyLoader



def evaluate(config_path: Text) -> None: 
    """
    Load and validate data
    """ 

    """Load raw data.
    Args:
        config_path {Text}: path to config
    """

    with open(src_dir/ config_path) as conf_file:
        config = dict2dot(yaml.safe_load(conf_file))

    logger = get_logger('Evaluate', log_level=config.base['log_level'])
    ## Modify params with what fits evaluation 
    config['train']['inverse']= False 
    config['train']['replace_ent']= False 
    ## Read relations file
    test_data = pd.read_json(src_dir / config.evaluate['dataset'])
    test_data = test_data.dropna(subset=[config.evaluate.text_col])
    ## Check missed columns 
    missed_columns = set([config.evaluate.text_col,
      config.evaluate.entity_1,
      config.evaluate.entity_2,
      config.evaluate.target]) - set(test_data.columns)

    # define the dataset metadata used for the training
    dataset_name = Path(config.load_data.dataset).name.split('.')[0]
    # read yaml config associated to the dataset
    with open(src_dir /'data/config/{}.yaml'.format(dataset_name), 'r')as o:
        data_files = yaml.safe_load(o)
        data_sources = {x:Path(y['dir']).name for x,y in data_files.items()}
    # load the ids mapping dict
    ids_map = np.load(src_dir / "data/train/row_ids.npy", allow_pickle=True)


    if len(missed_columns) > 0: 
        raise ValueError(f"Required columns missed <{','.join(x for x in missed_columns)}>")
    # Save sentence id for later use
    if config.base.index_col not in test_data.columns: 
        test_data[config.base.index_col] = list(range(len(test_data)))
    ## If spans pre-detected don't load entity extractor        
    test_data = test_data.drop_duplicates(subset = [config.evaluate.text_col,
                               config.evaluate.entity_1, 
                               config.evaluate.entity_2,]).reset_index(drop=True)
    ## Clean the test_data using preprocess pipe
    test_data.rename(columns={config.evaluate.text_col:'orig_sent',
                     config.evaluate.entity_1:'entity_1',
                     config.evaluate.entity_2: 'entity_2', 
                     config.evaluate.target: 'relations',
                     config.evaluate.spans: 'spans'}, inplace=True)

    test_data["org_groups"] = test_data.spans.apply(lambda x : create_org_groups(x))
    # Setup the basic targets
    test_data['relations'] = test_data['relations'].apply\
    (lambda x : x if x in config.base.main_relations else 'other')
    # Check spans
    extract_ent = False if 'spans' in test_data.columns else True
    if not extract_ent:
        from src.language_model.spacy_loader import Docs_Container, SpacyLoader
        # 
        docs_container = Docs_Container()
        docs_container._docs,\
        docs_container._spans,\
        docs_container._ents = (test_data['orig_sent'].tolist(),
                                test_data['spans'].tolist(),
                                test_data['org_groups'].apply(lambda x : set(x.keys())).to_list())

        spacy_loader = SpacyLoader(lm=None,
                                   require_gpu = True,
                                   entity_matcher= "sentence-transformers/all-MiniLM-L6-v2",
                                   load_matcher=True)
        group_docs, aliases_docs = spacy_loader.group_ents(docs_container)
        test_data.loc[:, 'org_groups']= group_docs

    # Initiate the inferer
    inferer = infer_from_trained(detect_entities=extract_ent,
                                 language_model="en_core_web_trf",
                                 require_gpu=True,
                                 basic_targets= ['supplier','customer'],
                                 load_matcher=True
                                )

    if extract_ent: 
        sents, spans, groups, aliases = inferer.spacy_loader.predictor(test_data['orig_sent'].tolist())
        test_data.loc[:, 'sents'] = sents
        test_data.loc[:, 'spans'] = spans
        test_data.loc[:, 'org_groups'] = groups
        test_data.loc[:, 'aliases'] = aliases

    # Load fine-tune model
    logger.info("Loading the pretrained model")
    inferer.load_model({'model_path': str(src_dir / config.train.model_path),
                       'batch_size': config.train.batch_size})
    tagged = create_re_data(test_data,
                   'orig_sent',
                   'entity_2',
                   'entity_1',
                   'relations',
                   inverse_dict,
                   static_position=config.base['entity_static_position'],
                  ).dropna(axis='columns')
    if len(missed_columns) > 0: 
        raise ValueError(f"Required columns missed <{','.join(x for x in missed_columns)}>")

    # Inverse the entity_tags
    inverse_data = tagged.copy()
    inverse_data.sents = inverse_data.sents.apply(inverse_relations)
    assert not all(inverse_data.sents == tagged.sents)
    inverse_data.relations = inverse_data.relations.apply(
        lambda x: inverse_dict[x]
    )

    # Test metricsa and errors reporting 
    test_errors, test_output = evaluation_report(inferer=inferer,
                      tagged_data=tagged,
                      tag_name='test',
                      report_dir= str(src_dir/ 'metrics'),
                      mutate= config.evaluate['mutate'],
                      reverse= config.evaluate['reverse'],
                      save_reports=True)
    # Inverse tags errors
    test_inv_errors, test_inv_output = evaluation_report(inferer=inferer,
                      tagged_data=inverse_data,
                      tag_name='test',
                      report_dir= str(src_dir/ 'metrics'),
                      mutate= config.evaluate['mutate'],
                      reverse= config.evaluate['reverse'],
                      save_reports=False)
    test_output["inverse_prediction"] = test_inv_output.prediction.apply(lambda x : inverse_dict[x])
    test_output['inverse_score'] = test_inv_output['score']
    direction_conflict = test_output.query("prediction != inverse_prediction")
    direction_conflict.drop(columns=['spans', 'prediction_id'])\
                      .to_excel(src_dir / "metrics/test_inconsistent_directions.xlsx")



    # create evaluation report for the dev-set
    dev_data = pd.read_json(src_dir / config['train']['valid_data'])

    if config.evaluate.return_source:
        dev_data.loc[:, 'source_data'] = dev_data[config.base.index_col]\
        .apply(lambda x : get_source(ids_map,data_sources,x))
    # Inverse the entity_tags
    dev_inverse = dev_data.copy()
    dev_inverse.sents = dev_inverse.sents.apply(inverse_relations)
    assert not all(dev_inverse.sents == dev_data.sents)
    dev_inverse.relations = dev_inverse.relations.apply(
        lambda x: inverse_dict[x]
    )
    assert not all(dev_inverse.relations == dev_data.relations)

    # Dev metrics and errors reports 
    dev_errors, dev_output = evaluation_report(inferer=inferer,
                      tagged_data=dev_data.drop(columns=['r_id']),
                      tag_name='dev',
                      report_dir= str(src_dir/ 'metrics'),
                      mutate= config.evaluate['mutate'],
                      reverse= config.evaluate['reverse'],
                      save_reports=True)
    # Inverse tags errors
    dev_inv_errors, dev_inv_output = evaluation_report(inferer=inferer,
                      tagged_data=dev_inverse.drop(columns=['r_id']),
                      tag_name='dev',
                      report_dir= str(src_dir/ 'metrics'),
                      mutate= config.evaluate['mutate'],
                       reverse= config.evaluate['reverse'],
                       save_reports=False)
    
    dev_output["inverse_prediction"] = dev_inv_output.prediction.apply(lambda x : inverse_dict[x])
    dev_output['inverse_score'] = dev_inv_output['score']
    direction_conflict = dev_output.query("prediction != inverse_prediction")
    direction_conflict.drop(columns=['spans', 'prediction_id'])\
                      .to_excel(src_dir / "metrics/dev_inconsistent_directions.xlsx")
if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', default='params.yaml')
    args_parser.add_argument('--mutate',  default=False)
    
    args = args_parser.parse_args()
    evaluate(config_path=args.config)