import argparse
# Read the data
from glob import glob
import pandas as pd
import sys 
import os
from pathlib import Path
import sys 
src_dir = Path.cwd()
sys.path.append(str(src_dir))

if __name__ == '__main__':
    # Create an argument parser
    parser = argparse.ArgumentParser(description='GPT Annotation Job arguments')

    # Define the arguments you want to accept
    parser.add_argument('--version', type=float, help='Which template version to be used for annotation')
    # parser.add_argument('--version', type=, help='Which template version to be used for annotation')

    parser.add_argument('--device', type=str, help='The device where the matcher model will be allocated', default='cpu')

    # Parse the command-line arguments
    args = parser.parse_args()
    # Access the values of the arguments
    version = args.version
    data = pd.concat([pd.read_json(x) for x in glob(f'data/llms_datasets/templates/v{version}/data/*.json')], axis= 0)
    data.reset_index(drop=True, inplace=True)
    run_supply = False
    run_spans = False
    if 'Label' in data.columns:
        if data.Label.isna().sum() > 0:
            run_supply = True
    else:
        run_supply = True
            
            
    if run_supply:
        from src.supply_chain_classifier.text_classification.classifier import TextClassifier
        model = TextClassifier(
                           model_name='supply_chain_classifier',
                           num_workers=0,
                           batch_size=32,
                           max_length=264)
        model.load_model('artifacts/supply_chain_classifier')
        labels = model.predict(data, feature="sentence")
        data['Label'] = labels[1]
    

    run_spans = False if 'org_groups' in data.columns.tolist() and data.org_groups.isna().sum() == 0 else True
    if run_spans:
        from src.language_model.spacy_loader import SpacyLoader
        spacy_model = SpacyLoader(lm='en_core_web_trf',
                             load_matcher=True)
        spans, org_groups, aliases = spacy_model.predictor(data.sentence.tolist())
        data['org_groups'] = org_groups
    if not os.path.isfile(f'data/llms_datasets/templates/v{version}/labels'):
        os.mkdir(f'data/llms_datasets/templates/v{version}/labels')
    data.to_json(f'data/llms_datasets/templates/v{version}/labels/labels.json')
