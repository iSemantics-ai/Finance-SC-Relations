import os
import pickle
import re
import pandas as pd
import json
from sklearn.metrics import classification_report,confusion_matrix
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import permutations
from pathlib import Path
import logging

logging.basicConfig(
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)
logger = logging.getLogger("__file__")
SPECIAL_CHAR = "[&()<>*#/\\]"
inverse_dict = {"supplier": "customer", "customer": "supplier", "other": "other"}





def extract_tagged_names(text):
    """
    Extracts entities between tags in a given text and removes the tags.

    @params
    -------
    text (str): The text containing the tagged entities.

    @returns
    --------
    tuple: A tuple containing the original text with the tags removed,
    and the extracted entities as strings.
    """

    # Extract entities between tags
    c1 = text[text.find("[E1]") + len("[E1]") :text.find("[/E1]")]
    c2 = text[text.find("[E2]") + len("[E2]") :text.find("[/E2]")]

    # Remove tags
    tags_to_remove = ["\\[E1\\]", "\\[/E1\\]", "\\[E2\\]", "\\[/E2\\]"]
    regex_pattern = '|'.join(tags_to_remove)
    org_text = re.sub(regex_pattern, '', text)
    org_text = re.sub(r'\\s+', ' ', org_text)

    return {"orig_sent":org_text, 'entity_1':c1.strip() , "entity_2":c2.strip()}


def evaluation_report(inferer,
                      tagged_data,
                      tag_name,
                      report_dir,
                      reverse=False,
                      mutate=True,
                      save_reports=True):
    """
    Evaluates a trained neural network and generates a report on the model's performance.

    @params
    -------
    inferer: A trained instance of the `infer_from_trained` class.
    tagged_data: A Pandas DataFrame containing the tagged data to evaluate.
    tag_name: A string representing the name of the tag to evaluate.
    report_dir: A string representing the directory to save the evaluation report.
    mutate: A boolean value indicating whether to mutate the input data during evaluation. Default is True.
    """
     # estimate softmax scores    
    tagged_data = inferer.predict_fn(tagged_data, reverse=reverse, mutate=mutate)
    
    if 'r_id' in tagged_data.columns:
        # aggregate multi-positioning relations.
        id_scores = tagged_data.groupby(['r_id'])\
         .apply(lambda x : list(np.mean(x['scores'].tolist(), axis=0))).to_dict()
        # assign aggregated relation for all positions
        tagged_data['scores'] = tagged_data['r_id'].apply(lambda x: id_scores[x])
        # drop duplicates from multi-positions
        tagged_data.drop_duplicates(subset=['r_id'], inplace=True, ignore_index=True)
    # define max scores and its label_ids
    score, labels = torch.tensor(tagged_data['scores']).max(1)
    # create relations info items to compine relations on each sentence
    tagged_data.loc[:, 'prediction_id'],\
    tagged_data.loc[:, 'score'] =  labels, score
    tagged_data.loc[:, 'prediction'] = tagged_data['prediction_id']\
    .apply(lambda x : inferer.id2label[str(x)])

    # Map true labels with label2id
    true_labels = list(map(lambda x: inferer.label2id[x], tagged_data['relations'].tolist()))


    # Create a mask to identify errors
    errors_mask = list(map(lambda x, y: x != y, labels.tolist(), true_labels))
    
    # Extract entities and remove tags
    if 'orig_sents' not in tagged_data.columns:
        tagged_data = pd.concat([tagged_data,
                                 pd.DataFrame(list(map(lambda x: extract_tagged_names(x),
                                                       tagged_data['sents'])))]
                                , axis=1)
    tagged_data = tagged_data\
        .rename(columns={"orig_sents": "sentence", "relations": "expected_relation"})
    
    # Generate a DataFrame of misclassified data
    miss_classified = tagged_data[errors_mask]\
        .sort_values(by='score', ascending=False)
    


    
    ######################################Threshold performance######################################
    thresholds=[0.90, 0.95, 0.99]
    cr = classification_report(tagged_data['expected_relation'].tolist(), tagged_data['prediction'].tolist(),
                               target_names=list(inferer.label2id.keys()),
                               digits=4, output_dict=True)
    print(f'The classification report:\n{pd.DataFrame(cr).T}\n')
    
    for threshold in thresholds:
        filtered = tagged_data[tagged_data['score'] > threshold]
        # print threshold status
        print(f'''At threshold `{threshold}` dropped: datapoint={tagged_data.shape[0]- filtered.shape[0]}
        frac={str(1 -(filtered.shape[0]/tagged_data.shape[0]))}''')
        # calculate the merics required
        y_true = filtered['expected_relation'].tolist()
        y_pred = filtered['prediction'].tolist()
        f_cr = classification_report(y_true,
                                   y_pred,
                                   target_names=list(inferer.label2id.keys()),
                                   digits=4, output_dict=True)
        print(f"Classification report for score above threshold:\n{pd.DataFrame(f_cr).T}\n\n")
        
    ######################################Threshold performance######################################

    # Compute confusion matrix
    cm = confusion_matrix(true_labels, labels, normalize='true')

    # Print metrics and save report
    print(f"{tag_name} metrics\n{cr}")
    cr = pd.DataFrame(cr)
    metrics = {f"{tag_name}_{k}": round(v, 3) for k, v in cr.T.loc['weighted avg'].to_dict().items()}
    metrics[f'{tag_name}_accuracy'] = round(cr.T.loc['accuracy']['f1-score'], 3)

    plt.figure(figsize=(10, 7))
    fig = sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=list(inferer.label2id.keys()),
                       yticklabels=list(inferer.label2id.keys()))
    plt.show()

    if save_reports:
        print(f"saving {tag_name} report")
        plt.savefig(f"{report_dir}/{tag_name}_confusion.png")
        with open(f"{report_dir}/{tag_name}_metrics.json", 'w') as obj:
            json.dump(metrics, obj)
        cr.to_markdown(f"{report_dir}/{tag_name}_classification_report.md")
        miss_classified.to_excel(f"{report_dir}/{tag_name}_errors.xlsx", index_label="index")

    return miss_classified, tagged_data

def create_org_groups(span):
    return {y:x for x,y in enumerate(set(\
            filter(None, [x.get('text') \
            if x.get("label") == "ORG" \
            else None for x in span]) ))}


def load_pickle(filename):
    with open(filename, "rb") as pkl_file:
        data = pickle.load(pkl_file)
    return data


def save_as_pickle(filename, data):
    with open(filename, "wb") as output:
        pickle.dump(data, output)


def get_subject_objects(sent_):
    ### get subject, object entities by dependency tree parsing
    # sent_ = next(sents_doc.sents)
    root = sent_.root
    subject = None
    objs = []
    pairs = []
    for child in root.children:
        # print(child.dep_)
        if child.dep_ in ["nsubj", "nsubjpass"]:
            if (
                len(re.findall("[a-z]+", child.text.lower())) > 0
            ):  # filter out all numbers/symbols
                subject = child
                # print('Subject: ', child)
        elif child.dep_ in ["dobj", "attr", "prep", "ccomp"]:
            objs.append(child)
            # print('Object ', child)
    if (subject is not None) and (len(objs) > 0):
        for a, b in permutations([subject] + [obj for obj in objs], 2):
            a_ = [w for w in a.subtree]
            b_ = [w for w in b.subtree]
            pairs.append(
                (a_[0] if (len(a_) == 1) else a_, b_[0] if (len(b_) == 1) else b_)
            )

    return pairs
