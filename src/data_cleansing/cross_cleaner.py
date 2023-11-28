"""
This module using slice-trained models to indicate bad labels with mean score aggregation and comflict score
"""
from typing import Dict, Optional, Callable
import yaml
from colorama import Fore
import os
from random import sample
import numpy as np
import torch
import pandas as pd
from glob import glob
import json
import torch
import gc
import nltk
from nltk.corpus import words
nltk.download('words')
import sys
from pathlib import Path
src_dir = Path(os.path.dirname(os.path.realpath(__file__)))\
.parent.parent
print(src_dir)
sys.path.append(str(src_dir))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from src.relation_extraction.infer import infer_from_trained
from src.relation_extraction.preprocessing_funcs import preprocess_custom_data
from src.utils.preprocess import create_re_data
from src.relation_extraction.trainer import train_and_fit
from src.utils import get_logger, dict2dot
from src.relation_extraction.misc import load_pickle
from src.relation_extraction.infer import infer_from_trained
from src.relation_extraction.preprocessing_funcs import inverse_relations, inverse_dict
from src.utils.preprocess import create_re_data
from .utils import process_weaklabels, map_relation, rel_from_text
from src.relation_extraction.misc import evaluation_report, create_org_groups
from cleanlab import Datalab



supplier_concepts= ['manufacturer', 'provider' ,'supplies', 'Supplier', 'supplier', 'provides', 'provide']

other_concepts = ['subsidiary', 'creditor','investor',
               'guarantor','lessor','sells products through','issuer',
               'lessor, manager, or operator','parent','', ' ','partner',
                'intermediary', 'primary contract manufacturer','licensee','nothing','competitor','unknown','licensor']

customer_concepts = ["Customer", 'customer']

concepts = {"supplier": supplier_concepts,
       "customer": customer_concepts,
       "other": other_concepts}


class CrossCleaner:
    def __init__(self,task_name=None, parameters_dir= 'params.yaml'):

        with open(src_dir / parameters_dir) as obj:
            self.params = dict2dot(yaml.safe_load(obj))
        self.logger = get_logger('\U0001F4DD CrossCleaner', log_level=self.params['base']['log_level'])
        self._task_name = task_name
        self._models_dir = src_dir / f'data/folds/{self._task_name}'
        # Save each fold data as train, test files
        if self._task_name:
            if not os.path.isdir(self._models_dir):
                os.makedirs(self._models_dir, exist_ok=True)
        else:
            self._task_name = '_'.join(sample(words.words(), 2))
            self._models_dir = src_dir / f'data/folds/{self._task_name}'
            self.logger.info(f"Creating new task dir with name " + Fore.GREEN + self._task_name + Fore.RESET)
            os.makedirs(self._models_dir, exist_ok=True)
            

        
    @property
    def folds(self):
        folds = glob(str(self._models_dir / 'fold') + "*")
        if len(folds) < 2:
            self.logger.info(
                f'There is less than 2 folds on the current work dir : {self._task_name}'\
                +'\n Please use CrossCleaner.cross_split to create new folds')
        return folds

    def cross_split(self, 
                    data_path: str, 
                    n_folds: int,
                    stratify_by: str,
                    index_col:str,
                    feature_column: str,
                    ent1: str,
                    ent2: str,
                    label: str,
                    additional_func: Optional[Callable] = None) -> str:
        """
        Cross-splits the data into train and test sets, and preprocesses the data for training.

        @params
        -------
        data_path (str): The path to the data file to use for splitting and preprocessing.
        n_folds (int): The number of folds to use for cross-validation.
        stratify_by (str): The name of the column to use for stratification during cross-validation.
        index_col (str): The name of index column each index represent unique sentence.
        feature_column (str): The name of the column containing the text features to use for training.
        ent1 (str): The name of the first entity column.
        ent2 (str): The name of the second entity column.
        label (str): The name of the label column.
        additional_func (Optional[Callable]): An optional function to apply to the data before preprocessing.

        @returns
        --------
        str: The path to the directory containing the preprocessed data.
        """
        # Load data
        data = pd.read_json(src_dir / data_path)
        # Split data into folds based on stratify_by
        unique_ids = data[index_col].unique()
        ids_labels = data.groupby(index_col).apply(lambda x: x[stratify_by].iloc[0]).to_dict()
        data_ref = pd.DataFrame({index_col:ids_labels.keys(), stratify_by:ids_labels.values()} )

        # Split data into folds based on stratify_by
        splits = [data_ref[data_ref[stratify_by] == x] for x in data_ref[stratify_by].unique()]
        weights = [data_ref[data_ref[stratify_by] == x].shape[0] / data_ref.shape[0] for x in data_ref[stratify_by].unique()]
        fold_num = int(data_ref.shape[0] / n_folds)
        split_weights = [int(fold_num * x) for x in weights]
        data_folds = []


        info = {"data_path": data_path,
                "data_size": data.shape[0],
                "n_folds":n_folds}        # Create data folds
        for i in range(n_folds):
            fold = []
            for j, s in enumerate(splits):
                if i == n_folds-1:
                    fold.append(s[split_weights[j] * i:])
                else:  
                    fold.append(s[split_weights[j] * i:split_weights[j] * (i+1)])
            fold = pd.concat(fold, axis=0).reset_index(drop=True)
            self.logger.info(f'at fold {i} with shape is : {fold.shape}')
            data_folds.append(fold)

        train_folds = []
        valid_folds = []

        # Split data into training and validation sets for each fold
        for i in range(len(data_folds)):
            data_copy = data_folds.copy()
            valid_folds.append(data_copy.pop(i))
            train_folds.append(data_copy)
        fold_num = 0

        for train_sets, valid_set in zip(train_folds, valid_folds):
            fold_path = src_dir / f'{self._models_dir}/fold{fold_num}'
            if not os.path.exists(fold_path):
                os.makedirs(fold_path)
            train_data = pd.concat(train_sets, axis=0).reset_index(drop=True)
            train_data = data[data[index_col].isin(train_data[index_col])]

            valid_data = data[data[index_col].isin(valid_set[index_col])]

            # Apply optional additional function to the data
            if additional_func is not None:
                self.logger.info(f"applying {additional_func.__name__}...")
                train_data = additional_func(train_data)
                valid_set = additional_func(valid_data)

            # Create tagged sentences to be tokenized
            train_data = create_re_data(train_data,
                                   feature_column,
                                   ent2,
                                   ent1,
                                   label,
                                   inverse_dict,
                                  static_position=self.params.base.entity_static_position,
                                  num_positions=self.params.data_split.num_positions,)
            valid_data = create_re_data(valid_data,
                                   feature_column,
                                   ent2,
                                   ent1,
                                   label,
                                   inverse_dict,
                                  static_position=self.params.base.entity_static_position,
                                  num_positions=0)

            # Save train and test data to JSON files
            train_data.reset_index(drop=True).to_json(f'{fold_path}/train.json')
            valid_data.reset_index(drop=True).to_json(f'{fold_path}/test.json')
            dataset_args =  {'files':[src_dir / f'{fold_path}/train.json', src_dir/ f'{fold_path}/test.json'],
                            'output_dir':str(src_dir / f'{fold_path}') + '/',
                            'relations_mapper':src_dir / 'data/train/relations.pkl',
                            'inverse': self.params.train_preprocess['inverse'], 
                            'stage':'train'}

            args = self.params.copy()
            args['train_preprocess'] = dataset_args
            # Preprocess the data for training
            train, valid, rm = preprocess_custom_data(dict2dot(args))
            fold_num += 1
            # save data_info 
            with open(os.path.join(self._models_dir, "config.json"), 'w') as fb:
                json.dump(info, fb)
    def train_models(self) -> None:
        """
        Train and fit models on each fold in a directory and generate cross scores.

        @params
        -------
        folds_dir (str): A string containing the directory path to the folds.
        training_parameters (dict): A dictionary containing the parameters for training and fitting the models.

        Returns:
            None
        """
        training_parameters = self.params.train.copy()
        training_parameters['relations_mapper'] = src_dir / self.params.train.relations_mapper
        training_parameters['src_dir'] = src_dir 
        # Loop over each fold and train the model
        for fold in self.folds:
            if os.path.isfile(os.path.join(fold, 're_model/model.pth.tar')):
                confirm = input(f"This Fold <{fold}> have been trained before are you sure you want to continue?!(yes|no|break)")
                if confirm == 'no':
                    continue
                elif confirm == 'yes':
                    self.logger.info("Retraining starts...")
                else:
                    self.logger.info("Training stopped!!!")
                    break
            print(Fore.GREEN, "="*60, '\n', f"Training Fold With Dir <{fold}>", "\n","="*60,'\n', Fore.RESET)
            # Update the training parameters with fold-specific data
            training_parameters['train_data'] = os.path.join(fold, "df_train.json")
            training_parameters['valid_data'] = os.path.join(fold, "df_test.json")
            training_parameters['model_path'] = os.path.join(fold, "re_model")
            # Train and fit the model
            train_and_fit(training_parameters)
            gc.collect()
            torch.cuda.empty_cache()
    def cross_evaluation(self):
        """
        Perform cross-validation on a relation extraction model using the specified configuration.

        @params
        -------
        - folds_dir (Text): The directory containing the folds to use for cross-validation.
        - config (dict): The configuration to use for the evaluation.

        @returns
        -------
        None
        """
        if len(self.folds) < 2:
            raise Exception("No folds trained yet")
        missed_models = []
        for fold in self.folds:
            if not os.path.isfile(os.path.join(fold, "re_model/model.pth.tar")):
                missed_models.append(fold)
        if len(missed_models) > 0:
            raise Exception(f"Missing models in {missed_models}")            
        config = self.params
        config.train.src_dir = src_dir
        logger = get_logger('Evaluate', log_level=config['base']['log_level'])
        ## Modify params with what fits evaluation 
        config['train']['inverse']= False 
        config['train']['replace_ent']= False 

        logger.info('Get test data')
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
                                       entity_matcher= str(src_dir / "artifacts/matcher_model"),
                                       load_matcher=True)
            group_docs, _ = spacy_loader.group_ents(docs_container)
            test_data.loc[:, 'org_groups']= group_docs
        if extract_ent: 
            sents, spans, groups, aliases = inferer.spacy_loader.predictor(test_data['orig_sent'].tolist())
            test_data.loc[:, 'sents'] = sents
            test_data.loc[:, 'spans'] = spans
            test_data.loc[:, 'org_groups'] = groups
            test_data.loc[:, 'aliases'] = aliases
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
        # Initiate the inferer
        inferer = infer_from_trained(detect_entities=extract_ent,
                                     language_model="en_core_web_trf",
                                     require_gpu=True,
                                     basic_targets= ['supplier','customer'],
                                     load_matcher=True
                                    )

        for fold in self.folds:
            # Get the name of the fold
            name = os.path.basename(fold)
            # Update the config with the fold-specific parameters
            config['train']['train_data'] = f'{fold}/df_train.json'
            config['train']['valid_data'] = f'{fold}/df_test.json'
            config['train']['model_path'] = f'{fold}/re_model'
            config['train']['relations'] = f'{fold}/re_model/relations.pkl'

            # Load the pretrained model
            print(f"Loading the pretrained model: {name}")
            inferer.load_model(config['train'])
            # Define reports directories
            report_dir = src_dir/ f'{fold}/metrics'
            report_dir.mkdir(parents=True, exist_ok=True)
            # Test metricsa and errors reporting 
            test_errors, test_output = evaluation_report(inferer=inferer,
                              tagged_data=tagged,
                              tag_name='test',
                              report_dir= str(report_dir),
                              mutate= config.evaluate['mutate'],
                              reverse= config.evaluate['reverse'],
                              save_reports=True)
    def process_folds(self):
        """
        Process folds for relation extraction.

        @params:
        -------
        - folds_dir (str): A string containing the directory path to the folds.

        @returns:
        --------
        - tuple: A tuple containing the processed data and all_scores tensor.
        """
        # Create inferer
        inferer = infer_from_trained(detect_entities=False,
                                     language_model="en_core_web_trf",
                                     require_gpu=False,
                                     basic_targets=['supplier', 'customer'],
                                     load_matcher=False)

        # Load relations mapper
        relations_mapper = load_pickle(src_dir / self.params['train']['relations_mapper'])
        all_data = []
        all_scores = torch.tensor([])
        # Process each fold
        for fold in self.folds:
            fold_name = fold.split('/')[-1]
            self.logger.info(f"Process {fold_name}...")
            # Read validation set
            tagged_frame = pd.read_json(src_dir / os.path.join(fold, 'test.json'))
            self.logger.info("before augmentation {}".format(tagged_frame.shape))
            self.logger.info("reverse main relations")
            augmentations = tagged_frame.copy()
            augmentations.sents = augmentations.sents.apply(inverse_relations)
            # assert not all(augmentations.sents == df.sents)
            augmentations.relations = augmentations.relations.apply(
                lambda x: inverse_dict[x] )
            augmentations = augmentations.query("relations != 'other'").reset_index(drop=True)
            self.logger.info("augmenations shape {}".format(augmentations.shape))
            tagged_frame = pd.concat([tagged_frame, augmentations], axis=0).reset_index(drop=True) 
            # Load model
            self.logger.info("after augmentation {}".format(tagged_frame.shape))
            inferer.load_model({"model_path": src_dir / os.path.join(fold, 're_model'),
                                "batch_size": 16})
            # Make predictions
            tagged_frame = inferer.predict_fn(tagged_frame, reverse=False, mutate=True)
            # aggregate multi-positioning relations.
            id_scores = tagged_frame.groupby(['r_id'])\
             .apply(lambda x : list(np.mean(x['scores'].tolist(), axis=0))).to_dict()
            # assign aggregated relation for all positions
            #tagged_frame['scores'] = tagged_frame['r_id'].apply(lambda x: id_scores[x])
            # drop duplicates from multi-positions
            # tagged_frame.drop_duplicates(subset=['r_id'], inplace=True, ignore_index=True)
            # define max scores and its label_ids
            scores = torch.tensor(tagged_frame['scores'])
            score, labels = scores.max(1)
            # create relations info items to compine relations on each sentence
            # tagged_frame.loc[:, 'scores'] =  score
            tagged_frame.loc[:, 'prediction_id'] = labels
            tagged_frame.loc[:, 'prediction'] = tagged_frame['prediction_id']\
            .apply(lambda x : inferer.id2label[str(x)])
            tagged_frame['relations_id'] = tagged_frame['relations'].apply(lambda x: relations_mapper.rel2idx[x])
            # all_scores = torch.cat((all_scores, scores))
            all_data.append(tagged_frame)            

        # Concatenate data from all folds
        data = pd.concat(all_data, axis=0).reset_index(drop=True)
        data[['entity_1','entity_2']] = data['sents'].apply(rel_from_text)
        return data
    def search_candidates(self, data_dir, output_dir=None, to_process=40_000, return_samples=5000):        
        # read the data
        dataset = pd.read_json(data_dir)
        # determine the number of datapoints to process
        datapoints= dataset.shape[0] if dataset.shape[0] <= to_process else to_process

        preprocessed_data, _= self.relation_extractor.tag_sentences(
                                    dataset['sentence'].sample(datapoints).tolist()) 
        
        data_scope = preprocessed_data.sample(return_samples)
        
        
        
        
        

        for fold in self.folds:
            name = os.path.basename(fold)
            config['train']['model_path'] =  f'{fold}/re_model'
            ## Load fine-tune model
            self.relation_extractor.load_model(config['train'])
             # Compute model logits and losses
            out_scores, out_labels, out_losses, dropped = self.relation_extractor.predict_fn(data_scope,mutate=True)
            # Map true labels with label2id
            labels = list(map(lambda x : self.relation_extractor.id2label[str(x)] ,out_labels))
            data_scope.loc[:, name] = labels
            data_scope.loc[:, f"{name}_score"] = list(out_scores)
            
        # Drop unneeded columns
        data_scope.drop(columns=['input', 'e1_e2_start'], inplace=True)
        # Get fold names
        names = [os.path.basename(fold) for fold in self.folds]
        scores  = [f'{name}_score' for name in names]
        # Calculate the aggreed votes
        agreed_labels= []
        for models_preds in list(data_scope[names].itertuples(index=False, name=None)): 
            votes =np.array(models_preds)
            agreed = all(votes==votes[0])
            if agreed:
                agreed_labels.append(votes[0])
            else:
                agreed_labels.append(-1)
        data_scope.loc[:,'agreed_labels']= agreed_labels
        agreed_votes = data_scope[data_scope['agreed_labels'] != -1]
        print(f"Percentage of agreement is {len(agreed_votes) / len(data_scope)}")
        
        
        props_votes = []
        for i, row in data_scope.iterrows():
            votes = defaultdict(lambda: 0)
            for name in names:
                votes[row[name]] += row[f'{name}_score']
            votes = {k:(v/len(names)) for k,v in votes.items()}
            votes[f'higher_confident'] = max(votes, key=votes.get)
            props_votes.append(votes)
        
        data_scope = pd.concat([data_scope.reset_index(drop=True) , pd.DataFrame(props_votes)],axis=1)
        data_scope.loc[:, 'max_conf'] = data_scope[['supplier', 'customer', 'other']].max(axis=1)
        data_scope = data_scope.sort_values(by='max_conf', ascending=True)
        if output_dir is True:
            data_scope.to_json(output_dir)
        return data_scope
    def generate_issues_report(self):
        """Generate training data attached with issues report
        """
        # Generate cross_validation data with confident distrubution
        data = self.process_folds()
        # issues = find_label_issues(
        #                         data['relations_id'].to_numpy(),
        #                         scores.numpy(),
        #                         return_indices_ranked_by="self_confidence"
        #                         )
        data_dict = {"sentence": data['sents'].tolist() , 'relation': data.relations_id.tolist()}

        lab = Datalab(data_dict, label_name="relation")
        lab.find_issues(pred_probs=torch.tensor(data['scores']).numpy())
        examples_w_issue = (
            lab.get_issues("label")
            .query("is_label_issue")
            .sort_values("label_score")
        )

        self.logger.info(f"issues distribution:\n{data.iloc[examples_w_issue.index].relations.value_counts()}")
        data_issues = data.iloc[examples_w_issue.index]
        data_issues['label_score'] = examples_w_issue.label_score
        data_issues.to_json(os.path.join(self._models_dir, "issues_report.json"))


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
