'''This module consist of many customized techniques to perform scalable and 
    effecient training process, train monitor and even augment the training 
    dataset using contrastive training techniques.
'''
from sys import executable
import time 
import os 
import shutil
from pathlib import Path
from colorama import Fore, Style
import numpy as np 
import pandas as pd 
from typing import Dict, List, Tuple, Union
from tqdm import  tqdm
from collections import defaultdict
import json
import csv
import torch 
from torch import nn
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report as cr
from transformers.utils import logging
from torch.utils.data.dataloader import DataLoader
from transformers import logging
logging.set_verbosity_error()
from src.utils import get_logger, mutate_sent
from src.sc_classifier.processing.tokenizers import Transformer_Tokenizer
from src.sc_classifier.processing.validation import model_eval
from src.sc_classifier.processing import ( data_loader,
                                           train_set_pipe_line,
                                           load_and_valid_dataset,
                                           save_checkpoint,
                                           load_checkpoint,
                                           save_frame, 
                                           load_augmentations,
                                               )
from src.sc_classifier.models import constructor
from src.sc_classifier.config.core import (config,
                                       TRAINED_MODEL_DIR,
                                       DATASET_DIR,
                                       PACKAGE_ROOT)


from dotenv import load_dotenv
import wandb as WandB
load_dotenv()
Wandb_API= os.getenv('wandb_key')
if Wandb_API:
    WandB.login(key=Wandb_API)

class Trainer:
    """
    Trainer class for training models.
    """
    def __init__(
        self,
        loss_function: nn.Module=None ,
        optimizer: torch.optim.Optimizer=None,
        config:dict=config,
        model_name:str="sc_model",
        load_data:bool= False,
    ) -> None:
        """
        Initializes the essential variables for the training process.

        @params:
        -------
        classes: List of classes.
        loss_function: Loss function to be used in the model.
        optimizer: Optimizer to be used in the model.
        TrainArgs:
            base_model: Base model for the trainer.
            seed: Seed for reproducibility.
            train_path: Path to the train file.
            valid_path: Path to the validation file.
            project_name: Name of the project.
            model_name: Name of the model to be saved with.
            batch_size: Batch size for training.
            num_workers: Number of workers for training.
            load_data: Boolean flag to load data or not.
            load_pretrained: Boolean flag to load pre-trained weights.
            wandb: Boolean flag to use wandb or not.
        """
        self.project_name = config.app_config.package_name
        self.seed = config.app_config.random_state
        self.device = "cuda" if torch.cuda.is_available() else 'cpu'
        self.batch_size = config.train_args.batch_size
        self.num_workers = config.train_args.num_workers
        self.wandb = WandB if config.train_args.wandb else None

        # Construct and load the model
        self.classes = config.ml_model_config.classes
        self.base_model = config.train_args.base_model
        self.model = constructor.construct_model(self.base_model, classes=self.classes)
        # Define losses and optimizer
        self.loss_function = loss_function
        self.optimizer = optimizer
        # Define name for model to be saved
        self.model_name = self.model.config.name_or_path if not model_name else model_name
        self.logger = get_logger('SCClassifier', log_level=config.app_config.log_level)
        # Define data arguments
        self.train_file= PACKAGE_ROOT/ config.app_config.train_file
        self.valid_file= PACKAGE_ROOT / config.app_config.valid_file
        self.feature = config.ml_model_config.features[0]
        self.target = config.ml_model_config.target
        self.basic_columns = config.app_config.basic_columns
        self.index_col = config.app_config.index_col
        # Try loading pre-trained if exist
        self.score, self.trained_epochs = 0.0, 0
        _pre_exist = (TRAINED_MODEL_DIR / "{}/{}".format(self.model_name,
                                                         'pytorch_model.pt')).is_file()
        if _pre_exist and config.train_args.load_pretrained:
            self.model, self.trained_epochs, self.score = load_checkpoint(model= self.model,
                                                        path= TRAINED_MODEL_DIR / self.model_name)
            self.logger.info("loading checkpoint from `{}`".format(self.model_name))
            
        self.model = self.model.to(self.device)
        # Define model name or path
        self.tokenizer  = Transformer_Tokenizer(self.model.config.name_or_path)        
        # If we want to use this model for predictions only, assign load_data = False 
        if load_data:
            self._load_training_data()
        else:
            self.logger.info("inference mode...")
            self._train_ds, self._valid_ds, self._test_ds = None, None, None
    def load_model(self): 
        try : 
            self.model, self.trained_epochs, self.score = load_checkpoint(model= self.model,
                                                                         path= TRAINED_MODEL_DIR / self.model_name)
        except:
            self.model = constructor.construct_model(self.base_model,
                                                     classes=self.classes).to(self.device)
            self.score, self.trained_epochs = 0.0, 0
    @property
    def train_ds(self): 
        return self._train_ds 
    
    @property
    def valid_ds(self): 
        return self._valid_ds 

    @property
    def test_ds(self): 
        return self._test_ds 
    

    @property
    def save_path(self):
        '''save path for transformers models'''
        return TRAINED_MODEL_DIR

    def set_augmentations(self, data:pd.DataFrame ,feature=None , target=None):
        feature = feature or self.feature
        target = target or self.target
        data = self.tokenizer.batch_encode_plus(data[feature].tolist(), data[target].tolist())
        self._train_ds = pd.concat([data, self.train_ds],axis = 0).reset_index(drop=True)

    def _load_training_data(self)->None:
        """Read and prepare train and validation set for training.
        this method depends on TraingArgs to determine which files to read.
        @args
        -----
        train_file
        valid_file
        feature
        target
        index_col
        basic_columns
        mutate
        """
        # Read train and validation sets from JSON files
        train = pd.read_json(self.train_file)
        valid = pd.read_json(self.valid_file)

        # Drop duplicate rows based on an index column
        train.drop_duplicates(self.index_col, inplace=True, ignore_index=True)
        valid.drop_duplicates(self.index_col, inplace=True, ignore_index=True)
        if dict(config.train_args).get("prune_stratify", False):
            max_rows = config.train_args.get("max_stratify", len(train))
            train = pd.concat([train.query(f"{self.target}=='{f}'")[:max_rows] \
                               for f in self.classes], axis=0).reset_index(drop=True)
    
        # validate that data contains all required inputs for train and validation
        if not all([x in train.columns or x in valid.columns for x in self.basic_columns]):
            raise ("Invalid train or valid dataset, missing required columns...")
        # transform inputs using pre-trained transformer tokenizer
        if isinstance(valid[self.target].iloc[0], str):

            train[self.target] = train[self.target].apply(lambda x: \
                                                    int(self.model.config.label2id[x])) 
            valid[self.target] = valid[self.target].apply(lambda x: \
                                                    int(self.model.config.label2id[x]))
        # mask entities
        # mutate the text to mask certain ents
        if config.train_args.mutate and 'org_groups' in train:
            train.loc[:, 'orig_sent'] = train[self.feature].tolist()
            valid.loc[:, 'orig_sent'] = valid[self.feature].tolist()
            tqdm.pandas(desc='mutate train text')
            train.loc[:, self.feature] = train.progress_apply(lambda x: \
                                              mutate_sent(sent=x[self.feature],
                                              org_groups=x['org_groups'],
                                              spans= x.get('spans')),axis=1)
            tqdm.pandas(desc='mutate valid text')
            valid.loc[:, self.feature] = valid.progress_apply(lambda x: \
                                              mutate_sent(sent=x[self.feature],
                                              org_groups=x['org_groups'],
                                              spans= x.get('spans')),axis=1)


        # Estimate proper value for max length
        self._max_len = int(train[self.feature].apply(lambda x: len(x.split(' '))).quantile(0.99)+20)
        # Create training and validation datasets by combining tokenized text, labels, and other columns
        self._train_ds = pd.concat([
            self.tokenizer.batch_encode_plus(docs=train[self.feature].tolist(),
                                             labels=train[self.target].tolist(),
                                            max_length=self._max_len),
            train[self.basic_columns]
        ], axis=1)

        self._valid_ds = pd.concat([
            self.tokenizer.batch_encode_plus(docs=valid[self.feature].tolist(),
                                             labels=valid[self.target].tolist(),
                                             max_length=self._max_len),
            valid[self.basic_columns]
        ], axis=1)

    
    def train(self,
            hp:dict,
            augmentations=[],
            save= True,
            train_description=None,
            train_ds=None,
            augment=True,
            test = True, 
            steps=np.Inf, 
            checkpoint='steps'
            ):
        '''Train instance model, with given hp dict
        
        @params
        -------
        hp(dict): dictionary of hpt for training and regularization 
        augmentations: If exist it get saved with model artifacts for later use
        save(bool): Boolean value to indicate if the model artifacts will 
        '''
        model_path = TRAINED_MODEL_DIR / self.model_name
        
        torch.cuda.empty_cache()
        # create data loaders 

        train_ds = train_ds if train_ds is not None else self.train_ds
        if augment is True: 
            augs = load_augmentations(self.model_name)
            if augs is not None:
                augs = self.tokenizer.batch_encode_plus(augs.text.tolist(), augs.label_id.tolist())
                train_ds  = pd.concat([train_ds , augs] , axis=0).reset_index(drop=True)
       
        train_loader = data_loader(train_ds, hp['batch_size'], shuffle=True) 
        valid_loader = data_loader(self.valid_ds, hp['batch_size'], shuffle=False) 
        
        
        #create optimizer and scheduler
        optimizer = self.optimizer(self.model.parameters(), lr=hp['learning_rate'], weight_decay=hp['weight_decay'])
        # A warmup scheduler
        t_total = hp['epochs'] * len(train_loader) * hp['warmup_smooth']
        warmup_steps = np.ceil(t_total / 10.0) * 2
        scheduler = get_cosine_schedule_with_warmup(
                                optimizer,
                                num_warmup_steps=warmup_steps,
                                num_training_steps=t_total)
        # Initiate WandB job
        if self.wandb != None:
            '''Setup WandB for  training monitoring'''
            self.wandb.init(
            # Set the project where this run will be logged
            project=self.project_name, 
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=f"{self.model_name}"if not train_description else train_description, 
            # Track hyperparameters and run metadata
            config=hp
            )
            
        # Load train_logs or define
        if config.train_args.load_pretrained and (model_path / "train_logs.json").is_file():
            with open(model_path / "train_logs.json", 'r') as ob:
                train_logs = json.load(ob)
        else:
            train_logs = defaultdict(lambda: [])
        self.logger.info("---------Trainig Started---------")
        for epoch in range(hp['epochs']):
            self.trained_epochs += 1
            train_loss = 0.0
            total = 0
            correct = 0
            self.model.train()
            self.logger.info('EPOCH -- {}'.format(epoch+1))
            with tqdm(train_loader, unit="batch", colour='MAGENTA' ) as tepoch:
                y_pred_all = None
                labels_all = None
                
                for i, (inputs,mask,label,_,_)  in enumerate(tepoch):

                    if labels_all is None:
                        labels_all = label.numpy()
                    else:
                        labels_all = np.concatenate((labels_all, label.numpy()))

                    optimizer.zero_grad()
                    # Point inputs to chossen device
                    inputs = inputs.to(self.device)
                    mask = mask.to(self.device)
                    label= label.to(self.device)
                    # Feedforward the model and return outputs(logits)
                    output = self.model(inputs,mask)[0]
                    _ , predicted = torch.max(output, 1)
                    y_pred = output.argmax(dim=1).cpu().numpy()

                    if y_pred_all is None:
                        y_pred_all = y_pred
                    else:
                        y_pred_all = np.concatenate((y_pred_all, y_pred))

                    total += label.size(0)

                    correct += (predicted.cpu() ==label.cpu()).sum()
                    # Calculate loss
                    loss = self.loss_function(output, label)
                    # Calculate gradients with respect to loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
                    # Update weights to minimze the loss
                    optimizer.step()
                    # Calculate loss and other metrics overall batches to infer with progress par
                    train_loss = train_loss + ((1 / (i + 1)) * (loss.data - train_loss))
                    accuracy = accuracy_score(labels_all,y_pred_all)
                    # Reflect results over the progress par
                    tepoch.set_description(f"Train Acc: {(100.00*accuracy):.4f}")
                    tepoch.set_postfix(loss=train_loss.item())
                    scheduler.step()
                    if self.wandb is not None:
                        self.wandb.log({'learning_rate':scheduler.get_last_lr()[0]})

                    if (i+1) % steps == 0:
                        train_f1 = f1_score(labels_all, y_pred_all, average='macro')
                        save_steps = True if checkpoint == 'steps' else False
                        self.train_eval(valid_loader ,train_f1 ,epoch ,hp, augmentations , save_steps, test=test)


                train_f1 = f1_score(labels_all, y_pred_all, average='macro')    
                self.logger.info(f'Train-Macro-F1 = {train_f1:.4f}')
                val_f1 = self.train_eval(valid_loader,
                                train_f1,
                                epoch,
                                hp,
                                augmentations,
                                save)
                # Append all metrics info for the epoch
                train_logs['train_loss'].append(train_loss.item())
                train_logs['train_f1'].append(train_f1)
                train_logs['val_f1'].append(val_f1)
                for key, logs in train_logs.items():
                    f  = open(PACKAGE_ROOT/ '{}/sc_{}.csv'.format(config.train_args.metric_dir,key),'w')
                    # using csv.writer method from CSV package
                    write = csv.writer(f)
                    write.writerow(['epoch', 'sc_{}'.format(key)])
                    write.writerows([(epoch, log) for epoch, log in enumerate(logs)])
                    f.close()

        print(Fore.MAGENTA,f"Highest Macro-F1 score : {self.score}")
        try: 
            if save:
                self.load_model() 
            else: 
                hp['score'] = self.score 
                return hp 
        except: 
            print("no progress achieved...")
        
        if self.wandb != None: 
            self.wandb.finish()

        hp['score'] = self.score
        return hp


    def eval_report(self, data, thresholds=[0.90, 0.95, 0.99], feature=None, target=None):
        """
        Evaluate the performance of a model on a given dataset and print classification reports at different thresholds.

        @params
        -------
        - data: The dataset to evaluate the model on
        - thresholds: A list of confidence score thresholds to use for filtering predictions
        - feature: The name of the feature column in the dataset
        - target: The name of the target column in the dataset

        @returns
        --------
        - A pandas DataFrame containing the predictions and classification reports at different thresholds
        """
        feature = feature or self.feature
        target = target or self.target
        self.model.to(self.device)
        predictions = self.predict(data[feature], accumilate=False, save=False)
        predictions['true'] = data[target]
        preds = pd.DataFrame(predictions[self.model_name].tolist()).rename({0:'score', 1:"predicted"}, axis=1)
        predictions = pd.concat([predictions[[self.feature, 'true']], preds[['score', 'predicted']]], axis=1)
        classification_report = cr(predictions['true'].tolist(), predictions['predicted'].tolist(),
                                   target_names=self.classes, digits=4, output_dict=True)
        print(f'The classification report:\n{pd.DataFrame(classification_report).T}\n')
        for threshold in thresholds:
            filtered = predictions[predictions['score'] > threshold]
            # print threshold status
            print(f'At threshold `{threshold}` dropped: datapoint={predictions.shape[0]- filtered.shape[0]} frac={str(1 -(filtered.shape[0] / predictions.shape[0]))}')
            # calculate the merics required
            y_true = filtered['true'].tolist()
            y_pred = filtered['predicted'].tolist()
            classification_report = cr(y_true, y_pred, target_names=self.classes, digits=4, output_dict=True)
            print(f"Classification report for score above threshold:\n{pd.DataFrame(classification_report).T}\n\n")

        return predictions

    def train_eval(self,valid_loader ,train_f1 ,epoch ,hp,augmentations , save):
        
        print(Fore.CYAN,Style.BRIGHT, "---------Validation---------")
        (val_f1,
         cr,
         conf_mat) = model_eval(self.model,
                                valid_loader,
                                self.loss_function,
                                self.device,
                                self.logger,
                                evaluation_metrics=True,
                                classes=self.classes,
                                set_name="valid set")
        cr = pd.DataFrame(cr).T
        test_f1 = 0
        if self.wandb != None: 
            self.wandb.log({"train_f1": train_f1, "val_f1": val_f1 })
        
        if val_f1 > self.score:
            # Save used in hyperparameter tuning jobs
            self.logger.info(f'valid macro_f1 increased from {self.score} to {val_f1} ')
            self.score = val_f1
            if save :    
                self.model.config.train_hp = hp
                self.logger.info("Saving Checkpoint ....")
                save_checkpoint(model=self.model,
                                epoch=self.trained_epochs,
                                score=self.score,
                                tokenizer=self.tokenizer,
                                path= TRAINED_MODEL_DIR / self.model_name, augmentations=augmentations)
                metrics = {
                "sc_valid_accuracy": round(cr.loc["accuracy"]["f1-score"], 4),
                "sc_valid_f1_macro": round(cr.loc["macro avg"]["f1-score"],4),
                "sc_valid_recall": round(cr.loc["macro avg"]["recall"], 4),
                "sc_valid_precision": round(cr.loc["macro avg"]["precision"], 4),
                **{
                    f"sc_{k}_f1_score": round(cr.loc[str(k)]["f1-score"], 4)
                    for k in self.model.config.label2id.keys()
                        },
                    }
                with open(PACKAGE_ROOT/ '{}/sc_valid_metrics.json'\
                          .format(config.train_args.metric_dir),'w') as o:
                    json.dump(metrics, o)
                    
        else:
            print(Fore.RED ,f"val_macro_f1 did not optimize: {self.score}")
            # self.load_model()
        print(Fore.LIGHTWHITE_EX,"="*80)
        self.model = self.model.to(self.device)
        return val_f1

    def eval(self, data:pd.DataFrame=None, batch_size=None)->float:
        """evaluate model on the test set with displaying classification report.

        @returns:
        float: score of validation
        """
        if self.score == 0.0:
            raise ValueError("model is not trained on this task, train before performing any evaluations")
        
        batch_size = self.model.config.train_hp['batch_size'] if not batch_size else batch_size
        validation_set = data if data is not None else self.test_ds
        test_loader = data_loader(validation_set,batch_size, shuffle=False) 
        test_f1, classification_report, conf_mat = model_eval(self.model,test_loader, self.loss_function, self.device ,self.logger, evaluation_metrics=True, classes=self.classes) 
        report = pd.DataFrame(classification_report).T
        self.logger.info("---evaluation---")
        self.logger.info(f"\n{report}")
        return report

    def hpt(self,
        hps:List[Dict],
        save=False)->None:
        score= 0 
        best_hp = None        
        self.logger.info("Starting >> Hyper-Paramenters-Tuning Job over...")
        for ex_number, hp in enumerate(hps):
            if not save :
                self.load_model()             
            hp_job = self.train(hp, save=save)
            if hp_job['score'] > score: 
                score = hp_job['score']
                best_hp = hp_job
        self.load_model() 
        self.logger.info("Hyper-Paramenters-Tuning Job over...")
        return best_hp


    @staticmethod
    def predict_fn(model:nn.Module, loader:DataLoader, device)-> List[Tuple[float,int]]: 
        """predict stream of encoded data

        Args:
            model (nn.Module): pre-trained model 
            loader (DataLoader): batch generator 
            device (_type_): device where model exists

        Returns:
            List[Tuple[float,int]]: list of scores and labels
        """
        preds = []
        scores= []
        softmax = nn.Softmax(dim=1)
        with torch.no_grad():
            model.eval()
            with tqdm(loader, unit="batch" , colour='green') as tepoch:
                for inputs, mask, text, idx  in tepoch:
                    inputs = inputs.to(device)
                    mask = mask.to(device)
                    output = model(inputs,mask)[0]
                    _, predicted = torch.max(output.data, 1)
                    preds += predicted.tolist()
                    scores += softmax(output).cpu().detach().tolist()

        return np.array(scores, dtype=np.float16), preds, idx
    
    def predict_seq(self,
                    data:Union[str,List[str]],
                    max_length:int=264,
                    save=False)->Union[Tuple,pd.DataFrame]:
        '''Compute prediction fot single text or list of strings 
        
        Args:
        data (Union[str,List[str]]) : could be single text raw or sequence of texts.
        save (bool): Save predictions as json file, (default=False)
        
        Returns:
        Union[Tuple,pd.DataFrame] : Tuple for single Text, or DataFrame for sequence of texts 

        '''

        self.model.eval() 
        if isinstance(data, str): 
            softmax = nn.Softmax(dim=1)
            with torch.no_grad():
                input = self.tokenizer.encode(data, max_length=max_length).to(self.device)
                output = self.model(input)[0].cpu()
            prob = softmax(output).numpy()
            label = np.argmax(prob, axis=1)[0]
            return prob.max(), label
        
        if isinstance(data, (list, tuple, pd.Series)):
            data = self.tokenizer.batch_encode_plus(list(data), max_length=max_length) 
            loader = data_loader(encoded_data=data, batch_size=self.batch_size, shuffle=False)
            scores, preds, idx = self.predict_fn(self.model, loader, self.device)
            return scores, preds


    def predict(self,data:Union[str,List[str]]=None , file_name:str=None , accumilate:bool=False, save=False)->pd.DataFrame:
        """Predict raws of text

        Args:
            data:Union[str,List[str]]: str or list[str], if exist 
                        prioritize to predict the data.

            file_name(str): the path of the text files

            accumilate(bool): overwrite the file with it's model_name as column
                            and predictions as rows.(default=True)
                
            save(bool): if True the model will save new_file or overwrite an existing file for same data

        Raises:
            ValueError: if score == 0 that mean model didn't train before
        """

        if self.score == 0.0:
            raise ValueError("model is not trained on this task, train before performing any predictions")
        
        if data is not None:
            return self.predict_seq(data, save=save)

        data_columns = ['transformer_ids', 'attention_mask', 'text']
        in_path =  file_name
        out_path = f"output/{file_name}"
    
        if accumilate and os.path.exists( DATASET_DIR /out_path):
            in_path = out_path
        
        df = load_and_valid_dataset(file_name=in_path)
        dataset = self.tokenizer.batch_encode_plus(df[config.ml_model_config.features[0]].tolist()) 
        assert set(data_columns).issubset(dataset.columns)
        loader = data_loader(encoded_data=dataset, batch_size=self.batch_size, shuffle=False)
        scores, preds, idxs = self.predict_fn(self.model, loader, self.device)

        predictions = list(zip(scores[np.arange(len(scores)), preds].tolist(), preds))

        assert len(predictions) == df.shape[0]
        df[self.model_name] = predictions 
        if accumilate:
            saved = save_frame(df, path=DATASET_DIR/out_path)
            if not saved:
                self.logger.info("failed to save the predictions!")
            return df 

        elif save: 
            out_split = out_path.split('.')
            saved = save_frame(df, path= DATASET_DIR / f'{out_split[0]}_{self.model_name}.{out_split[1]}')
            if not saved:
                self.logger.info("failed to save the predictions!")
            return df
        return df 

        

