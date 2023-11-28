'''This module contains variety of evaluations methods needed to evaluate ML models
'''
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from colorama import Fore, Style
import numpy as np
import pandas as pd 
from sklearn.metrics import accuracy_score, f1_score 
from sklearn.metrics import confusion_matrix as cm, classification_report as cr
import torch 
import torch.nn.functional as F


def model_eval(model,
               valid_loader,
               loss_function,
               device,
               logger,
               classes,
               evaluation_metrics=False,
               multi_model=False,
               set_name="valid",
               progress_bar=False): 
    '''Evaluate model based on evaluation metrics

    Args: 
        model(:class:`~transformers.PreTrainedModel` or :obj:`torch.nn.Module`) 
                    The model to train, evaluate or use for predictions. If not provided, a ``model_init`` must be passed.

        valid_loader (:obj:`torch.utils.data.dataset.Dataset`, `optional`):
            The dataset to use for evaluation. If it is an :obj:`datasets.Dataset`, columns not accepted by the
            ``model.forward()`` method are automatically removed.

        evaluation_metrics(Boolean): If True returns classification_report and confussion mertics, If Flase return
                                    f1_macro score only, default(False) 

    Returns: 
        evaluation_metrics(Dict[str:float]): dict carries requierd metrics
    '''
    tepoch = tqdm(valid_loader, unit="batch", colour='green') if progress_bar else valid_loader
    model = model.to(device)
    valid_loss = 0
    total = 0
    correct = 0
    y_pred_all = None
    labels_all = None
    with torch.no_grad():
        model.eval()
        for i , (inputs,mask,label,_,idx) in enumerate(tepoch):
            if labels_all is None:
                labels_all = label.detach() .cpu().numpy()
            else:
                labels_all = np.concatenate((labels_all, label.numpy()))
            inputs = inputs.to(device) if not multi_model else inputs 
            mask = mask.to(device) if not multi_model else mask 
            label= label.to(device)
            output = model(inputs,mask)[0]
            _ , predicted = torch.max(output, 1)
            y_pred = output.argmax(dim=1).cpu().numpy()
            if y_pred_all is None:
                y_pred_all = y_pred
            else:
                y_pred_all = np.concatenate((y_pred_all, y_pred))
            loss = loss_function(output, label)
            total += label.size(0)
            correct += (predicted.cpu() ==label.cpu()).sum()
            valid_loss += loss.item()
        accuracy = 100.00 * (correct.numpy() / total)
        valid_loss = valid_loss/len(valid_loader)
        f1 = f1_score(labels_all, y_pred_all, average='macro')    
        accuracy = accuracy_score(labels_all,y_pred_all)

        # print(f'{set_name}_Loss = {loss.item():.4f}')
        # print(f'{set_name}_Acc = {accuracy:.4f}')
        # print(Fore.CYAN, Style.BRIGHT, f'{set_name}-macro-f1 = {f1:.4f}')
        logger.info(f'{set_name}_Loss = {loss.item():.4f} | {set_name}_Acc = {accuracy:.4f} | {set_name}-Macro-F1 = {f1:.4f}')

    if evaluation_metrics: 
        classification_report = cr(labels_all , y_pred_all , target_names=classes , digits=4,output_dict=True) 
        confusion_mtx = cm(labels_all, y_pred_all)
        cm_df = pd.DataFrame(confusion_mtx, index=classes, columns=classes)
        cm_df = cm_df.apply(lambda row: row/row.sum(), axis=1).round(4)
        return f1,classification_report,cm_df
    return f1
