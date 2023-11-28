#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 09:37:26 2019

@author: weetee
"""
import os
from itertools import cycle
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import (
    roc_curve,
    auc,
    roc_auc_score,
    classification_report,
    RocCurveDisplay,
)

import logging
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)
logger = logging.getLogger(__file__)


def load_state(net, model_path, optimizer, scheduler, load_best=False):
    """Loads saved model and optimizer states if exists"""

    amp_checkpoint = None
    checkpoint_path = model_path / "checkpoint.pth.tar"
    best_path = model_path / "model.pth.tar"
    start_epoch, best_pred, checkpoint = 0, 0, None
    if (load_best is True) and os.path.isfile(best_path):
        if torch.cuda.is_available():
            print("Load with CUDA")
            checkpoint = torch.load(best_path)
        else:
            print("Load with CPU")
            checkpoint = torch.load(best_path, map_location=torch.device("cpu"))
        logger.info("Loaded model.")
    elif os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        logger.info("Loaded checkpoint model.")
    if checkpoint != None:
        start_epoch = checkpoint["epoch"]
        best_pred = checkpoint["best_score"]
        net.load_state_dict(checkpoint["state_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        amp_checkpoint = checkpoint["amp"]
    return start_epoch, best_pred, amp_checkpoint


def load_results(model_path, logs_columns):
    """Loads saved results if exists"""

    files_ext = all([os.path.isfile(f"metrics/{col}.csv") for col in logs_columns])
    if files_ext:
        logs = pd.concat(
            [pd.read_csv(f"metrics/{col}.csv")[[col]] for col in logs_columns], axis=1
        ).to_dict(orient='records')
        logger.info("Loaded results buffer")
    else:
        logs = []
    return logs


def evaluate_(output, labels, ignore_idx):
    ### ignore index 0 (padding) when calculating accuracy
    idxs = (labels != -1).squeeze()
    scores = torch.softmax(output, dim=1)
    o_labels = scores.max(1)[1]
    l = labels.squeeze()[idxs]
    o = o_labels[idxs]
    if len(idxs.size()) == 0:
        o = o[0]
        acc = (l == o).sum().item()

    elif len(idxs) > 1:
        acc = (l == o).sum().item() / len(idxs)
    else:
        acc = (l == o).sum().item()

    l = l.tolist() 
    o = o.tolist() 
    scores = scores.tolist()
    return acc, (o, l), scores



def _roc_auc(y_test, y_scores, label_binarizer, id2label):
    y_onehot_test = label_binarizer.transform(y_test)
    micro_roc_auc_ovr = roc_auc_score(
        y_test,
        y_scores,
        multi_class="ovr",
        average="macro",
    )
    # store the fpr, tpr, and roc_auc for all averaging strategies
    fpr, tpr, roc_auc = dict(), dict(), dict()
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot_test.ravel(), y_scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    n_classes = y_onehot_test.shape[1]
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_onehot_test[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr_grid = np.linspace(0.0, 1.0, 1000)

    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(fpr_grid)

    for i in range(n_classes):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation

    # Average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = fpr_grid
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    fig, ax = plt.subplots(figsize=(6, 6))

    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label=f"micro-average ROC curve (AUC = {roc_auc['micro']:.2f})",
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label=f"macro-average ROC curve (AUC = {roc_auc['macro']:.2f})",
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for class_id, color in zip(range(n_classes), colors):
        RocCurveDisplay.from_predictions(
            y_onehot_test[:, class_id],
            y_scores[:, class_id],
            name=f"ROC curve for {id2label[class_id]}",
            color=color,
            ax=ax,
        )

    plt.plot([0, 1], [0, 1], "k--", label="ROC curve for chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(
        "Extension of Receiver Operating Characteristic\nto One-vs-Rest multiclass"
    )
    plt.legend()
    return fig


def evaluate_results(
    net, test_loader, label_binarizer, id2label, pad_id, cuda, calc_ruc=True
):
    logger.info("Evaluating test samples...")

    acc = 0
    out_labels = []
    true_labels = []
    all_scores = []
    net.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
            x, e1_e2_start, labels, sents_ids, _,  _, _ = data
            attention_mask = (x != pad_id).float()
            token_type_ids = torch.zeros((x.shape[0], x.shape[1])).long()

            if cuda:
                x = x.cuda()
                attention_mask = attention_mask.cuda()
                token_type_ids = token_type_ids.cuda()

            classification_logits = net(
                x,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                Q=None,
                e1_e2_start=e1_e2_start,
            ).detach().cpu()

            accuracy, (o, l), scores = evaluate_(
                classification_logits, labels, ignore_idx=-1
            )
            out_labels += o
            true_labels += l
            all_scores += scores
            acc += accuracy
    if calc_ruc:
        roc_plot = _roc_auc(
            np.array(true_labels), np.array(all_scores), label_binarizer, id2label
        )
    else:
        roc_plot = None
    cr = classification_report(
        true_labels,
        out_labels,
        output_dict=True,
        target_names=list(net.config.id2label.values()),
    )

    return pd.DataFrame(cr).T, roc_plot

    # logger.info("Evaluating test samples...")
    # acc = 0; out_labels = []; true_labels = []
    # net.eval()
    # with torch.no_grad():
    #     for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
    #         x, e1_e2_start, labels, _,_,_ = data
    #         attention_mask = (x != pad_id).float()
    #         token_type_ids = torch.zeros((x.shape[0], x.shape[1])).long()

    #         if cuda:
    #             x = x.cuda()
    #             labels = labels.cuda()
    #             attention_mask = attention_mask.cuda()
    #             token_type_ids = token_type_ids.cuda()

    #         classification_logits = net(x, token_type_ids=token_type_ids, attention_mask=attention_mask, Q=None,\
    #                       e1_e2_start=e1_e2_start)

    #         accuracy, (o, l) = evaluate_(classification_logits, labels, ignore_idx=-1)
    #         out_labels.append([str(i) for i in o]); true_labels.append([str(i) for i in l])
    #         acc += accuracy

    # accuracy = acc/(i + 1)
    # results = {
    #     "accuracy": accuracy,
    #     "precision": precision_score(true_labels, out_labels),
    #     "recall": recall_score(true_labels, out_labels),
    #     "f1": f1_score(true_labels, out_labels)
    # }
    # logger.info("***** Eval results *****")
    # for key in sorted(results.keys()):
    #     logger.info("  %s = %s", key, str(results[key]))

    # return results
