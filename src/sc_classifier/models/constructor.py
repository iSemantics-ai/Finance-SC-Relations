import torch
from torch import nn
from transformers import AutoModel, AutoModelForSequenceClassification
from transformers import AutoConfig

def construct_model(base_model:str, classes):
    """
    Constructing Bert Model
    """
    transformer_model = base_model
    config = AutoConfig.from_pretrained(
        transformer_model, 
        num_labels=len(classes),
        label2id= {k:i for i,k in enumerate(classes)},
        id2label={i:k for i,k in enumerate(classes)},
        hidden_dropout_prob = 0.1,
        attention_probs_dropout_prob=0.1
                                    )
    config.calsses = classes 

    config.output_attentions=True
    return AutoModelForSequenceClassification.from_pretrained(transformer_model,
                                                    config = config)