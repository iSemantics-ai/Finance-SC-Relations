# Finance-SC-Relations
Finance-SC-Relations is a project that focuses on extracting supply-chain relations from financial text using Transformer-based models and Large Language Models (LLMs). The project aims to enhance the accuracy and reliability of relation extraction in the supply-chain domain by leveraging the power of transformer-based models.

## Usage:
### _Clone_
```
git clone https://github.com/iSemantics-ai/Finance-SC-Relations.git
cd Finance-SC-Relations/
```
---
### _Install required packages_
``` sh install.sh```

### _Pull data and artifacts with `DVC`_
```
dvc pull
```
### _Load Inference Module_
```python
from src.relation_extraction.infer import infer_from_trained

relation_extractor = infer_from_trained(detect_entities=True,
                             language_model="en_core_web_trf",
                             require_gpu=True,
                             load_matcher=True)

model_args = {'model_path': 'artifacts/re_model', 
              'batch_size':32 }

relation_extractor.load_model(model_args)
```

### _Extract supply chain relations from sequence of text_
```python 
text = ['TWENTY-FIRST CENTURY FOX INC has an approximate 19% interest in Rotana Holding FZ-LLC ("Rotana"), a diversified media company in the Middle East and North Africa.']

output = relation_extractor.predict_relations(text)
```

### _Extract supply chain relations from file_ 
```python
output = relation_extractor.predict_frame(path=<Local Dir>, sentence_column='Sentence')
```
