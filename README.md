# Finance-SC-Relations

Finance-SC-Relations is a project that focuses on extracting supply-chain relations from financial text using Transformer-based models and Large Language Models (LLMs). The project aims to enhance the accuracy and reliability of relation extraction in the supply-chain domain by leveraging the power of transformer-based models.

As an added value, this model can extract a company and its customers from a report and provide this information to investors and stake holders to do analysis or link to different databases using an [Entity Linking Model](https://github.com/iSemantics-ai/Entity-Linking).

## Usage

### Clone

```
git clone https://github.com/iSemantics-ai/Finance-SC-Relations.git
```

```
cd Finance-SC-Relations/
```

---

### Install Required Packages

```
sh install.sh
```

### Connect AWS's S3 Bucket

- Create a new dirctory for the credentials.

```
mkdir aws_cred
```

- Upload the credentials into the created path.

- Add the credentials using the AWS CLI

```
cp aws_cred ~/.aws -r
```

### Pull Data and Artifacts with `DVC`

```
dvc pull
```

### Load Inference Module

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

### Extract Supply Chain Relations from Text

```python
text = ['TWENTY-FIRST CENTURY FOX INC has an approximate 19% interest in Rotana Holding FZ-LLC ("Rotana"), a diversified media company in the Middle East and North Africa.']

output = relation_extractor.predict_relations(text)
```

### Extract Supply Chain Relations from File

```python
output = relation_extractor.predict_frame(path=<Local Dir>, sentence_column='Sentence')
```
