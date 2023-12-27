import os
import re
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from num2words import num2words
from sklearn.preprocessing import LabelBinarizer
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from .misc import save_as_pickle, load_pickle
from src.utils.preprocess import spread_rows, mutate_sent
from tqdm import tqdm
from transformers import AutoTokenizer as Tokenizer
import logging

tqdm.pandas(desc="prog_bar")
logging.basicConfig(
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)
logger = logging.getLogger("__file__")
inverse_dict = {"supplier": "customer", "customer": "supplier", "other": "other"}

def process_text(text, mode="train"):
    sents, relations, comments, blanks = [], [], [], []
    for i in range(int(len(text) / 4)):
        sent = text[4 * i]
        relation = text[4 * i + 1]
        comment = text[4 * i + 2]
        blank = text[4 * i + 3]

        # check entries
        if mode == "train":
            assert int(re.match("^\d+", sent)[0]) == (i + 1)
        else:
            assert (int(re.match("^\d+", sent)[0]) - 8000) == (i + 1)
        assert re.match("^Comment", comment)
        assert len(blank) == 1

        sent = re.findall('"(.+)"', sent)[0]
        sent = re.sub("<e1>", "[E1]", sent)
        sent = re.sub("</e1>", "[/E1]", sent)
        sent = re.sub("<e2>", "[E2]", sent)
        sent = re.sub("</e2>", "[/E2]", sent)
        sents.append(sent)
        relations.append(relation), comments.append(comment)
        blanks.append(blank)
    return sents, relations, comments, blanks


def inverse_relations(sent):
    sent = sent.replace("[E1]", "[E3]")
    sent = sent.replace("[/E1]", "[/E3]")
    sent = sent.replace("[E2]", "[E4]")
    sent = sent.replace("[/E2]", "[/E4]")
    sent = sent.replace("[E4]", "[E1]")
    sent = sent.replace("[/E4]", "[/E1]")
    sent = sent.replace("[E3]", "[E2]")
    sent = sent.replace("[/E3]", "[/E2]")
    return sent


def read_frame(file_path, suffix):
    if suffix == ".csv":
        return pd.read_csv(file_path)
    elif suffix == ".json":
        return pd.read_json(file_path)
    elif suffix == ".tsv":
        return pd.read_csv(file_path, sep="\t")
    else:
        raise TypeError("{0} is invalid".format(suffix))



def preprocess_custom_data(args):
    """
    Preprocess custom data for training.
    
    @params:
    -------
    - args (Namespace): The arguments containing preprocessing parameters.

    @returns:
    --------
    - Tuple: Processed DataFrames and relations mapper.
    """
    processed_frames = []

    for file_path in args.train_preprocess.files:
        file = Path(file_path)
        file_name = file.name.replace(file.suffix, "")
        print("Processing file:", file_name)

        # Check if the file exists
        if not file.is_file():
            raise FileNotFoundError(f"File not found: {file_name}")

        # Read data from the file into a DataFrame
        df = read_frame(file, file.suffix)

        # Perform data augmentation if specified conditions are met
        if args.train_preprocess.inverse and not args.base.entity_static_position:
            df = augment_data_with_inverse_relations(df, args.base.index_col)
        else:
            # Balance relations in the data
            df = balance_relations(df, args.base.main_relaions, args.base.index_col)
        # Check and load or create Relations Mapper
        relations_mapper = load_or_create_relations_mapper(
            args.train_preprocess.relations_mapper, df, file_name
        )
        # Assign numerical IDs to relations
        df["relations_id"] = df.progress_apply(
            lambda x: relations_mapper.rel2idx[x["relations"]], axis=1
        )

        # Save processed DataFrame as a JSON file
        output_file_path = args.train_preprocess.output_dir + f"df_{file_name}.json"
        df.to_json(output_file_path)

        processed_frames.append(df)

    processed_frames.append(relations_mapper)

    return tuple(processed_frames)

def augment_data_with_inverse_relations(df, index_col):
    """
    Augment data by creating a copy with reversed relations.

    @params:
    -------
    - df (DataFrame): Input DataFrame.
    - index_col (str): Index column that identifies each unique sentence on the DataFrame
    
    @returns:
    --------
    - DataFrame: Augmented DataFrame.
    """
    logger.info("Duplicating relations with reversed direction.")
    augmentations = df.copy()
    augmentations.sents = augmentations.sents.apply(inverse_relations)
    augmentations.relations = augmentations.relations.apply(
        lambda x: inverse_dict[x]
    )
    augmentations = augmentations.query("relations != 'other'").reset_index(drop=True)
    dataset = pd.concat([df, augmentations], axis=0).reset_index(drop=True)
    idxs = dataset[index_col].unique()
    np.random.shuffle(idxs)
    return spread_rows(dataset, idxs, index_col)

def balance_relations(df, main_relations, index_col):
    """
    Balance relations in the DataFrame.

    @params:
    -------
    - df (DataFrame): Input DataFrame.
    - main_relations (str): Relations with reflection property
    - 
	TODO: Comment the code lines
    @returns:
    --------
    - DataFrame: DataFrame with balanced relations.
    """
    main_relations = set(main_relations) & set(df.relations.unique())
    total_relations = df['relations'].isin(main_relations).sum()
    majority = sorted(df['relations'].value_counts()
                     [list(main_relations)].to_dict().items(),
                     key=lambda x: x[1], reverse=True)[0]
    majority_rows = df.query(f"relations == '{majority[0]}'")
    balance_idx = majority_rows.sample(majority[1] - (total_relations // 2)).index
    df.loc[balance_idx, 'sents'] = df.sents.apply(inverse_relations)
    df.loc[balance_idx, 'relations'] = df.relations.apply(
        lambda x: inverse_dict[x]
    )
    idxs = df[index_col].unique()
    np.random.shuffle(idxs)
    return spread_rows(df, idxs, index_col)

def load_or_create_relations_mapper(mapper_path, df, file_name, existing_mapper= None):
    """
    Load or create Relations Mapper.

    @params:
    -------
    - mapper_path (str): Path to the Relations Mapper file.
    - df (DataFrame): Input DataFrame.
    - file_name (str): Name of the file being processed.

    @returns:
    --------
    - Relations_Mapper: Loaded or created Relations Mapper.
    """
    if os.path.isfile(mapper_path):
        relations_mapper = load_pickle(mapper_path)
    elif file_name == "train":
        relations_mapper = Relations_Mapper(df["relations"])
        save_as_pickle(mapper_path, relations_mapper)
    else:
        raise FileNotFoundError(f"Missing RelationsMapper artifact. Must exist in the following dir: <{mapper_path}>")

    return relations_mapper

def inverse_relations(sent):
    sent = sent.replace("[E1]", "[E3]")
    sent = sent.replace("[/E1]", "[/E3]")
    sent = sent.replace("[E2]", "[E4]")
    sent = sent.replace("[/E2]", "[/E4]")
    sent = sent.replace("[E4]", "[E1]")
    sent = sent.replace("[/E4]", "[/E1]")
    sent = sent.replace("[E3]", "[E2]")
    sent = sent.replace("[/E3]", "[/E2]")
    return sent


class Relations_Mapper(object):
    def __init__(self, relations):
        self.rel2idx = {}
        self.idx2rel = {}

        logger.info("Mapping relations to IDs...")
        self.n_classes = 0
        for relation in tqdm(relations):
            if relation not in self.rel2idx.keys():
                self.rel2idx[relation] = self.n_classes
                self.n_classes += 1

        for key, value in self.rel2idx.items():
            self.idx2rel[value] = key

class Pad_Sequence:
    """
    collate_fn for dataloader to collate sequences of different lengths into a fixed length batch
    Returns padded x sequence, y sequence, x lengths and y lengths of batch
    """

    def __init__(
        self,
        seq_pad_value,
        label_pad_value=-1,
        label2_pad_value=-1,
    ):
        self.seq_pad_value = seq_pad_value
        self.label_pad_value = label_pad_value
        self.label2_pad_value = label2_pad_value

    def __call__(self, batch):
        sorted_batch = batch
        seqs = [x[0] for x in sorted_batch]
        seqs_padded = pad_sequence(
            seqs, batch_first=True, padding_value=self.seq_pad_value
        )
        x_lengths = torch.LongTensor([len(x) for x in seqs])

        labels = list(map(lambda x: x[1], sorted_batch))
        labels_padded = pad_sequence(
            labels, batch_first=True, padding_value=self.label_pad_value
        )
        y_lengths = torch.LongTensor([len(x) for x in labels])
        
        if len(sorted_batch[0]) == 3:
            idxs = list(map(lambda x: x[2], sorted_batch))
            ids_padded = pad_sequence(idxs, batch_first=True, padding_value=self.label_pad_value)
            return seqs_padded, labels_padded, idxs, x_lengths, y_lengths
        labels2 = list(map(lambda x: x[2], sorted_batch))
        labels2_padded = pad_sequence(
            labels2, batch_first=True, padding_value=self.label2_pad_value
        )
        y2_lengths = torch.LongTensor([len(x) for x in labels2])

        idxs = list(map(lambda x: x[3], sorted_batch))
        ids_padded = pad_sequence(idxs, batch_first=True, padding_value=self.label_pad_value)

        
        return (
            seqs_padded,
            labels_padded,
            labels2_padded,
            idxs,
            x_lengths,
            y_lengths,
            y2_lengths,
        )

def get_e1e2_start(x, e1_id, e2_id):
    try:
        e1_e2_start = (
            [i for i, e in enumerate(x) if e == e1_id][0],
            [i for i, e in enumerate(x) if e == e2_id][0],
        )
    except Exception as e:
        e1_e2_start = None
    return e1_e2_start

class re_dataset(Dataset):
    """
    This module used to package and mutate the inserted input to be loaded 
    and processed with Transformer-based model for relation extraction
    """
    def __init__(self, dataframe:pd.DataFrame, tokenizer, e1_id, e2_id, mutate=True, max_len=128):
        """
        The dataframe must satisfy the model requirements to contain essential info 
        to be feeded to the model
        """
        self.e1_id = e1_id
        self.e2_id = e2_id
        self.df = dataframe.copy()
        self.feature = 'sents'
        # Assert that data is valid for the down-stream task
        missed_columns = set(["sents", "org_groups", "idx"]) - set(self.df.columns)
        if len (missed_columns) > 0:
            raise AssertionError(f'data missing {missed_columns} column!!!')
        if mutate: 
            # Mutate the text to replace organization names to refer to it's group
            tqdm.pandas(desc='mutate text')
            self.df['mutated_sents'] = self.df.progress_apply(lambda x: \
                                                              mutate_sent(sent=x[self.feature],
                                                              org_groups=x['org_groups']),axis=1)
            self.feature = "mutated_sents"
        # Text tokenizer with transformer tokenizer
        logger.info("Tokenizing data...")
        tqdm.pandas(desc='tokenization')
        self.df["input"] = self.df.progress_apply(
            lambda x: tokenizer.encode(x[self.feature]), axis=1
        )
        tqdm.pandas(desc='tags positioning')
        self.df["e1_e2_start"] = self.df.progress_apply(
            lambda x: get_e1e2_start(x["input"], e1_id=self.e1_id, e2_id=self.e2_id),
            axis=1,
        )
        invalid = self.df["e1_e2_start"].isnull().sum()
        print(
            "\nInvalid rows/total: %d/%d" % (invalid, len(self.df))
        )
        if invalid > 0:
            self.df.dropna(axis=0, inplace=True)
            self.df.reset_index(drop=True, inplace=True)
    def __len__(
        self,
    ):
        return len(self.df)

    def __getitem__(self, idx):
        if "relations_id" in self.df.columns:
            return (
                torch.LongTensor(self.df.iloc[idx]["input"]),
                torch.LongTensor(self.df.iloc[idx]["e1_e2_start"]),
                torch.LongTensor([self.df.iloc[idx]["relations_id"]]),
                torch.LongTensor([self.df.iloc[idx]["idx"]])
            )

        return (
            torch.LongTensor(self.df.iloc[idx]["input"]),
            torch.LongTensor(self.df.iloc[idx]["e1_e2_start"]),
            torch.LongTensor(self.df.iloc[idx]["idx"]),
            
        )
def load_dataloaders(args, shuffle):
    model = args["model_size"]  #'bert-large-uncased' 'bert-base-uncased'
    src_dir = Path(args.get('src_dir', './'))

    tokenizer_path = src_dir / "artifacts/assets/{}_tokenizer.pkl".format(model)
    tokenizer_path.parent.mkdir(exist_ok=True, parents=True)

    if os.path.isfile(tokenizer_path):
        tokenizer = load_pickle(tokenizer_path)
        logger.info("preprocessing_funcs tokenizer from pre-trained model")
    else:
        tokenizer = Tokenizer.from_pretrained(model)
        tokenizer.add_tokens(["[E1]", "[/E1]", "[E2]", "[/E2]", "[BLANK]"])
        tokenizer_dir = os.path.dirname(tokenizer_path)
        if not os.path.isdir(tokenizer_dir):
            os.mkdir(tokenizer_dir)
        
        save_as_pickle(tokenizer_path, tokenizer)
        logger.info(
            "Saved %s tokenizer at ./artifacts/assets/%s_tokenizer.pkl"
            % (model, model)
        )

    e1_id = tokenizer.convert_tokens_to_ids("[E1]")
    e2_id = tokenizer.convert_tokens_to_ids("[E2]")
    assert e1_id != e2_id != 1
    ##load preprocessed data
    df_train = pd.read_json(args["train_data"])
    df_valid = pd.read_json(args["valid_data"])
    # determine the proper max_length values for padding
    max_len = int(df_train['sents'].apply(lambda x : len(x.split())).quantile(0.95)) + 20
    print("max_length={}".format(max_len))
    rm = load_pickle(args["relations_mapper"])
    label_binarizer = LabelBinarizer()
    label_binarizer.fit(df_train["relations_id"].to_numpy())
    
    # Calculate the normalized weight for each class
    weights  = 1- (df_train['relations_id'].value_counts() / len(df_train))
    weights = torch.tensor(weights.sort_index().values,
                           dtype=torch.float32)
    weights = weights / weights.sum()
    
    ## Validate the text structure according to the task
    train_set = re_dataset(df_train,
                           tokenizer=tokenizer,
                           e1_id=e1_id,
                           e2_id=e2_id,
                           mutate=args.get('mutate_train', True),
                           max_len=args.get('max_length', max_len))
    valid_set = re_dataset(df_valid,
                           tokenizer=tokenizer,
                           e1_id=e1_id,
                           e2_id=e2_id,
                           mutate=args.get('mutate_test', True),
                           max_len=args.get('max_length', max_len))
    train_length = len(train_set)
    valid_length = len(valid_set)
    ## Pad sequences
    PS = Pad_Sequence(
        seq_pad_value=tokenizer.pad_token_id,
        label_pad_value=tokenizer.pad_token_id,
        label2_pad_value=-1,
    )
    ##Create data loader for train and validation sets
    train_loader = DataLoader(
        train_set,
        batch_size=args["batch_size"],
        shuffle=shuffle,
        num_workers=0,
        collate_fn=PS,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=args["batch_size"],
        shuffle=False,
        num_workers=0,
        collate_fn=PS,
        pin_memory=True,
    )

    return train_loader, valid_loader, train_length, valid_length, rm, label_binarizer, weights


def word_search(word, text):
    return (
        [(ele.start(), ele.end()) for ele in re.finditer(re.escape(word), text)]
        if word is not None
        else []
    )


def Intersection(lst1, lst2):
    return set(lst1).intersection(lst2)

