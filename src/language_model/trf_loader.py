"""This modules manage NER estimations
Main tasks: 
- predict entities
- group entities using Transformer Matcher

"""
import string
import sys
import os
from pathlib import Path
from typing import List, Tuple, Dict
import string
import re
from collections import UserDict, defaultdict
from itertools import chain
import numba as nb
import numpy as np
import pandas as pd
import math
import torch
import spacy
from transformers import pipeline
from spacy.matcher import Matcher
os.environ["TOKENIZERS_PARALLELISM"] = "false"
src_dir = Path(os.path.dirname(os.path.realpath(__file__))).parent
sys.path.append(str(src_dir))
from .logs import get_logger
from utils.data_clean import clean_pipe, rm_special_char
from matcher.core import SimCSE_Matcher
# Icon of main module
ICON = "\U0001F30C "


class Org_Group(UserDict):
    def add_values(self, values: List, counter: int):
        for value in values:
            super().__setitem__(value, counter)


@nb.jit(nopython=True, fastmath=True)
def nb_cosine(x, y):
    xx, yy, xy = 0.0, 0.0, 0.0
    for i in range(len(x)):
        xx += x[i] * x[i]
        yy += y[i] * y[i]
        xy += x[i] * y[i]
    return xy / np.sqrt(xx * yy)


def ref2group(
    ents_vec: Dict, references: List[str], target_names, threshold: float = 0.95
):
    """Group list of entities based on the vector representation
    of each name in the two list searching for any score exceed the
    threshold.
    """
    # Lookup the target vectors using the ents_vec which contains
    # all the entities representations
    target = [ents_vec.get(ref) for ref in target_names]
    # Search for cosine similarity score that fit the threshold
    for ref in references:
        scores = [
            nb_cosine(ents_vec.get(ref).numpy(), target[x].numpy())
            for x in range(len(target))
        ]
        if len(scores) > 0:
            max_arg = np.argmax(scores)
            if scores[max_arg] > threshold:
                return target_names[max_arg]


class DocsContainer:
    """Container class for storing document information."""

    def __init__(self):
        self._docs = []
        self._spans = []
        self._ents = []

    def insert_docs(self, text, spans):
        """
        Insert documents and their corresponding spans into the container.

        @params:
        -------
        - text (List[str]): The list of document texts.
        - spans (List[List[Dict]]): The list of span dictionaries for each document.
        """
        for sent, ents in zip(text, spans):
            self._docs.append(sent.text)
            self._spans.append(
                [
                    {
                    "text": sent.text[ent["start"]:ent["end"]],
                    "label": ent['entity_group'],
                    "start": ent['start'],
                    "end": ent['end'],
                    "score": ent['score']
                    }
                    for ent in ents
                ]
            )
            self._ents.append(
                set(filter(None, [sent.text[ent["start"]:ent["end"]] \
                            for ent in ents if ent['entity_group'] == "ORG"]))
            )

    @property
    def docs(self):
        """Get the list of document texts."""
        return self._docs

    @property
    def spans(self):
        """Get the list of span dictionaries for each document."""
        return self._spans

    @property
    def ents(self):
        """Get the set of entities for each document."""
        return self._ents

class TrfLoader:
    """
    This class manages Spacy language models for NER (Named Entity Recognition).

    @params:
    -------
    - lm (str): Language model used for NER.
    - entity_matcher (str): Sentence transformer model for entity matching.
    - load_matcher (bool): Flag indicating whether to load the entity matcher.

    @attrs:
    - logger: Logger object for logging information.
    - device: Device to be used for running the models (GPU or CPU).
    - nlp: Pipeline object for NER.
    - blank: Blank Spacy model for text cleaning.
    - c_pipe: Clean pipeline for processing text.
    - entity_matcher: SimCSE Matcher for encoding company names.

    Methods:
    - __init__: Initializes the TrfLoader class.
    """

    def __init__(
        self,
        lm=None,
        entity_matcher: str = "sentence-transformers/all-MiniLM-L6-v2",
        load_matcher=False,
    ):
        """
        Initialize the TrfLoader class.
        """
        self.logger = get_logger(f"{ICON}TrfTagger", log_level="INFO")
        self.logger.info(f"Language model used is {lm}")
        self.device = 0 if torch.cuda.is_available() else -1
        self.nlp = pipeline("ner",
                            model=lm,
                            tokenizer=lm,
                            device=self.device,
                            aggregation_strategy="simple",
                            )
        self.blank = spacy.blank("en")
        self.c_pipe = clean_pipe([rm_special_char])
        if load_matcher:
            self.entity_matcher = SimCSE_Matcher(entity_matcher)
        else:
            self.entity_matcher = None

    def filter_aliases(self, cand_aliases: List[Tuple]):
        """
        Filter a list of candidate aliases based on matching criteria.

        @params:
        -------
        - cand_aliases (List[Tuple]): List of candidate aliases to filter.

        @returns:
        -------
        - List[Tuple]: Filtered list of candidate aliases.
        """        
        filter_out = []
        for target, alias in cand_aliases:
            # Clean Target to get pure intitial characters
            target_clean = (
                re.sub(f"[{string.punctuation} ]+", " ", target)
                .lower()
                .replace("the", "")
                .strip()
            )
            target_words = [
                token.text for token in self.blank(target_clean) if token.is_alpha
            ]

            alias_clean = (
                re.sub(f"[{string.punctuation} ]+", " ", alias)
                .lower()
                .replace("the", "")
                .strip()
            )
            alias_words = [
                token.text for token in self.blank(alias_clean) if token.is_alpha
            ]
            # If first two words match confirm the alias pair
            if any([word in target_words for word in alias_words]):
                filter_out.append((target, alias))
            elif len(alias_words) == 1 and len(target_words) > 1:
                target_initials = "".join([x[0] for x in target_words])
                if len(
                    re.findall(
                        f"[{alias.translate(str.maketrans('', '', string.punctuation)).lower()}]",
                        target_initials,
                    )
                ) >= 0.8 * len(alias):
                    filter_out.append((target, alias))
            else:
                # Check similarity between target and alias using entity matcher
                if self.entity_matcher.similarity(target, alias) > 0.8:
                    filter_out.append((target, alias))
        return filter_out

    @staticmethod
    def ents_grouping(
        ents: List[str],
        filtered_aliases: List[Tuple[str]],
        candidate_matches: List[List[str]],
        all_aliases:List[Tuple[str]],
        ents_vec: Dict,
    ) -> Dict:
        """
        Group the entities based on their relationships and aliases.

        @params:
        -------
        - ents: List of entities extracted from the text.
        - filtered_aliases: List of aliases for the entities.
        - candidate_matches: List of candidate matches for each entity.
        - all_aliases: List of all aliases.
        - ents_vec: Dictionary of entity vectors.

        @returns:
        -------
        - Dictionary mapping entities to their respective group IDs.
        """
        alias2name = defaultdict(lambda: [])
        name2alias = defaultdict(lambda: [])

        # Create mappings between aliases and entity names
        for k, v in all_aliases:
            name2alias[k].append(v)
            alias2name[v].append(k)

        org_keys = Org_Group()
        counter = 0

        # Assign group IDs to entities and their aliases
        for target, alias in filtered_aliases:
            org_keys[target] = counter
            org_keys[alias] = counter
            counter += 1

        # Assign group IDs to candidate matches
        for name in candidate_matches:
            if org_keys.get(name, None) is None:
                references = list(chain(name2alias.get(name, []), alias2name.get(name, [])))

                # Check if references already exist in org_keys
                pre_exist = list(filter(None, [org_keys.get(ref, None) for ref in references]))
                if len(pre_exist) > 0:
                    org_keys.add_values([name], pre_exist[0])
                    continue

                # Check if references can be grouped together
                ref_group = ref2group(ents_vec, references + [name], list(org_keys.keys()))
                if ref_group is not None:
                    group_id = org_keys.get(ref_group)
                    org_keys.add_values([name], group_id)
                else:
                    org_keys.add_values([name], counter)
                    counter += 1

        # Assign group IDs to remaining entities
        for name in set(ents) - org_keys.keys():
            org_keys[name] = counter
            counter += 1

        return org_keys.data
    
    def group_ents(self, docs_container) -> Tuple[Dict, List, List]:
        """Extract NER Spans from spaCy docs, with grouping similar ents
        @params:
        -------
        - docs (spacy.Docs) : spacy Docs from NER Pipeline
        - size (int)

        @returns:
        -------
        Tuple[
            spans dictionart with offsets,
            organization groups on the sentence level,
            the aliases founded on each sentence
            ]
        """
        aliases_docs = []  # An list to memorize the aliases founded
        initial_candidates = []  # Save the entities that share same initials

        unqiue_ents = set(chain(*docs_container.ents))
        if len(unqiue_ents) == 0:
            return [{} for _ in range(len(docs_container.docs))], [
                [] for _ in range(len(docs_container.docs))
            ]

        # Encode all the vectors with transformer encoder
        ents_vec = {
            k: v
            for k, v in zip(
                list(unqiue_ents), self.entity_matcher.encode(list(unqiue_ents), normalize_to_unit=False)
            )
        }
        # del unique_ents
        for ents, text in zip(docs_container.ents, docs_container.docs):
            doc = self.blank(text)
            ents = np.array(sorted(ents, key=len)[::-1])
            candidate_matches = ents
            candidate_alias = []
            alias_triggers = np.array( ['(',')', '"','“', ' or '])
            alias_triggers = alias_triggers[[x in doc.text for x in alias_triggers]].tolist()
            if len(alias_triggers) > 0:
                # If found match
                ent2ids = {ent: f"ORG{i}" for i, ent in enumerate(ents)}
                ids2int = {k: v for v, k in ent2ids.items()}
                spare = doc.text
                for ent in ents:
                    spare = spare.replace(ent, ent2ids[ent])
                for match_str in alias_triggers:
                    if match_str == "(":
                        # input: Massachusetts Institute of Technology ("MIT")
                        # output: [(Massachusetts Institute of Technology, MIT)]
                        found_matches = re.findall(
                            r'(ORG\d+)\s*\W*[a-zA-Z-\s]*[(]\s?\w*\W?\s?["]?(ORG\d+)["]?[)]',
                            spare,
                        )
                        for match in found_matches:
                            alias = ids2int.get(match[0])
                            company = ids2int.get(match[1])
                            if alias and company:
                                candidate_alias.append(
                                    (alias, company)
                                )
                    elif match_str == '"' or match_str == "“":
                        found_matches = re.findall(
                            r'(ORG\d+)\s*\w*["“](ORG\d+)["”]', spare
                        )
                        for match in found_matches:
                            candidate_alias.append(
                                (ids2int[match[0]], ids2int[match[1]])
                            )
                    else:
                        found_matches = re.findall(
                            r"(ORG\d)\W?\sor\s\W*(ORG\d)\W*", spare
                        )
                        for match in found_matches:
                            candidate_alias.append(
                                (ids2int[match[0]], ids2int[match[1]])
                            )
            filtered_aliases = self.filter_aliases(candidate_alias)
            initial_candidates.append(candidate_matches)
            aliases_docs.append(filtered_aliases)

        all_aliases = set(chain(*list(filter(None, aliases_docs))))
        group_docs = list(
            map(
                lambda x, y, z: self.ents_grouping(
                    x, y, z, all_aliases , ents_vec
                ),
                docs_container.ents,
                aliases_docs,
                initial_candidates,
            )
        )
        return group_docs, aliases_docs

    def predictor(self, sents: List[str], chunck_size: int = 10000):
        """Predict entities and spans from a list of sentences.

        @params:
        -------
        - sents (List[str]): List of sentences to process.
        - chunck_size (int): Maximum number of examples to allocate in memory.

        @returns:
        --------
        - Tuple[List[str], List[Dict], Dict, List]
            - Cleaned sentences.
            - Spans of the extracted entities.
            - Grouped entities.
            - Detected aliases.
        """
        # Clean Data
        sents = self.c_pipe.fit(pd.Series(sents)).tolist()
        num_of_chunks = math.ceil(len(sents) / chunck_size)
        docs_container = DocsContainer()

        # Start predictions steps
        if num_of_chunks > 1:
            self.logger.info(f"Start batch job for {num_of_chunks} chunks")
        for b_index in range(num_of_chunks):
            if num_of_chunks > 1:
                self.logger.info(f'process chunk#{b_index+1} ...')
            # Define start and end indices
            start = b_index * chunck_size
            end = start + chunck_size
            if b_index == num_of_chunks - 1:
                text = sents[start:]
            else:
                text = sents[start:end]

            ents = self.nlp(text, batch_size=32, num_workers=1)

            docs_container.insert_docs(self.blank.pipe(text), ents)
        group_docs, aliases_docs = self.group_ents(docs_container)

        return sents, docs_container.spans, group_docs, aliases_docs

