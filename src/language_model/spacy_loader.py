"""This modules manage NER estimations
Main tasks: 
- predict entities
- group entities using Transformer Matcher

"""
import string
import sys
import os
from GPUtil import showUtilization as gpu_usage
from pathlib import Path
from typing import List, Tuple, Dict
import string
from tqdm import tqdm
import re
from collections import UserDict, defaultdict
from itertools import chain
import numba as nb
import numpy as np
import pandas as pd
import math
import torch
import spacy
import gc
from spacy.matcher import Matcher
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

src_dir = Path(os.path.dirname(os.path.realpath(__file__))).parent
sys.path.append(str(src_dir))
from .logs import get_logger
from utils.data_clean import clean_pipe, rm_special_char
from matcher.core import SimCSE_Matcher
from thinc.api import set_gpu_allocator


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

class Docs_Container:
    def __init__(self):
        self._docs = []
        self._spans = []
        self._ents = []

    def insert_docs(self, spacy_docs):
        # self._docs = spacy_docs
        for i, doc in enumerate(spacy_docs):
            # ents = [(ent.text, ent.label_ , ent.start, ent.end) for ent in doc.ents]
            self._docs.append(doc.text)
            self._spans.append(
            [
            {
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
            "token_start": ent.start,
            "token_end": ent.end,
            }
            for ent in doc.ents
            ]
            )
            self._ents.append(
                set(filter(None, [ent.text for ent in doc.ents if ent.label_ == "ORG"]))
            )

    @property
    def docs(self):
        return self._docs

    @property
    def spans(self):
        return self._spans

    @property
    def ents(self):
        return self._ents


class SpacyLoader:
    """This class manage spacy language models"""

    def __init__(
        self,
        lm=None,
        require_gpu: float = True,
        entity_matcher: str = "artifacts/matcher_model",
        load_matcher=False,
    ):
        icon = "\U0001F30C "
        self.logger = get_logger(f"{icon}spaCy", log_level="INFO")
        self.logger.info(f"Language model used is {lm}")
        if lm is None:
            self.nlp = spacy.blank("en")
            self.blank = self.nlp
        else:
            # Check if GPU required and accessed by spaCy
            if require_gpu and spacy.prefer_gpu():
                spacy.require_gpu()
                # req_gpu(0)
                set_gpu_allocator("pytorch")
                self.logger.info("spaCy Work On GPU")
            else:
                self.logger.info("spaCy Work On CPU")
            self.nlp = spacy.load(
                lm, disable=["tok2vec", "tagger", "parser", "lemmatizer", ""]
            )
            self.blank = spacy.blank("en")
        # Aliases matching roles
        self.matcher = Matcher(self.nlp.vocab)
        self.matcher.add("BRAC_PATTERN", [[{"TEXT": "("}]])
        self.matcher.add("QOUTE_PATTERN", [[{"TEXT": '"'}]])
        self.matcher.add("QOUTE2_PATTERN", [[{"TEXT": "“"}]])
        self.matcher.add("OR_PATTERN", [[{"LOWER": "or"}]])
        self.c_pipe = clean_pipe([rm_special_char])

        # Load SimCse Matcher to encode company names
        if load_matcher:
            self.entity_matcher = SimCSE_Matcher(entity_matcher)
        else:
            self.entity_matcher = None
    def filter_aliases(self, cand_aliases: List[Tuple]):
        """Filter list of candidate aliases"""
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
        alias2name = defaultdict(lambda: [])
        name2alias = defaultdict(lambda: [])
        for k, v in all_aliases:
            name2alias[k].append(v)
            alias2name[v].append(k)
        org_keys = Org_Group()
        counter = 0
        for target, alias in filtered_aliases:
            org_keys[target] = counter
            org_keys[alias] = counter
            counter += 1
        for name in candidate_matches:
            if org_keys.get(name, None) is None:
                references = list(
                    chain(name2alias.get(name, []), alias2name.get(name, []))
                )
                # Is refrences pre_exist?
                pre_exist = list(
                    filter(None, [org_keys.get(ref, None) for ref in references])
                )
                if len(pre_exist) > 0:
                    org_keys.add_values([name], pre_exist[0])
                    continue

                ref_group = ref2group(
                    ents_vec, references + [name], list(org_keys.keys())
                )
                if ref_group is not None:
                    group_id = org_keys.get(ref_group)
                    org_keys.add_values([name], group_id)

                else:
                    org_keys.add_values([name], counter)
                    counter += 1

        for name in set(ents) - org_keys.keys():
            org_keys[name] = counter
            counter += 1
        return org_keys.data

    def group_ents(self, docs_container) -> Tuple[Dict, List, List]:
        """Extract NER Spans from spaCy docs, with grouping similar ents
        Args
            docs (spacy.Docs) : spacy Docs from NER Pipeline
            size (int):

        Output:
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
            
            # # Define intial words
            # initials = [
            #     s.lower().replace("the", "").strip().split(" ")[0] for s in ents
            # ]
            # # Group ents with same initial
            # ents_dict = defaultdict(lambda: [])
            # for i, word in enumerate(initials):
            #     ents_dict[word].append(i)
            # candidate_matches = list(
            #     chain(
            #         *[
            #             ents[ents_dict[k]].tolist()
            #             for k in ents_dict.keys()
            #             if len(ents_dict[k]) > 1
            #         ]
            #     )
            # )
            candidate_matches = ents
            # Search alias patterns using spacy matcher
            matches = self.matcher(doc)
            candidate_alias = []
            if matches:
                # If found match
                ent2ids = {ent: f"ORG{i}" for i, ent in enumerate(ents)}
                ids2int = {k: v for v, k in ent2ids.items()}
                spare = doc.text
                for ent in ents:
                    spare = spare.replace(ent, ent2ids[ent])
                matched_rules = set([doc[st:et].text for _, st, et in matches])
                for match_str in matched_rules:
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
    def predictor(self, sents: List[str], chunck_size: int = 20000):
        """Compute spaCy pipeline with respect to huge text files.

        Args:
            sents (List[str]): List of sentences.
            chunck_size (int): Max number of examples to be allocated in memory.

        Returns:
            Tuple: A tuple of (sents, spans, group_docs, aliases_docs).

        """
        # Clean Data
        sents = self.c_pipe.fit(pd.Series(sents))
        num_of_chuncks = math.ceil(len(sents) / chunck_size)
        docs_container = Docs_Container()
        # Start predictions steps
        if num_of_chuncks > 1:
            self.logger.info(f"Start batch job for {num_of_chuncks} chunks")
        for b_index in range(num_of_chuncks):
            if num_of_chuncks > 1:
                self.logger.info(f'process chunk#{b_index+1} ...')
            # Define start and end indecies
            start = b_index * chunck_size
            end = start + chunck_size
            if b_index == num_of_chuncks - 1:
                text = sents[start:]
            else:
                text = sents[start:end]
            docs = self.nlp.pipe(text, batch_size=1000)
            docs_container.insert_docs(docs)
            gc.collect()
            torch.cuda.empty_cache()
            if (b_index+1) % 5 == 0:
                gpu_usage()
        group_docs, aliases_docs = self.group_ents(docs_container)
        return sents, docs_container.spans, group_docs, aliases_docs