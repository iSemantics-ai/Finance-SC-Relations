import re
from tqdm import tqdm
import random
from copy import copy
import random
from tqdm.notebook import tqdm_notebook
import pandas as pd
import textdistance
from num2words import num2words
import spacy
import string
from thinc.api import set_gpu_allocator, require_gpu


def word_search(word, text):
    return (
        [(ele.start(), ele.end()) for ele in re.finditer(word.lower(), text.lower())]
        if word is not None
        else []
    )


def Intersection(lst1, lst2):
    return set(lst1).intersection(lst2)


def random_string_generator(str_size, allowed_chars):
    return "".join(random.choice(allowed_chars) for x in range(str_size))


def initial_char(text):
    return [s[0] for s in text.split()]


org_numbers = [
    "ONE",
    "TWO",
    "THREE",
    "FOUR",
    "FIVE",
    "SEX",
    "SEVEN",
    "EIGHT",
    "NINE",
    "TEN",
    "ELEVEN",
    "TWELVE",
    "THIRTEEN",
    "FOURTEEN",
    "FIFTEEN",
    "EIGHTEEN",
    "NINETEEN",
    "TWENTY",
]


def ent_decay(docs):
    sents = []
    for sent in docs:
        e1_start = "[E1] "
        e1_end = " [/E1]"
        e2_start = "[E2] "
        e2_end = " [/E2]"
        e1 = sent[sent.find(e1_start) + len(e1_start) : sent.rfind(e1_end)]
        e2 = sent[sent.find(e2_start) + len(e2_start) : sent.rfind(e2_end)]
        rand_orgs = [
            num2words(i, lang="en").upper() for i in random.sample(range(51, 100), k=2)
        ]
        org1 = f"ORG {rand_orgs[0]}"
        org2 = f"ORG {rand_orgs[1]}"
        r = sent.replace(e1.strip(), org1)
        r = r.replace(e2.strip(), org2)
        sent = r
        sents.append(sent)

    return sents


def hamming_search(nlp, query: str, text: str, ent="ORG"):
    """Search similarities in entities, based on hamming distance between two
    entities, with more bias toward first and intitial charater simialrites

    Args:
    query(str): the entity to search it's similar in text
    text(str): text that query meant to search inside
    ent(str): NER tag to match with query
    """
    query = query.lower().translate(str.maketrans("", "", string.punctuation))
    doc = nlp(text)
    search = set(
        filter(
            None, [entity.text if entity.label_ == ent else None for entity in doc.ents]
        )
    )
    main_ents = list(search)
    clean_search = filter(
        None,
        [
            t.lower().translate(str.maketrans("", "", string.punctuation))
            for t in search
        ],
    )
    results = [
        textdistance.hamming.normalized_similarity(query.split()[0], s.split()[0])
        + textdistance.hamming.normalized_similarity(query, s)
        + textdistance.hamming.normalized_similarity(query, initial_char(s))
        + textdistance.hamming.normalized_similarity(initial_char(query), s)
        for s in clean_search
    ]
    if len(results) == 0:
        return None, 0.0
    if max(results) <= 0.6:
        return None, max(results)
    index = results.index(max(results))
    e = main_ents[index]
    return e, max(results)


class RelationsAnnotator:
    def __init__(self, spacy_model="en_core_web_trf"):
        try:
            set_gpu_allocator("pytorch")
            require_gpu(0)
        except:
            print("No Gpu founded for spacy_model")

        self.nlp = spacy.load(spacy_model)

    def entity_annotation(
        self,
        dataframe,
        text,
        ent1,
        ent2,
        label,
        binary=True,
        replace=False,
        random_choice=False,
    ):
        """Create entity relation extraction model by insert entity tags around
        each entity token given in dataframe rows
        """
        SPECIAL_CHAR = "[&()<>*#/\\]"
        dataframe[ent1] = dataframe[ent1].str.translate(
            str.maketrans("", "", SPECIAL_CHAR)
        )
        dataframe[ent2] = dataframe[ent2].str.translate(
            str.maketrans("", "", SPECIAL_CHAR)
        )
        dataframe[text] = dataframe[text].str.translate(
            str.maketrans("", "", SPECIAL_CHAR)
        )
        chars = string.ascii_letters
        chains = []
        sentences = []
        labels = []
        low_score_sample = []
        orginal_sents = []
        filers = []
        companies = []
        for index, row in tqdm_notebook(dataframe.iterrows(), total=dataframe.shape[0]):
            relation = row[label]
            if row["Label"] == 0 and not binary:
                relation = "other"
            sentence, e1, e2 = row[text], row[ent1], row[ent2]

            original = copy(sentence)

            res1 = word_search(e1, sentence)

            if len(res1) == 0:
                b1 = copy(e1)
                e1, score1 = hamming_search(self.nlp, e1, sentence)
                res1 = word_search(e1, sentence)
                if score1 < 1.0 and score1 > 0.6:
                    low_score_sample.append(
                        {
                            "Sentence": sentence,
                            "Query": b1,
                            "Result": e1,
                            "Score": score1,
                        }
                    )

            res2 = word_search(e2, sentence)
            if len(res2) == 0:
                b2 = copy(e2)
                e2, score2 = hamming_search(self.nlp, e2, sentence)
                res2 = word_search(e2, sentence)
                if score2 < 1.0 and score2 > 6.0:
                    low_score_sample.append(
                        {
                            "Sentence": sentence,
                            "Query": b2,
                            "Result": e2,
                            "Score": score2,
                        }
                    )

            if e1 == e2:
                continue

            if len(res1) == 0 or len(res2) == 0:
                continue

            if replace:
                org1 = "org" + random_string_generator(4, chars)
                org2 = "org" + random_string_generator(4, chars)

                r = sentence.replace(sentence[res1[0][0] : res1[0][1]].strip(), org1)
                r = r.replace(sentence[res2[0][0] : res2[0][1]].strip(), org2)
                sentence = r
                res1 = word_search(org1, r)
                res2 = word_search(org2, r)

            if random_choice:
                res1 = [random.choice(res1)]
                res2 = [random.choice(res2)]

            for r1 in res1:
                s = sentence[: r1[0]] + "[E1] " + sentence[r1[0] :]
                s = s[0 : (r1[1] + 5)] + " [/E1] " + s[(r1[1] + 5) :]
                for r in res2:
                    intersec = Intersection(
                        list(range(r[0], r[1])), list(range(r1[0], r1[1]))
                    )
                    if len(intersec) > 0:
                        continue
                    if r[0] < r1[0]:
                        r2 = r[0], r[1]
                    else:
                        r2 = r[0] + 12, r[1] + 12

                    intersec = Intersection(
                        list(range(r2[0], r2[1])), list(range(r1[0], r1[1]))
                    )
                    if len(intersec) > 0:
                        continue
                    out = s[: r2[0]] + "[E2] " + s[r2[0] :]
                    out = out[0 : (r2[1] + 5)] + " [/E2] " + out[(r2[1] + 5) :]
                    sentences.append(out)
                    labels.append(relation)
                    orginal_sents.append(original)
                    chains.append(row["Label"])
                    filers.append(row[ent1])
                    companies.append(row[ent2])

        return pd.DataFrame(
            {
                "sents": sentences,
                "relations": labels,
                "supply_chain": chains,
                "orginal_sents": orginal_sents,
                "Filer": filers,
                "Company": companies,
            }
        )
