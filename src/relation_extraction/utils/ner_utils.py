"""This modules manage NER estimations
"""
import types
import spacy
import logging


logging.basicConfig(
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)
logger = logging.getLogger("__file__")


class SpacyLoader:

    """This class manage spacy language models"""

    def __init__(self, lm=None, require_gpu: float = False):
        logger.info(f"Language model used is {lm}")
        if lm is None:
            self.nlp = None
        else:
            ## Check if GPU required and accessed by spaCy
            if require_gpu and spacy.prefer_gpu():
                spacy.require_gpu()
                logger.info("spaCy Require GPU")
            else:
                logger.info("spaCy Work on CPU")
            self.nlp = spacy.load(lm)

    @staticmethod
    def unique_ents(doc, ent="ORG"):
        if isinstance(doc, spacy.tokens.doc.Doc):
            return set(
                filter(
                    None,
                    [
                        entity.text if entity.label_ == ent else None
                        for entity in doc.ents
                    ],
                )
            )
        elif isinstance(doc, list):
            if len(doc) > 0:
                return set(
                    filter(
                        None,
                        [
                            entity.get("text", None) if entity["label"] == ent else None
                            for entity in doc
                        ],
                    )
                )
