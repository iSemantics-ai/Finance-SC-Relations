import logging
from tqdm.auto import tqdm
import math
import numpy as np
import pandas as pd
from numpy import ndarray
import torch
from torch import Tensor, device
import transformers
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from typing import List, Dict, Tuple, Type, Union

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class SimCSE_Matcher(object):
    """
    A class for embedding entit names, calculating similarities, and retriving entities by SimCSE.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str = None,
        num_cells: int = 100,
        num_cells_in_search: int = 10,
        pooler=None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.index = None
        self.is_faiss_index = False
        self.num_cells = num_cells
        self.num_cells_in_search = num_cells_in_search

        if pooler is not None:
            self.pooler = pooler
        elif "unsup" in model_name_or_path:
            logger.info(
                "Use `cls_before_pooler` for unsupervised models. If you want to use other pooling policy, specify `pooler` argument."
            )
            self.pooler = "cls_before_pooler"
        else:
            self.pooler = "cls"

    def encode(
        self,
        sentence: Union[str, List[str]],
        device: str = None,
        return_numpy: bool = False,
        normalize_to_unit: bool = True,
        keepdim: bool = False,
        batch_size: int = 64,
        max_length: int = 128,
    ) -> Union[ndarray, Tensor]:
        target_device = self.device if device is None else device
        self.model = self.model.to(target_device)

        single_sentence = False
        if isinstance(sentence, str):
            sentence = [sentence]
            single_sentence = True

        embedding_list = []
        with torch.no_grad():
            total_batch = len(sentence) // batch_size + (
                1 if len(sentence) % batch_size > 0 else 0
            )
            progress_bar = False if total_batch > 100 else True
            for batch_id in tqdm(
                range(total_batch), total=total_batch, disable=progress_bar
            ):
                inputs = self.tokenizer(
                    sentence[batch_id * batch_size : (batch_id + 1) * batch_size],
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                inputs = {k: v.to(target_device) for k, v in inputs.items()}
                outputs = self.model(**inputs, return_dict=True)
                if self.pooler == "cls":
                    embeddings = outputs.pooler_output
                elif self.pooler == "cls_before_pooler":
                    embeddings = outputs.last_hidden_state[:, 0]
                else:
                    raise NotImplementedError
                if normalize_to_unit:
                    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
                embedding_list.append(embeddings.cpu())
        embeddings = torch.cat(embedding_list, 0)

        if single_sentence and not keepdim:
            embeddings = embeddings[0]

        if return_numpy and not isinstance(embeddings, ndarray):
            return embeddings.numpy()
        return embeddings

    def match_data(
        self,
        source_data: List[str],
        query_data: List[str],
        b_size: int = 500_000,
        top_k: int = 5,
        threshold: float = 0.7,
        index_memory="cpu",
    ):
        """
        Match entities between source and query, using simcse encoder and faiss to map quick matches

        @params
        -------
        source_data(List[str]): List of names to search inside
        query_data(str): List of required named to detect it's matches
        b_size(int): max number of indecies to allocate inside the GPU at once(default=500K)
        top_k(int): the number of candidate matches for each query(default=5)
        threshold(float): the similarity threshold for matches(default=0.7)

        @return
        pd.DataFrame for queries and matches in the source
        """
        if not isinstance(source_data, list) and isinstance(query_data, list):
            raise ValueError("source_data and query_data must be list of str!!!")

        # Calc number of branches
        b_number = math.ceil(len(source_data) / b_size)
        # Initialize the search candidates dictionary
        candidates = {k: [] for k in query_data}
        # Loop over all branches
        logger.info(f"Match steps: {b_number}")
        for b_index in range(b_number):
            logger.info(f"Step number #{b_index+1}")
            # Define start and end indecies
            start = b_index * b_size
            end = start + b_size
            if b_index == b_number - 1:
                data = source_data[start:]
            else:
                data = source_data[start:end]

            # Build Faiss idx in GPU if exist
            self.build_index(data, index_memory=index_memory)
            # Search in the faiss matrix for most simialr vectors to the query
            results = self.search(query_data, top_k=top_k, threshold=threshold)
            # Fill the dict of canditates
            for i, k in enumerate(candidates.keys()):
                candidates[k] += results[i]

        # Create DataFrame for the results
        pred = []
        score = []
        matches = []
        for k, v in candidates.items():
            # If no candidates return None
            if len(v) == 0:
                pred.append([None])
                score.append(None)
                matches.append([])
                continue
            # Set each query with it's matches as one row in the frame
            names = np.array([r[0] for r in v])
            scores = np.array([r[1] for r in v])
            sort = np.argsort(scores, axis=0)[::-1]

            names = names[sort]
            scores = scores[sort]

            pred.append(names[0])
            score.append(scores[0])
            matches.append(list(zip(names[:top_k], scores[:top_k])))
        d_results = pd.DataFrame(
            {"query": query_data, "pred": pred, "score": score, "matches": matches}
        )
        return d_results

    def similarity(
        self,
        queries: Union[str, List[str]],
        keys: Union[str, List[str], ndarray],
        device: str = None,
    ) -> Union[float, ndarray]:
        query_vecs = self.encode(
            queries, device=device, return_numpy=True
        )  # suppose N queries

        if not isinstance(keys, ndarray):
            key_vecs = self.encode(
                keys, device=device, return_numpy=True
            )  # suppose M keys
        else:
            key_vecs = keys

        # check whether N == 1 or M == 1
        single_query, single_key = len(query_vecs.shape) == 1, len(key_vecs.shape) == 1
        if single_query:
            query_vecs = query_vecs.reshape(1, -1)
        if single_key:
            key_vecs = key_vecs.reshape(1, -1)

        # returns an N*M similarity array
        similarities = cosine_similarity(query_vecs, key_vecs)

        if single_query:
            similarities = similarities[0]
            if single_key:
                similarities = float(similarities[0])

        return similarities

    def build_index(
        self,
        sentences_or_file_path: Union[str, List[str]],
        use_faiss: bool = None,
        faiss_fast: bool = False,
        device: str = None,
        batch_size: int = 64,
        return_emb:bool = False,
        index_memory: str = "cuda",
    ):
        if use_faiss is None or use_faiss:
            try:
                import faiss

                assert hasattr(faiss, "IndexFlatIP")
                use_faiss = True
            except:
                logger.warning(
                    "Fail to import faiss. If you want to use faiss, install faiss through PyPI. Now the program continues with brute force search."
                )
                use_faiss = False

        # if the input sentence is a string, we assume it's the path of file that stores various sentences
        if isinstance(sentences_or_file_path, str):
            sentences = []
            with open(sentences_or_file_path, "r") as f:
                # logging.info("Loading sentences from %s ..." % (sentences_or_file_path))
                for line in tqdm(f):
                    sentences.append(line.rstrip())
            sentences_or_file_path = sentences

        # logger.info("Encoding embeddings for sentences...")
        embeddings = self.encode(
            sentences_or_file_path,
            device=device,
            batch_size=batch_size,
            normalize_to_unit=True,
            return_numpy=True,
        )

        # logger.info("Building index...")
        self.index = {"sentences": sentences_or_file_path}

        if use_faiss:
            quantizer = faiss.IndexFlatIP(embeddings.shape[1])
            if faiss_fast:
                index = faiss.IndexIVFFlat(
                    quantizer,
                    embeddings.shape[1],
                    min(self.num_cells, len(sentences_or_file_path)),
                )
            else:
                index = quantizer

            if self.device == "cuda" and device != "cpu" and index_memory == "cuda":
                if hasattr(faiss, "StandardGpuResources"):
                    # logger.info("Use GPU-version faiss")
                    res = faiss.StandardGpuResources()
                    res.setTempMemory(20 * 1024 * 1024 * 1024)
                    index = faiss.index_cpu_to_gpu(res, 0, index)
                # else:
                #     logger.info(
                #         "StandardGpuResources not found in faiss, Use CPU-version faiss"
                #     )
            # else:
            #     logger.info("Use CPU-version faiss")

            if faiss_fast:
                index.train(embeddings.astype(np.float32))
            index.add(embeddings.astype(np.float32))
            index.nprobe = min(self.num_cells_in_search, len(sentences_or_file_path))
            self.is_faiss_index = True
        else:
            index = embeddings
            self.is_faiss_index = False
        self.index["index"] = index
        # logger.info("Finished")
        if return_emb:
            return embeddings

    def add_to_index(
        self,
        sentences_or_file_path: Union[str, List[str]],
        device: str = None,
        batch_size: int = 64,
    ):
        # if the input sentence is a string, we assume it's the path of file that stores various sentences
        if isinstance(sentences_or_file_path, str):
            sentences = []
            with open(sentences_or_file_path, "r") as f:
                # logging.info("Loading sentences from %s ..." % (sentences_or_file_path))
                for line in tqdm(f):
                    sentences.append(line.rstrip())
            sentences_or_file_path = sentences

        # logger.info("Encoding embeddings for sentences...")
        embeddings = self.encode(
            sentences_or_file_path,
            device=device,
            batch_size=batch_size,
            normalize_to_unit=True,
            return_numpy=True,
        )

        if self.is_faiss_index:
            self.index["index"].add(embeddings.astype(np.float32))
        else:
            self.index["index"] = np.concatenate((self.index["index"], embeddings))
        self.index["sentences"] += sentences_or_file_path
        # logger.info("Finished")

    def search(
        self,
        queries: Union[str, List[str], dict],
        device: str = None,
        threshold: float = 0.6,
        top_k: int = 5,
    ) -> Union[List[Tuple[str, float]], List[List[Tuple[str, float]]]]:
        if not self.is_faiss_index:
            if isinstance(queries, list):
                combined_results = []
                for query in queries:
                    results = self.search(query, device, threshold, top_k)
                    combined_results.append(results)
                return combined_results

            similarities = self.similarity(queries, self.index["index"]).tolist()
            id_and_score = []
            for i, s in enumerate(similarities):
                if s >= threshold:
                    id_and_score.append((i, s))
            id_and_score = sorted(id_and_score, key=lambda x: x[1], reverse=True)[
                :top_k
            ]
            results = [
                (self.index["sentences"][idx], score) for idx, score in id_and_score
            ]
            return results
        else:
            if isinstance(queries, tuple):
                queries ,query_vecs = queries
            else:
                query_vecs = self.encode(
                    queries,
                    device=device,
                    normalize_to_unit=True,
                    keepdim=True,
                    return_numpy=True,
                )

            distance, idx = self.index["index"].search(
                query_vecs.astype(np.float32), top_k
            )

            def pack_single_result(dist, idx):
                results = [
                    (self.index["sentences"][i], s)
                    for i, s in zip(idx, dist)
                    if s >= threshold
                ]
                return results

            if isinstance(queries, list):
                combined_results = []
                for i in range(len(queries)):
                    results = pack_single_result(distance[i], idx[i])
                    combined_results.append(results)
                return combined_results
            else:
                return pack_single_result(distance[0], idx[0])
