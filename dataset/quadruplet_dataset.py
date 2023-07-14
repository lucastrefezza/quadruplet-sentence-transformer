import os
import math
import random
from typing import Callable, List, Union, final, Dict
import json
import time
import heapq
import torch
from torch.utils.data import Dataset
import numpy as np
from sortedcollections import ItemSortedDict, ValueSortedDict
from dataset.constants import INSTANCES, CHUNK_DIM, REFERENCE_EXAMPLE, POS_EXAMPLES, PART_POS_EXAMPLES, NEG_EXAMPLES
from dataset.positive_examples_selection import pop_random_caption, compute_cosine_scores


CACHE_SIZE_DEFAULT: final = 30
HARD_CONTRASTIVE_TRAIN: final = 1
HARD_CONTRASTIVE_TEST: final = 0
RANDOM: final = -1
NEG_EXAMPLE_SIM_TRESHOLD: final = 0.2
MAX_ATTEMPTS_NEGATIVE_SAMPLING: final = 3
_CACHE_LOG: final = False


def get_chunk_idx(idx: int, chunk_dim: int = CHUNK_DIM) -> tuple[int, int]:
    chunk_idx = math.floor(idx / chunk_dim)  # chunk index
    local_idx = idx % chunk_dim  # local index of the instance inside the chunk
    return chunk_idx, local_idx


def hard_contrastive_sampling(population: List,
                              scores: Union[np.ndarray, List[float]],
                              k: int,
                              max_mode: bool = True) -> List:
    if max_mode:
        key = lambda score: score[1]  # should select maximum
    else:
        key = lambda score: -score[1]  # should select minimum

    # Select the top-k largest scores
    scores = [(i, score) for i, score in enumerate(scores)]
    scores = heapq.nlargest(k, scores, key=key)

    # Select the corresponding objects
    selected = [population[i] for i, score in scores]

    return selected


def _choose_examples(choices: List[str], n: int) -> List[str]:
    examples = set()
    for _ in range(0, n):
        example = pop_random_caption(captions=choices, forbidden=examples)
        examples.add(example)

    return list(examples)


class QuadrupletDataset(Dataset):
    def __init__(self,
                 root: str,
                 *dataset_chunks: str,
                 hard_contrastive_mode: int = RANDOM,
                 n_pos: int = 1,
                 n_part_pos: int = 1,
                 n_neg: int = 1,
                 cache_size: int = CACHE_SIZE_DEFAULT,
                 transform: Callable = None):
        if cache_size <= 0:
            raise ValueError(f"Cache size must be > 0. {cache_size} given.")
        if len(dataset_chunks) == 0:
            raise ValueError(f"At least 1 chunk must be given.")
        if not -1 <= hard_contrastive_mode <= 1:
            raise ValueError(f"hard_contrastive_mode must be either {RANDOM}, "
                             f"{HARD_CONTRASTIVE_TRAIN} or {HARD_CONTRASTIVE_TEST}, "
                             f"respectively meaning no hard contrastive, hard "
                             f"contrastive training mode and hard contrastive test "
                             f"mode. {hard_contrastive_mode} given.")
        if n_pos <= 0:
            raise ValueError(f"n_pos must be > 0, {n_pos} given.")
        if n_part_pos <= 0:
            raise ValueError(f"n_part_pos must be > 0, {n_part_pos} given.")
        if n_neg <= 0:
            raise ValueError(f"n_neg must be > 0, {n_neg} given.")

        self.__root: str = root
        self.__hard_contrastive_mode: int = hard_contrastive_mode
        self.__n_pos: int = n_pos
        self.__n_part_pos: int = n_part_pos
        self.__n_neg: int = n_neg
        self._transform: Callable = transform
        self.__dataset_chunks: List[str] = [*dataset_chunks]

        # Ordered set that stores couples (chunk, last_access_timestamp), ordered by
        # last_acces_timestamp
        self.__chunk_cache = ValueSortedDict(lambda value: value[1])
        self.__cache_size: int = cache_size

        # Compute chunk dim (assuming all chunks have the same dim, except the last)
        chunk_path = os.path.join(root, self.__dataset_chunks[0])
        with open(chunk_path) as fp:
            chunk = json.load(fp)
            self.__chunk_dim = len(chunk[INSTANCES])

        # Compute last chunk dim
        chunk_path = os.path.join(root, self.__dataset_chunks[-1])
        with open(chunk_path) as fp:
            chunk = json.load(fp)
            self.__last_chunk_dim = len(chunk[INSTANCES])

    @property
    def root(self) -> str:
        return self.__root

    @property
    def hard_contrastive_mode(self) -> int:
        return self.__hard_contrastive_mode

    @hard_contrastive_mode.setter
    def hard_contrastive_mode(self, hard_contrastive_mode: int):
        if not -1 <= hard_contrastive_mode <= 1:
            raise ValueError(f"hard_contrastive_mode must be either {RANDOM}, "
                             f"{HARD_CONTRASTIVE_TRAIN} or {HARD_CONTRASTIVE_TEST}, "
                             f"respectively meaning no hard contrastive, hard "
                             f"contrastive training mode and hard contrastive test "
                             f"mode. {hard_contrastive_mode} given.")
        self.__hard_contrastive_mode = hard_contrastive_mode

    @property
    def n_pos(self) -> int:
        return self.__n_pos

    @n_pos.setter
    def n_pos(self, n_pos: int):
        if n_pos <= 0:
            raise ValueError(f"n_pos must be > 0, {n_pos} given.")
        self.__n_pos = n_pos

    @property
    def n_part_pos(self) -> int:
        return self.__n_part_pos

    @n_part_pos.setter
    def n_part_pos(self, n_part_pos: int):
        if n_part_pos <= 0:
            raise ValueError(f"n_part_pos must be > 0, {n_part_pos} given.")
        self.__n_part_pos = n_part_pos

    @property
    def n_neg(self) -> int:
        return self.__n_neg

    @n_neg.setter
    def n_neg(self, n_neg: int):
        if n_neg <= 0:
            raise ValueError(f"n_neg must be > 0, {n_neg} given.")
        self.__n_neg = n_neg

    @property
    def dataset_chunks(self) -> List[str]:
        return self.__dataset_chunks

    @property
    def n_chunks(self) -> int:
        return len(self.dataset_chunks)

    @property
    def chunk_dim(self) -> int:
        return self.__chunk_dim

    @property
    def last_chunk_dim(self) -> int:
        return self.__last_chunk_dim

    @property
    def cache_size(self) -> int:
        return self.__cache_size

    #def remove_transforms(self):
    #    self._transform = None

    def __len__(self) -> int:
        return (self.n_chunks - 1) * self.chunk_dim + self.last_chunk_dim

    def choose_negative(self,
                        chunk_idx: int,
                        instance: Dict,
                        local_idx: int,
                        n: int,
                        hard_contrastive_mode: int = RANDOM) -> List[str]:
        if not -1 <= hard_contrastive_mode <= 1:
            raise ValueError(f"hard_contrastive_mode must be either {RANDOM}, "
                             f"{HARD_CONTRASTIVE_TRAIN} or {HARD_CONTRASTIVE_TEST}, "
                             f"respectively meaning no hard contrastive, hard "
                             f"contrastive training mode and hard contrastive test "
                             f"mode. {hard_contrastive_mode} given.")

        # Select a random chunk to sample from
        selected_chunk_idx = random.randint(0, self.n_chunks - 1)
        chunk, _, _ = self.__get_chunk(selected_chunk_idx * self.chunk_dim)
        instance_indexes = list(range(len(chunk[INSTANCES])))

        # If the selected chunk is the one the current instance is coming, remove it
        if selected_chunk_idx == chunk_idx:
            instance_indexes[local_idx], instance_indexes[-1] = instance_indexes[-1], instance_indexes[
                local_idx]  # exchange local_idx and last
            instance_indexes.pop()  # remove the last (now local_idx)

        diff = -1
        attempt_count = 0
        selected_captions_total = []
        # Continue to sample caption until we have n
        while diff < 0 and attempt_count < MAX_ATTEMPTS_NEGATIVE_SAMPLING:

            # Choose at most n*5 instances to select the captions from
            selected_instance_indexes = random.sample(
                population=instance_indexes,
                k=min(math.ceil(n * 5), len(instance_indexes))
            )

            selected_captions = []
            for i in selected_instance_indexes:
                caption = random.choice(chunk[INSTANCES][i][POS_EXAMPLES])
                selected_captions.append(caption)
            selected_captions = np.array(selected_captions)

            # Discard captions too similar to the reference instance, according to SBERT
            reference_caption = instance[REFERENCE_EXAMPLE]
            cos_scores = compute_cosine_scores(
                caption=reference_caption,
                captions=selected_captions
            ).detach().cpu().numpy()
            selected_captions = selected_captions[cos_scores <= NEG_EXAMPLE_SIM_TRESHOLD]
            cos_scores = cos_scores[cos_scores <= NEG_EXAMPLE_SIM_TRESHOLD]
            selected_captions = selected_captions.tolist()
            selected_captions_total.extend(selected_captions)

            # Among the chosen <= n*2, select n negative, either randomly or with HCS
            attempt_count += 1
            diff = len(selected_captions_total) - n

        if diff > 0:
            if hard_contrastive_mode == HARD_CONTRASTIVE_TRAIN:
                # Select the hardest
                selected_captions_total = hard_contrastive_sampling(
                    population=selected_captions_total,
                    scores=cos_scores,
                    max_mode=True,
                    k=n
                )
            else:
                # Random sampling
                selected_captions_total = random.sample(selected_captions_total, k=n)
            """
      elif hard_contrastive_mode == HARD_CONTRASTIVE_TEST:
        # Select the easiest, or maybe random
        selected_captions = hard_constrastive_sampling(
            population=selected_captions,
            scores=score,
            max_mode=True,
            k=n
        )
      """

        # Replicate the selected until we have n negative examples
        if diff < 0:
            replicated = random.choices(selected_captions_total, k=abs(diff))
            selected_captions_total.extend(replicated)

        return selected_captions_total

    def example_sampling(self,
                         instance: Dict,
                         local_idx: int,
                         chunk_idx: int,
                         n_pos: int = 1,
                         n_part_pos: int = 1,
                         n_neg: int = 1,
                         hard_contrastive_mode: int = RANDOM) -> Dict:
        if n_pos <= 0:
            raise ValueError(f"n_pos must be > 0, {n_pos} given.")
        if n_part_pos <= 0:
            raise ValueError(f"n_part_pos must be > 0, {n_part_pos} given.")
        if n_neg <= 0:
            raise ValueError(f"n_neg must be > 0, {n_neg} given.")
        if not -1 <= hard_contrastive_mode <= 1:
            raise ValueError(f"hard_contrastive_mode must be either {RANDOM}, "
                             f"{HARD_CONTRASTIVE_TRAIN} or {HARD_CONTRASTIVE_TEST}, "
                             f"respectively meaning no hard contrastive, hard "
                             f"contrastive training mode and hard contrastive test "
                             f"mode. {hard_contrastive_mode} given.")

        # Positive examples
        choices = list(instance[POS_EXAMPLES])
        pos_examples: list[str] = _choose_examples(choices=choices, n=n_pos)
        if len(pos_examples) == 1:
            pos_examples = pos_examples[0]

        # Partially positive examples
        choices = list(instance[PART_POS_EXAMPLES])
        part_pos_examples: list[str] = _choose_examples(choices=choices,
                                                        n=n_part_pos)
        if len(part_pos_examples) == 1:
            part_pos_examples = part_pos_examples[0]

        # Negative examples, choose candidate negative examples from random chunk
        neg_examples = self.choose_negative(
            chunk_idx=chunk_idx,
            instance=instance,
            local_idx=local_idx,
            n=n_neg,
            hard_contrastive_mode=hard_contrastive_mode
        )

        if len(neg_examples) == 1:
            neg_examples = neg_examples[0]

        return_instance = instance.copy()
        return_instance[POS_EXAMPLES] = pos_examples
        return_instance[PART_POS_EXAMPLES] = part_pos_examples
        return_instance[NEG_EXAMPLES] = neg_examples

        return return_instance

    def __get_chunk(self, idx) -> tuple[Dict, int, int]:
        """
    Gets the chunk in which the required item resides, handling the caching.

    :param idx: the index of the required item.

    :return the chunk in which the required item resides and the index of the
      item inside the chunk.
    """
        # Get chunk index
        chunk_idx, local_idx = get_chunk_idx(idx, chunk_dim=self.chunk_dim)

        if _CACHE_LOG:
            print(f"Accessing chunk {chunk_idx}")

        # If chunk is cached
        if chunk_idx in self.__chunk_cache:

            # Get it from the cache
            chunk = self.__chunk_cache[chunk_idx][0]

            # Refresh its value, putting it at the top
            self.__chunk_cache[chunk_idx] = [chunk, time.time()]  # O(logn)

        # If chunk is not cached
        else:
            # Read it from the disk and add it to the cache
            chunk_path = os.path.join(self.root, self.__dataset_chunks[chunk_idx])
            with open(chunk_path, "r") as fp:
                chunk = json.load(fp)

            # Check if cache is full, and in that case, delete the oldest item
            if len(self.__chunk_cache) == self.__cache_size:

                if _CACHE_LOG:
                    print([(k, self.__chunk_cache[k][1]) for k in self.__chunk_cache.keys()])

                item = self.__chunk_cache.popitem(index=0)  # O(logn)

                if _CACHE_LOG:
                    print(f"Deleted chunk: {item[0]} with value {item[1][1]}")

            # Add the read chunk to the cache
            self.__chunk_cache[chunk_idx] = [chunk, time.time()]  # O(logn)

        return chunk, local_idx, chunk_idx

    def __getitem__(self, idx) -> Union[Dict[str, str], List[Dict[str, str]]]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if not isinstance(idx, list):
            idx = [idx]

        instances = []
        # For each index
        for i in idx:
            # Get the corresponding chunk and instance
            chunk, local_idx, chunk_idx = self.__get_chunk(i)
            instance = chunk[INSTANCES][local_idx]

            # Hard contrastive/random sample the neg., sample pos. and part. pos.
            instance = self.example_sampling(
                instance=instance,
                local_idx=local_idx,
                chunk_idx=chunk_idx,
                hard_contrastive_mode=self.hard_contrastive_mode,
                n_pos=self.n_pos,
                n_part_pos=self.n_part_pos,
                n_neg=self.n_neg
            )

            # Apply transform
            if self._transform is not None:
                instance = self._transform(instance)

            # Add to the instances
            instances.append(instance)

        return instances if len(instances) > 1 else instances[0]
