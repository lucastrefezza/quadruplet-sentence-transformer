import math
import os
import re
from typing import Optional, Callable, List, final

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import nltk
from transformers import MarianMTModel, MarianTokenizer

import tqdm
import json
import traceback

from dataset.constants import CHUNK_DIM, N_EXAMPLES, N_PART_EXAMPLES, SIMILARITY_THRESHOLD, ADAPTIVE_CROP, INSTANCES, \
    ID, REFERENCE_EXAMPLE, POS_EXAMPLES, PART_POS_EXAMPLES, DATASET_NAME, ANNOTATION_FILE, CHUNK_NAME
from dataset.partially_positive_examples_selection import get_part_pos_examples
from dataset.positive_examples_selection import select_positive_examples

MIN_RESPONSE_NUM: final = 5


class CocoCaptionsOnly(datasets.CocoCaptions):
    def __init__(
            self,
            root: str,
            ann_file: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
            dataset_name: str = "CoCoCaptionDataset"
    ) -> None:
        super().__init__(root=root, annFile=ann_file, transforms=transforms,
                         transform=transform, target_transform=target_transform)
        self.__ann_file = ann_file
        self.__dataset_name = dataset_name

    @property
    def ann_file(self) -> str:
        return self.__ann_file

    @property
    def dataset_name(self) -> str:
        return self.__dataset_name

    def _load_image(self, id: int) -> Image.Image:
        # Return mock empty image because we need caption only
        return torch.zeros(3, 427, 640)


def mock_llm_response(caption: str, n_responses: int = MIN_RESPONSE_NUM) -> str:
    return "1. Woman wearing a hat;  2. Woman taking a photo;  3. Woman riding " \
           "a bike;  4. Parking lot surrounded by trees;  5. Woman standing in " \
           "the parking lot."


def parse_llm_response(llm_response: str, min_response_num: int = MIN_RESPONSE_NUM) -> List[str]:
    # Split the response on the numbers discarding the first, empty one
    responses = re.split(string=llm_response, pattern=r"[0-9]\.")[1:]

    # Check if the number of created sub-phrases is correct
    assert len(responses) >= min_response_num

    # Format the sub-phrases correctly, removing semi-column, spaces, ...
    for i, response in enumerate(responses):
        responses[i] = response.strip().lower().replace(";", "").replace(".", "")

    return responses


def get_chunk_idx(idx: int, chunk_dim: int = CHUNK_DIM) -> tuple[int, int]:
    chunk_idx = math.floor(idx / chunk_dim)  # chunk index
    local_idx = idx % chunk_dim  # local index of the instance inside the chunk
    return chunk_idx, local_idx


def create_dataset_chunk(dataset: CocoCaptionsOnly,
                         start_idx: int = 0,
                         chunk_dim: int = CHUNK_DIM,
                         n_pos_examples: int = N_EXAMPLES,
                         n_part_pos_examples: int = N_PART_EXAMPLES,
                         sim_threshold: float = SIMILARITY_THRESHOLD,
                         augment: bool = True) -> dict:
    end_idx: int = min(start_idx + chunk_dim, len(dataset))  # end index

    dataset_chunk = {
        DATASET_NAME: dataset.dataset_name,
        ANNOTATION_FILE: dataset.ann_file,
        INSTANCES: []
    }
    # For each index in the chunk
    for idx in range(start_idx, end_idx):
        # Get the corresponding captions
        _, captions = dataset[idx]

        # Get the reference and positive examples
        positive_examples, caption, _ = select_positive_examples(
            captions=captions,
            threshold=sim_threshold,
            n_examples=n_pos_examples,
            augment=augment,
            return_similarities=True,
            max_attempts=n_pos_examples
        )

        # Get the partially positive examples
        part_pos_examples = get_part_pos_examples(
            caption=caption,
            n_part_pos_examples=n_part_pos_examples,
            algorithm_type=ADAPTIVE_CROP
        )

        # Add dataset entry to chuck
        dataset_entry = {
            ID: idx,
            REFERENCE_EXAMPLE: caption,
            POS_EXAMPLES: positive_examples,
            PART_POS_EXAMPLES: part_pos_examples
        }
        dataset_chunk[INSTANCES].append(dataset_entry)

    return dataset_chunk


def create_dataset(root: str,
                   dataset: CocoCaptionsOnly,
                   start_chunk: int = 0,
                   max_chunk: Optional[int] = None,
                   chunk_dim: int = CHUNK_DIM,
                   n_pos_examples: int = N_EXAMPLES,
                   n_part_pos_examples: int = N_PART_EXAMPLES,
                   sim_threshold: float = SIMILARITY_THRESHOLD,
                   augment: bool = True,
                   log_tqdm: bool = False) -> int:
    # Create dataset directory if it doesn't exist
    root = os.path.join(root, dataset.dataset_name)
    os.makedirs(root, exist_ok=True)

    # Calculate max chunk
    chunk_num = math.ceil(len(dataset) / chunk_dim)
    chunk_num = chunk_num if max_chunk is None else min(chunk_num, max_chunk)

    # For each chunk
    iterable = range(start_chunk, chunk_num)
    iterable = tqdm.tqdm(iterable, "Creating chunks...") if log_tqdm else iterable
    last_chunk_idx = -1
    for chunk_idx in iterable:

        try:
            # Create the chunk
            dataset_chunk = create_dataset_chunk(
                dataset=dataset,
                start_idx=chunk_idx * chunk_dim,
                n_pos_examples=n_pos_examples,
                n_part_pos_examples=n_part_pos_examples,
                sim_threshold=sim_threshold,
                augment=augment
            )

            # Store the chunk
            chunk_filename = os.path.join(root, f"{CHUNK_NAME}_{chunk_idx}.json")
            with open(chunk_filename, "w") as fp:
                json.dump(dataset_chunk, fp)
            last_chunk_idx = chunk_idx

        except Exception as e:
            print(f"Chunk {chunk_idx} creation failed due to error: {e}.")
            print(f"Traceback: {traceback.format_exc()}")
            return chunk_idx - 1

    return last_chunk_idx

# skip this if already created
# create_dataset("./data/coco/train", dataset=cap_train, max_chunk=None, log_tqdm=True)
