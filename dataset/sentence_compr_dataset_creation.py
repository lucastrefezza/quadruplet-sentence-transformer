import os
from typing import final, Optional, Union, Iterable, List, Dict, Tuple
import traceback
import json
import math
from tqdm import tqdm
import numpy as np
import nlpaug.augmenter.word as naw
from dataset.constants import ID, REFERENCE_EXAMPLE, POS_EXAMPLES, PART_POS_EXAMPLES, INSTANCES, NO_REPLACE_WORDS, \
    ADAPTIVE_CROP_AUGMENT, CHUNK_DIM, N_EXAMPLES, N_PART_EXAMPLES, ANNOTATION_FILE, CHUNK_NAME, DATASET_NAME
from dataset.partially_positive_examples_selection import REPLACE_WORDNET, BACKTRANSL, REPLACE_BERT, \
    get_part_pos_examples
from dataset.positive_examples_selection import back_translation


TRAIN_SENT_COMPR: final = "train"
TEST_SENT_COMPR: final = "validation"
GRAPH: final = "graph"
COMPRESSION: final = "compression"
COMPR_RATIO: final = "compression_ratio"
SENTENCE: final = "sentence"
TEXT: final = "text"
COMPRESSION_RATIO_THRESHOLD: final = 0.6
INSERT_BERT: final = "insert_bert"
MAX_INSERT_BERT: final = 4
MAX_REPLACE_BERT: final = 3
DEFAULT_AUGS: final = frozenset([REPLACE_WORDNET, BACKTRANSL, INSERT_BERT])


def generate_variations(sentence: Union[str, List[str]],
                        n: int,
                        augs: Iterable[str] = DEFAULT_AUGS) -> List[str]:
    sentences = list(np.repeat(sentence, n))

    if INSERT_BERT in augs:
        aug = naw.ContextualWordEmbsAug(
            model_path='roberta-base',
            action="insert",
            aug_min=0,
            aug_max=MAX_INSERT_BERT
        )
        sentences = aug.augment(sentences)
    if REPLACE_BERT in augs:
        aug = naw.ContextualWordEmbsAug(
            model_path='roberta-base',
            action="substitute",
            aug_min=1,
            aug_max=MAX_REPLACE_BERT
        )
        sentences = aug.augment(sentences)
    if REPLACE_WORDNET in augs:
        aug = naw.SynonymAug(
            aug_src='wordnet',
            aug_min=1,
            aug_max=4,
            stopwords=NO_REPLACE_WORDS,
            verbose=True
        )
        sentences = aug.augment(sentences)
    if BACKTRANSL in augs:
        sentences = back_translation(sentences)
    return sentences


def get_pos_examples_sentence_compr(sentence_compression_instance: Dict,
                                    n: int) -> Tuple[List[str], str]:
    reference_text = sentence_compression_instance[GRAPH][SENTENCE]
    compression = sentence_compression_instance[COMPRESSION][TEXT]
    compression_ratio = sentence_compression_instance[COMPR_RATIO]
    pos_examples = [reference_text]

    # If compression ratio is small, consider the compressed as positive example
    if compression_ratio >= COMPRESSION_RATIO_THRESHOLD:
        pos_examples.append(compression)

    # Generate variations and add it to the pos_examples
    n = n - 1 if len(pos_examples) == 2 else n
    variations = generate_variations(reference_text, n=n)
    pos_examples.extend(variations)

    return pos_examples, reference_text


def get_part_pos_examples_sentence_compr(sentence_compression_instance: Dict,
                                         n: int) -> List[str]:
    reference_text = sentence_compression_instance[GRAPH][SENTENCE]
    compression = sentence_compression_instance[COMPRESSION][TEXT]
    compression_ratio = sentence_compression_instance[COMPR_RATIO]

    part_pos_examples = []
    # If compression ratio is high, consider the compressed as part pos example
    if compression_ratio < COMPRESSION_RATIO_THRESHOLD:
        part_pos_examples.append(compression)

        # Generate variations of the part pos examples
        variations = generate_variations(reference_text,
                                         n=math.ceil(n / 2),
                                         augs=[REPLACE_WORDNET, BACKTRANSL])
        part_pos_examples.extend(variations)
        n = math.floor(n / 2) - 1

    # Get the remaining partially positive examples with adaptive crop
    adaptive_crop_examples = get_part_pos_examples(
        caption=reference_text,
        n_part_pos_examples=n,
        algorithm_type=ADAPTIVE_CROP_AUGMENT
    )
    part_pos_examples.extend(adaptive_crop_examples)

    return part_pos_examples


def create_dataset_chunck_sentence_compression(dataset,
                                               dataset_name: str = "sent_compr",
                                               split: str = TRAIN_SENT_COMPR,
                                               start_idx: int = 0,
                                               chunk_dim: int = CHUNK_DIM,
                                               n_pos_examples: int = N_EXAMPLES,
                                               n_part_pos_examples: int = N_PART_EXAMPLES) -> dict:
    end_idx: int = min(start_idx + chunk_dim, len(dataset[split]))  # end index

    dataset_chunk = {
        DATASET_NAME: dataset_name,
        ANNOTATION_FILE: None,
        INSTANCES: []
    }
    # For each index in the chunck
    for idx in range(start_idx, end_idx):
        # Get the corresponsing instance
        sentence_compression_instance = dataset[split][idx]

        # Get the reference and positive examples
        positive_examples, caption = get_pos_examples_sentence_compr(
            sentence_compression_instance,
            n=n_pos_examples
        )

        # Get the partially positive examples
        part_pos_examples = get_part_pos_examples_sentence_compr(
            sentence_compression_instance,
            n=n_part_pos_examples
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


def create_dataset_sentence_compression(root: str,
                                        dataset,
                                        dataset_name: str = "sent_compr",
                                        split: str = TRAIN_SENT_COMPR,
                                        start_chunk: int = 0,
                                        max_chunk: Optional[int] = None,
                                        chunk_dim: int = CHUNK_DIM,
                                        n_pos_examples: int = N_EXAMPLES,
                                        n_part_pos_examples: int = N_PART_EXAMPLES,
                                        log_tqdm: bool = False) -> int:
    # Create dataset directory if it doesn't exist
    root = os.path.join(root, dataset_name)
    os.makedirs(root, exist_ok=True)

    # Calculate max chunk
    chunk_num = math.ceil(len(dataset[split]) / chunk_dim)
    chunk_num = chunk_num if max_chunk is None else min(chunk_num, max_chunk)

    # For each chunk
    iterable = range(start_chunk, chunk_num)
    iterable = tqdm(iterable, "Creating chunks...") if log_tqdm else iterable
    last_created_chunk_idx = -1
    for chunk_idx in iterable:

        try:
            # Create the chunk
            dataset_chunk = create_dataset_chunck_sentence_compression(
                dataset=dataset,
                split=split,
                dataset_name=dataset_name,
                start_idx=chunk_idx * chunk_dim,
                n_pos_examples=n_pos_examples,
                n_part_pos_examples=n_part_pos_examples
            )

            # Store the chunk
            chunk_filename = os.path.join(root, f"{CHUNK_NAME}_{chunk_idx}.json")
            with open(chunk_filename, "w") as fp:
                json.dump(dataset_chunk, fp)
            last_created_chunk_idx = chunk_idx

        except Exception as e:
            print(f"Chunk {chunk_idx} creation failed due to error: {e}.")
            print(f"Traceback: {traceback.format_exc()}")
            return chunk_idx - 1

    return last_created_chunk_idx
