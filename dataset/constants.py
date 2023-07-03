from typing import final
import os


RANDOM_SEED: final = 14
SIMILARITY_THRESHOLD: final = 0.6
N_EXAMPLES: final = 4
MAX_ATTEMPTS: final = 5
MAX_WORDS_TO_REPLACE: final = 5
NO_REPLACE_WORDS: final = ["pole", "stop", "sign", "it", "its", "post",
                           "airway", "airways", "a", "the", "tree", "trees",
                           "room", "giraffe"]
CHUNK_DIM: final = 500
N_PART_EXAMPLES: final = 8
CHAT_GPT: final = "chatgpt"
ALPACA: final = "alpaca"
FALCON: final = "falcon"
ADAPTIVE_CROP: final = "adaptive_crop"
ADAPTIVE_CROP_AUGMENT: final = "adaptive_crop_augment"
ID: final = "id"
REFERENCE_EXAMPLE: final = "reference"
POS_EXAMPLES: final = "positive"
NEG_EXAMPLES: final = "negative"
PART_POS_EXAMPLES: final = "part_positive"
CHUNK_NAME: final = "chunk"
ANNOTATION_FILE: final = "ann_file"
INSTANCES: final = "instances"
DATASET_NAME_DEFAULT: final = "dataset_name"


# Paths
DATA_PATH: final = "data"
RAW_DATA: final = os.path.join(DATA_PATH, "raw")
COCO_ANNOTATIONS_DIR: final = os.path.join(RAW_DATA, "annotations")
COCO_CAPTIONS_TRAIN: final = os.path.join(COCO_ANNOTATIONS_DIR, "captions_train2017.json")
COCO_CAPTIONS_VAL: final = os.path.join(COCO_ANNOTATIONS_DIR, "captions_val2017.json")
CLEANED_DATA: final = os.path.join(DATA_PATH, "cleaned")
CLEANED_COCO_DIR: final = os.path.join(CLEANED_DATA, "coco")
CLEANED_COCO_TRAIN: final = os.path.join(CLEANED_COCO_DIR, "train")
CLEANED_COCO_VAL: final = os.path.join(CLEANED_COCO_DIR, "validation")
CLEANED_COCO_TEST: final = os.path.join(CLEANED_COCO_DIR, "test")
COCO_DS_TYPE: final = "coco"
SENTENCE_SUMMARIZATION_DS_TYPE: final = "sentence_summarization"
COCO_DS_NAME: final = "coco_ds"
