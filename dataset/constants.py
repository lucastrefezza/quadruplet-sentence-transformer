from typing import final

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
DATASET_NAME: final = "dataset_name"
