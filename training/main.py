import argparse
import os
import math
from dataset.constants import COCO_CAPTIONS_TRAIN, COCO_CAPTIONS_VAL, CLEANED_COCO_TRAIN, CLEANED_COCO_VAL, \
    CLEANED_COCO_TEST, COCO_DS_TYPE, SENTENCE_SUMMARIZATION_DS_TYPE, RAW_DATA, COCO_DS_NAME, CHUNK_DIM, N_EXAMPLES, \
    N_PART_EXAMPLES, SIMILARITY_THRESHOLD
from dataset.quadruplet_dataset import QuadrupletDataset, RANDOM, HARD_CONTRASTIVE_TRAIN


def main(args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_root', type=str, default=RAW_DATA,
                        help='raw data root (must contain an "images" empty subdir)')
    parser.add_argument('--dataset_name', type=str, default=COCO_DS_NAME, help='cleaned dataset folder name')
    parser.add_argument('--raw_train_path', type=str, default=COCO_CAPTIONS_TRAIN, help='train dataset path')
    parser.add_argument('--raw_val_path', type=str, default=COCO_CAPTIONS_VAL, help='val dataset path')
    parser.add_argument('--train_out_path', type=str, default=CLEANED_COCO_TRAIN, help='train set output path')
    parser.add_argument('--test_out_path', type=str, default=CLEANED_COCO_TEST, help='test set output path')
    parser.add_argument('--dataset_type', type=str, default=COCO_DS_TYPE,
                        choices=[COCO_DS_TYPE, SENTENCE_SUMMARIZATION_DS_TYPE],
                        help=f'dataset type, either {COCO_DS_TYPE} or {SENTENCE_SUMMARIZATION_DS_TYPE}')
    parser.add_argument('--verbose_check', type=bool, default=True,
                        help='whether to print random elements from created datasets for sanity checking')
    parser.add_argument('--verbose_creation', type=bool, default=True,
                        help='whether to log the progress of the dataset creation')
    parser.add_argument('--start_chunk_train', type=int, default=0,
                        help='the train set chunk from which to start the dataset creation (inclusive)')
    parser.add_argument('--last_chunk_train', type=int, default=-1,
                        help='the train set chunk where to stop the dataset creation (inclusive), -1 means last')
    parser.add_argument('--start_chunk_test', type=int, default=0,
                        help='the test set chunk from which to start the dataset creation (inclusive)')
    parser.add_argument('--last_chunk_test', type=int, default=-1,
                        help='the test set chunk where to stop the dataset creation (inclusive), -1 means last')
    parser.add_argument('--chunk_dim', type=int, default=CHUNK_DIM, help="number of instances per dataset chunk")
    parser.add_argument('--n_pos_examples', type=int, default=N_EXAMPLES,
                        help="number of positive examples per instance")
    parser.add_argument('--n_part_pos_examples', type=int, default=N_PART_EXAMPLES,
                        help="number of partially positive examples per instance")
    parser.add_argument('--pos_sim_threshold', type=float, default=SIMILARITY_THRESHOLD,
                        help='semantic similarity threshold for two instances to be considered a positive example')
    arguments = parser.parse_args()
    main(arguments)
