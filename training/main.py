import argparse
import os
import math
from typing import List, Callable

import torch
from sentence_transformers.util import cos_sim, dot_score
from torch import Tensor
from torch.utils.data import random_split
from dataset.constants import COCO_CAPTIONS_TRAIN, COCO_CAPTIONS_VAL, CLEANED_COCO_TRAIN, CLEANED_COCO_VAL, \
    CLEANED_COCO_TEST, COCO_DS_TYPE, SENTENCE_SUMMARIZATION_DS_TYPE, RAW_DATA, COCO_DS_NAME, CHUNK_DIM, N_EXAMPLES, \
    N_PART_EXAMPLES, SIMILARITY_THRESHOLD, RANDOM_SEED
from dataset.quadruplet_dataset import QuadrupletDataset, RANDOM, HARD_CONTRASTIVE_TRAIN, CACHE_SIZE_DEFAULT


def main(args):
    chunk_n = torch.load(os.path.join(args.dataset_path_train, "chunk_n.pt"))
    qds = QuadrupletDataset(args.dataset_path_train,
                            *[f"chunk_{i}.json" for i in range(0, chunk_n)],
                            hard_contrastive_mode=RANDOM if not args.use_hard_contrastive_sampling \
                                else HARD_CONTRASTIVE_TRAIN,
                            n_pos=args.n_pos,
                            n_neg=args.n_neg,
                            n_part_pos=args.n_part_pos,
                            cache_size=args.cache_size)
    train_set, val_set = random_split(dataset=qds, lengths=[1 - args.validation_split, args.validation_split])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Dataset params
    parser.add_argument('--dataset_path_train', type=str, default=os.path.join(CLEANED_COCO_TRAIN, "coco_ds_train"),
                        help='train dataset path (will be split in train/validation by default).')
    parser.add_argument('--validation_split', type=float, default=0.1,
                        help="validation set split fraction (float between 0 and 1 exclusive)")
    parser.add_argument('--use_hard_contrastive_sampling', type=bool, default=True,
                        help="whether to use hard contrastive sampling in negative examples selection")
    parser.add_argument('--n_pos', type=int, default=1,
                        help='number of positive examples to select for each training instance')
    parser.add_argument('--n_part_pos', type=int, default=1,
                        help='number of partially positive examples to select for each training instance')
    parser.add_argument('--n_neg', type=int, default=1,
                        help='number of negative examples to select for each training instance')
    parser.add_argument('--cache_size', type=int, default=CACHE_SIZE_DEFAULT, help='the dataset cache size (in chunks)')

    # Evaluation params
    """
    main_score_function: str = None,
    main_distance_function: SimilarityFunction = None,
    name: str = '',
    data_loader_sampler = None,
    additional_model_kwargs: Optional[List[str]] = None,
    additional_loss_kwargs: Optional[List[str]] = None,
    use_amp: bool = False
    """

    loss_choices = ["gamma", "discriminator"]
    parser.add_argument('--loss', type=str, default="gamma",
                        choices=loss_choices,
                        help=f'the loss to use, one of the following: {loss_choices}')
    parser.add_argument('--evaluation_queries_path', type=str, default="to be defined",
                        help='the path of the file containing the evaluation queries')
    parser.add_argument('--corpus_chunk_size', type=int, default=50000,
                        help=f'the corpus chunk size')
    parser.add_argument('--mrr_at_k', type=List[int], default=[10],
                        help='the value for the mrr_at_k metric, a python list of int')
    parser.add_argument('--ndcg_at_k', type=List[int], default=[10],
                        help='the value for the ndcg_at_k metric, a python list of int')
    parser.add_argument('--accuracy_at_k', type=List[int], default=[1, 3, 5, 10],
                        help='the value for the accuracy_at_k metric, a python list of int')
    parser.add_argument('--precision_recall_at_k', type=List[int], default=[1, 3, 5, 10],
                        help='the value for the precision_recall_at_k metric, a python list of int')
    parser.add_argument('--map_at_k', type=List[int], default=0,
                        help='the value for the map_at_k metric, a python list of int')
    parser.add_argument('--show_progress_bar', type=bool, default=False,
                        help='whether to show a progress bar')
    parser.add_argument('--batch_size', type=int, default=32, help="the batch size")
    parser.add_argument('--write_csv', type=bool, default=True,
                        help="whether to write the output in a csv")
    parser.add_argument('--score_functions', type=List[Callable[[Tensor, Tensor], Tensor]],
                        default={'cos_sim': cos_sim, 'dot_score': dot_score},
                        help="the score functions to use")
    # fino all'add_argument precedente incluso
    parser.add_argument('--pos_sim_threshold', type=float, default=SIMILARITY_THRESHOLD,
                        help='semantic similarity threshold for two instances to be considered a positive example')

    # Training params
    arguments = parser.parse_args()
    main(arguments)
