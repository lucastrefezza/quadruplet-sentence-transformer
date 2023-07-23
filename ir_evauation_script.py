import argparse
import json
import os
from typing import Optional, Dict, Union, Set, List
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.util import cos_sim, dot_score
from torch.utils.data import random_split, Subset
from dataset.constants import CLEANED_COCO_TRAIN
from dataset.quadruplet_dataset import QuadrupletDataset, RANDOM, HARD_CONTRASTIVE_TRAIN, CACHE_SIZE_DEFAULT
from models.quadruplet_sentence_transformer import to_input_example
from models.evaluators import N_IR_SAMPLES, euclidean_score, create_ir_evaluation_set


# noinspection PyTypeChecker
def main(args):
    print(f"Launching IR evaluation with params: {json.dumps(args.__dict__, indent=2)}...")

    # Load the dataset
    chunk_n = torch.load(os.path.join(args.dataset_path_train, "chunk_n.pt"))
    hard_contrastive_mode = RANDOM if not args.use_hard_contrastive_sampling else HARD_CONTRASTIVE_TRAIN
    qds = QuadrupletDataset(args.dataset_path_train,
                            *[f"chunk_{i}.json" for i in range(0, chunk_n)],
                            hard_contrastive_mode=hard_contrastive_mode,
                            n_pos=args.n_pos,
                            n_neg=args.n_neg,
                            n_part_pos=args.n_part_pos,
                            cache_size=args.cache_size,
                            transform=to_input_example)

    # Needed for information retrieval evaluation (this sucks but there seems not to be another way)
    nt_qds = QuadrupletDataset(args.dataset_path_train,
                               *[f"chunk_{i}.json" for i in range(0, chunk_n)],
                               hard_contrastive_mode=hard_contrastive_mode,
                               n_pos=4,
                               n_neg=1,
                               n_part_pos=4,
                               cache_size=args.cache_size,
                               transform=None)

    train_set, val_set = random_split(dataset=qds, lengths=[1 - args.validation_split, args.validation_split])
    no_transform_val_set = Subset(nt_qds, val_set.indices[0:N_IR_SAMPLES])

    # Create output folders if they don't exist
    out_path = os.path.join(args.out_path, args.model_path)
    os.makedirs(out_path, exist_ok=True)

    # Create the evaluators
    score_functions = {'cos_sim': cos_sim, 'dot_score': dot_score, "euclid_score": euclidean_score}
    if args.score_functions == "cos_sim":
        del score_functions['dot_score']
        del score_functions['euclid']
    elif args.score_functions == "dot_score":
        del score_functions['cos_sim']
        del score_functions['euclid']
    elif args.score_functions == 'euclid':
        del score_functions['dot_score']
        del score_functions['cos_sim']
    elif args.score_functions == "euclid_and_cos":
        del score_functions['dot_score']
    elif args.score_functions == 'euclid_and_dot':
        del score_functions['cos_sim']
    elif args.score_functions == 'cos_and_dot':
        del score_functions['euclid']

    evaluation_queries = None
    if args.evaluation_queries_path is not None:
        try:
            with open(args.evaluation_queries_path, "r") as fp:
                evaluation_queries: Optional[Dict[str, Dict[str, Union[str, List[str], Set[str]]]]] = json.load(fp)

                # Convert the relevant queries to sets as required by the evaluator
                for q in evaluation_queries["relevant"]:
                    evaluation_queries["relevant"][q] = set(evaluation_queries["relevant"])
        except IOError as e:
            evaluation_queries = None
            print(f"Error: {args.evaluation_queries_path} file could not be opened due to error: {e}")
    if evaluation_queries is None and no_transform_val_set is not None:
        evaluation_queries = create_ir_evaluation_set(dataset=no_transform_val_set,
                                                      out_path=os.path.join(out_path, "created_eval_queries.json"),
                                                      use_pos=not args.dont_use_pos_rel,
                                                      use_part_pos=not args.dont_use_part_pos_rel,
                                                      use_cross_encoder=args.use_cross_encoder,
                                                      add_part_pos_corpus=not args.dont_use_part_pos)

    evaluator = InformationRetrievalEvaluator(
        queries=evaluation_queries["queries"],
        corpus=evaluation_queries["corpus"],
        relevant_docs=evaluation_queries["relevant"],
        corpus_chunk_size=args.corpus_chunk_size,
        mrr_at_k=args.mrr_at_k,
        ndcg_at_k=args.ndcg_at_k,
        accuracy_at_k=args.accuracy_at_k,
        precision_recall_at_k=args.precision_recall_at_k,
        map_at_k=args.map_at_k,
        show_progress_bar=args.show_progress_bar,
        batch_size=args.batch_size,
        write_csv=args.write_csv,
        score_functions=score_functions,
        main_score_function=None,
        name=f"{args.model_path.replace('/', '_')}"
    )

    # Create the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline = SentenceTransformer(args.baseline_model, device=device)
    model = SentenceTransformer(args.model_path, device=device)

    evaluator(model=baseline, output_path=out_path)
    evaluator(model=model, output_path=out_path)

    print("IR evaluation completed.")


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
    parser.add_argument("--out_path", type=str, default='_out_ir_eval/eval1')
    parser.add_argument('--evaluation_queries_path', type=str, default="to be defined",
                        help='the path of the file containing the evaluation queries')
    parser.add_argument('--corpus_chunk_size', type=int, default=50000,
                        help=f'the corpus chunk size')
    parser.add_argument('--mrr_at_k', type=int, nargs='+', default=[5, 10, 20, 30, 40, 50, 100, 200, 500, 900],
                        help='the value for the mrr_at_k metric, a python list of int')
    parser.add_argument('--ndcg_at_k', type=int, nargs='+', default=[5, 10, 20, 30, 40, 50, 100, 200, 500, 900],
                        help='the value for the ndcg_at_k metric, a python list of int')
    parser.add_argument('--accuracy_at_k', type=int, nargs='+',
                        default=[1, 3, 5, 10, 20, 30, 40, 50, 100, 200, 500, 900],
                        help='the value for the accuracy_at_k metric, a python list of int')
    parser.add_argument('--precision_recall_at_k', type=int, nargs='+',
                        default=[1, 3, 5, 10, 20, 30, 40, 50, 100, 200, 500, 900],
                        help='the value for the precision_recall_at_k metric, a python list of int')
    parser.add_argument('--map_at_k', type=int, nargs='+', default=[1, 3, 5, 10, 20, 30, 40, 50, 100, 200, 500, 900],
                        help='the value for the map_at_k metric, a python list of int')
    parser.add_argument('--show_progress_bar', type=bool, default=True,
                        help='whether to show a progress bar')
    parser.add_argument('--write_csv', type=bool, default=True,
                        help="whether to write the output in a csv")
    score_choices = ["cos_sim", "dot_score", "euclid", "euclid_and_cos", "euclid_and_dot", "cos_and_dot", "all"]
    parser.add_argument('--score_functions', type=str,
                        default="all",
                        choices=score_choices,
                        help=f"the score functions to use, one of ${score_choices}")
    parser.add_argument('--main_score_function', type=str, default=None,
                        help='the main score function')
    parser.add_argument('--batch_size', type=int, default=32, help='the batch size for train and validation')

    # Eval set params
    parser.add_argument('--use_cross_encoder', action="store_true",
                        help="whether to use the cross encoder to create the eval set")
    parser.add_argument('--dont_use_part_pos', action="store_true",
                        help="do not use partially positive examples in corpus")
    parser.add_argument('--dont_use_part_pos_rel', action="store_true",
                        help="do not use partially positive examples as relevant")
    parser.add_argument('--dont_use_pos_rel', action="store_true",
                        help="do not use positive examples as relevant")

    # Model params
    parser.add_argument('--baseline_model', type=str, default='all-MiniLM-L6-v2',
                        help="the SentenceTransformer model type")
    parser.add_argument('--model_path', type=str, default='trained/exp5',
                        help="the quadruplet sentence transformer")

    arguments = parser.parse_args()
    main(arguments)
