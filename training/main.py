import argparse
import os
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim, dot_score
from torch.utils.data import random_split, DataLoader, Subset
from dataset.constants import CLEANED_COCO_TRAIN, OUTPUT_PATH
from dataset.quadruplet_dataset import QuadrupletDataset, RANDOM, HARD_CONTRASTIVE_TRAIN, CACHE_SIZE_DEFAULT
from models.quadruplet_sentence_transformer import QuadrupletSentenceTransformerLossModel, to_input_example
from models.evaluators import get_sequential_evaluator
from models.losses import GammaQuadrupletLoss
from models.losses.losses import DEFAULT_GAMMA
from training.callbacks import EarlyStoppingCallback, EarlyStoppingException


# noinspection PyTypeChecker
def main(args):
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
                               n_pos=args.n_pos,
                               n_neg=args.n_neg,
                               n_part_pos=args.n_part_pos,
                               cache_size=args.cache_size,
                               transform=None)

    train_set, val_set = random_split(dataset=qds, lengths=[1 - args.validation_split, args.validation_split])
    dl_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    no_transform_val_set = Subset(nt_qds, val_set.indices)

    # Create the loss
    quadruplet_loss = GammaQuadrupletLoss(
        gamma=args.gamma,
        margin_pos_neg=args.margin_pos_neg,
        margin_pos_part=args.margin_pos_part,
        margin_part_neg=args.margin_part_neg,
        p=args.p
    )

    # Create the evaluators
    experiment_name = args.experiment_name
    score_functions = {'cos_sim': cos_sim, 'dot_score': dot_score}
    if args.score_functions == "cos_sim":
        del score_functions['dot_score']
    elif args.score_functions == "dot_score":
        del score_functions['cos_sim']
    evaluator = get_sequential_evaluator(
        dataset=val_set,
        no_transform_dataset=no_transform_val_set,
        loss=quadruplet_loss,
        evaluation_queries_path=args.evaluation_queries_path,
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
        main_distance_function=None,
        name=experiment_name,
        use_amp=args.use_amp
    )
    '''evaluator = QuadrupletLossEvaluator(
        quadruplet_dataset=val_set,
        quadruplet_loss=quadruplet_loss,
        batch_size=args.batch_size
    )'''

    # Create output folders if they don't exist
    complete_experiment_path = os.path.join(args.experiment_path, experiment_name)
    checkpoints_path = os.path.join(complete_experiment_path, "checkpoints")
    os.makedirs(complete_experiment_path, exist_ok=True)
    os.makedirs(checkpoints_path, exist_ok=True)

    # Create the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(model_name_or_path=args.model_name, device=device)
    quadruplet_loss_model = QuadrupletSentenceTransformerLossModel(
        st_model=model,
        quadruplet_loss=quadruplet_loss
    )

    # Train the model
    train_objectives = [(dl_train, quadruplet_loss_model)]
    early_stopping = EarlyStoppingCallback(patience=args.early_stopping_patience,
                                           delta=args.early_stopping_delta,
                                           minimization=True)
    try:
        model.fit(
            train_objectives=train_objectives,
            evaluator=evaluator,
            epochs=args.epochs,
            steps_per_epoch=None,
            scheduler=args.scheduler,
            warmup_steps=args.warmup_steps,
            optimizer_class=torch.optim.AdamW,
            optimizer_params={'lr': args.learning_rate},
            weight_decay=args.weight_decay,
            evaluation_steps=args.evaluation_steps,
            output_path=complete_experiment_path,
            save_best_model=args.save_best_model,
            max_grad_norm=args.max_grad_norm,
            use_amp=args.use_amp,
            callback=early_stopping,
            show_progress_bar=args.show_progress_bar,
            checkpoint_path=checkpoints_path,
            checkpoint_save_steps=args.checkpoint_save_steps,
            checkpoint_save_total_limit=args.checkpoint_save_total_limit
        )
    except EarlyStoppingException as e:
        print(f"Training ended earlier due to early stopping. {e}")


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
    loss_choices = ["gamma", "discriminator"]
    parser.add_argument('--loss', type=str, default="gamma",
                        choices=loss_choices,
                        help=f'the loss to use, one of the following: {loss_choices}')
    parser.add_argument('--evaluation_queries_path', type=str, default="to be defined",
                        help='the path of the file containing the evaluation queries')
    parser.add_argument('--corpus_chunk_size', type=int, default=50000,
                        help=f'the corpus chunk size')
    parser.add_argument('--mrr_at_k', type=int, nargs='+', default=[10],
                        help='the value for the mrr_at_k metric, a python list of int')
    parser.add_argument('--ndcg_at_k', type=int, nargs='+', default=[10],
                        help='the value for the ndcg_at_k metric, a python list of int')
    parser.add_argument('--accuracy_at_k', type=int, nargs='+', default=[1, 3, 5, 10],
                        help='the value for the accuracy_at_k metric, a python list of int')
    parser.add_argument('--precision_recall_at_k', type=int, nargs='+', default=[1, 3, 5, 10],
                        help='the value for the precision_recall_at_k metric, a python list of int')
    parser.add_argument('--map_at_k', type=int, nargs='+', default=[1, 3, 5, 10],
                        help='the value for the map_at_k metric, a python list of int')
    parser.add_argument('--show_progress_bar', type=bool, default=True,
                        help='whether to show a progress bar')
    parser.add_argument('--write_csv', type=bool, default=True,
                        help="whether to write the output in a csv")
    score_choices = ["cos_sim", "dot_score", "both"]
    parser.add_argument('--score_functions', type=str,
                        default="both",
                        choices=score_choices,
                        help=f"the score functions to use, one of ${score_choices}")
    parser.add_argument('--main_score_function', type=str, default=None,
                        help='the main score function')
    parser.add_argument('--use_amp', type=bool, default=False, help="whether to use mixed-precision evaluation")

    # Experiment params
    parser.add_argument('--experiment_path', type=str, default=OUTPUT_PATH,
                        help="the path to save the experiment files in")
    parser.add_argument('--experiment_name', type=str, default='', help='the experiment name')

    # Loss params
    parser.add_argument('--gamma', type=float, default=DEFAULT_GAMMA, help='gamma parameter in gamma quadruplet loss')
    parser.add_argument('--margin_pos_neg', type=float, default=1.0,
                        help='margin between positive and negative examples')
    parser.add_argument('--margin_pos_part', type=float, default=0.5,
                        help='margin between positive and partially positive examples')
    parser.add_argument('--margin_part_neg', type=float, default=0.5,
                        help='margin between partially positive and negative examples')
    parser.add_argument('--p', type=float, default=2.0, help='p-norm type to use in the loss')

    # Training params
    parser.add_argument('--batch_size', type=int, default=32, help='the batch size for train and validation')
    parser.add_argument('--epochs', type=int, default=10, help='the number of training epochs')
    schedulers = ['constantlr', 'warmupconstant', 'warmuplinear', 'warmupcosine', 'warmupcosinewithhardrestarts']
    parser.add_argument('--scheduler', type=str, default='warmuplinear',
                        choices=schedulers,
                        help=f'learning rate scheduler, one of {schedulers}')
    parser.add_argument('--warmup_steps', type=int, default=10000, help='the number of warmup steps')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='the learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='the weight decay rate')
    parser.add_argument('--evaluation_steps', type=int, default=500,
                        help='if > 0, an evaluation step is performed after that much training steps')
    parser.add_argument('--save_best_model', type=bool, default=True, help='whether to save the best model')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='the maximum gradient norm')
    parser.add_argument('--checkpoint_save_steps', type=int, default=500,
                        help='every how many steps save a checkpoint')
    parser.add_argument('--checkpoint_save_total_limit', type=int, default=0,
                        help='the max number of checkpoint to save')
    parser.add_argument('--early_stopping_patience', type=int, default=5, help='early stopping patience (in epochs)')
    parser.add_argument('--early_stopping_delta', type=float, default=0.0,
                        help='the minimum improvement considered by early stopping')

    # Model params
    parser.add_argument('--model_name', type=str, default='all-MiniLM-L6-v2', help="the SentenceTransformer model type")

    arguments = parser.parse_args()
    main(arguments)
