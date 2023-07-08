import csv
import json
import logging
import os
import random
from typing import Optional, List, Callable, Dict, Union, Set, final
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import SentenceEvaluator, SimilarityFunction, TripletEvaluator, \
    SequentialEvaluator, InformationRetrievalEvaluator
from sentence_transformers.util import cos_sim, dot_score
from torch import Tensor
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from dataset.constants import REFERENCE_EXAMPLE, POS_EXAMPLES, PART_POS_EXAMPLES, NEG_EXAMPLES
from dataset.quadruplet_dataset import QuadrupletDataset
from models.losses import GammaQuadrupletLoss
from models.losses.losses import QuadrupletLoss
from models.quadruplet_sentence_transformer import QuadrupletSentenceTransformerLossModel


LOGGER = logging.getLogger(__name__)


class QuadrupletLossEvaluator(SentenceEvaluator):
    def __init__(self,
                 data_loader: DataLoader,
                 quadruplet_loss: QuadrupletLoss,
                 additional_model_kwargs: Optional[List[str]] = None,
                 additional_loss_kwargs: Optional[List[str]] = None,
                 use_amp: bool = False):
        self._data_loader = data_loader
        self._quadruplet_loss = quadruplet_loss
        self._additional_model_kwargs = additional_model_kwargs
        self._additional_loss_kwargs = additional_loss_kwargs
        self._use_amp = use_amp

    def __call__(self, model: SentenceTransformer, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        """
        This is called during training to evaluate the model.
        It returns a score for the evaluation with a higher score indicating a better result.

        :param model:
            the model to evaluate
        :param output_path:
            path where predictions and metrics are written to
        :param epoch:
            the epoch where the evaluation takes place.
            This is used for the file prefixes.
            If this is -1, then we assume evaluation on test data
        :param steps: the steps in the current epoch at time of the evaluation.
            This is used for the file prefixes.
            If this is -1, then we assume evaluation at the end of the epoch.
        :return: a score for the evaluation with a higher score indicating a better result
        """

        loss_model = QuadrupletSentenceTransformerLossModel(
            st_model=model,
            quadruplet_loss=self._quadruplet_loss,
            additional_model_kwargs=self._additional_model_kwargs,
            additional_loss_kwargs=self._additional_loss_kwargs
        )

        average_loss = 0.0

        with torch.no_grad():
            for i, instance in enumerate(self._data_loader):
                features, labels = instance
                labels = labels.to(model.device)

                if self._use_amp:
                    with autocast():
                        loss_value = loss_model(features, labels)
                else:
                    loss_value = loss_model(features, labels)

                average_loss = average_loss + 1 / (i + 1) * (loss_value - average_loss)

        if output_path is not None:
            log_dict = {
                "epoch": epoch,
                "steps": steps,
                "average_loss": average_loss
            }

            if not os.path.exists(output_path):
                full_log_dict: dict[str, list] = {}
            else:
                with open(output_path, "r") as fp:
                    full_log_dict: dict[str, list] = json.load(fp)

            for k in log_dict:
                if k not in full_log_dict:
                    full_log_dict[k] = []
                full_log_dict[k].append(log_dict[k])

            with open(output_path, "w") as fp:
                json.dump(full_log_dict, fp)

        return average_loss


class QuadrupletEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on a quadruplet: (sentence, positive_example, partially_positive_example, negative_example).
    Checks if distance(sentence, positive_example) < distance(sentence, negative_example),
    distance(sentence, positive_example) < distance(sentence, partially_positive_example) and
    distance(sentence, partially_positive_example) < distance(sentence, negative_example).
    """

    N_EPOCHS_RESET_EXAMPLES: final = 5  # number of epochs after which examples are re-sampled

    def __init__(
            self,
            anchors: List[str],
            positives: List[str],
            partially_positives: List[str],
            negatives: List[str],
            gamma: float = 0.8,
            main_distance_function: SimilarityFunction = None,
            name: str = "",
            batch_size: int = 16,
            show_progress_bar: bool = False,
            write_csv: bool = True,
            all_examples: Optional[QuadrupletDataset] = None
    ):
        """
        :param anchors: Sentences to check similarity to. (e.g. a query)
        :param positives: List of positive sentences
        :param partially_positives: List of partially positive sentences
        :param negatives: List of negative sentences
        :param main_distance_function: One of 0 (Cosine), 1 (Euclidean) or 2 (Manhattan). Defaults to None, returning
            all 3.
        :param name: Name for the output
        :param batch_size: Batch size used to compute embeddings
        :param show_progress_bar: If true, prints a progress bar
        :param write_csv: Write results to a CSV file
        """
        self.anchors = anchors
        self.positives = positives
        self.partially_positives = partially_positives
        self.negatives = negatives
        self.name = name
        self._gamma: float = gamma
        self._all_examples = all_examples  # full dataset to make possible resetting examples

        assert len(self.anchors) == len(self.positives)
        assert len(self.anchors) == len(self.partially_positives)
        assert len(self.anchors) == len(self.negatives)

        self.main_distance_function = main_distance_function

        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (
                    LOGGER.getEffectiveLevel() == logging.INFO or LOGGER.getEffectiveLevel() == logging.DEBUG
            )
        self.show_progress_bar = show_progress_bar

        self._pos_part_triplet = TripletEvaluator(
            anchors=anchors,
            positives=positives,
            negatives=partially_positives,
            main_distance_function=main_distance_function,
            name="pos_part",
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            write_csv=write_csv,
        )
        self._pos_neg_triplet = TripletEvaluator(
            anchors=anchors,
            positives=positives,
            negatives=negatives,
            main_distance_function=main_distance_function,
            name="pos_neg",
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            write_csv=write_csv,
        )
        self._part_neg_triplet = TripletEvaluator(
            anchors=anchors,
            positives=partially_positives,
            negatives=negatives,
            main_distance_function=main_distance_function,
            name="part_neg",
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            write_csv=write_csv,
        )

        self.csv_file: str = "quadruplet_evaluation" + ("_" + name if name else "") + "_results.csv"
        self.csv_headers = ["epoch", "steps", "pos_part_accuracy", "pos_neg_accuracy", "part_neg_accuracy",
                            "global_accuracy"]
        self.write_csv = write_csv
        self._epoch_counter = 0

    @classmethod
    def from_input_examples(cls, examples: QuadrupletDataset, **kwargs):
        anchors = []
        positives = []
        partially_positives = []
        negatives = []

        for example in examples:
            # Get the anchor
            anchors.append(example[REFERENCE_EXAMPLE])

            # Get one random positive, negative and partially positive
            positives.append(example[POS_EXAMPLES][random.randint(0, len(example[POS_EXAMPLES]) - 1)])
            partially_positives.append(
                example[PART_POS_EXAMPLES][random.randint(0, len(example[PART_POS_EXAMPLES]) - 1)]
            )
            negatives.append(example[NEG_EXAMPLES][random.randint(0, len(example[NEG_EXAMPLES]) - 1)])

        return cls(anchors, positives, partially_positives, negatives, all_examples=examples, **kwargs)

    def _reset_examples(self):
        self._epoch_counter += 1  # increase epoch counter

        # Every N_EPOCHS_RESET epochs, reset the positive, negative and partially positive examples by resampling them
        if self._all_examples is not None and self._epoch_counter % self.N_EPOCHS_RESET_EXAMPLES == 0:
            anchors = []
            positives = []
            partially_positives = []
            negatives = []
            for example in self._all_examples:
                # Get the anchor
                self.anchors.append(example[REFERENCE_EXAMPLE])

                # Get one random positive, negative and partially positive
                positives.append(example[POS_EXAMPLES][random.randint(0, len(example[POS_EXAMPLES]) - 1)])
                partially_positives.append(
                    example[PART_POS_EXAMPLES][random.randint(0, len(example[PART_POS_EXAMPLES]) - 1)]
                )
                negatives.append(example[NEG_EXAMPLES][random.randint(0, len(example[NEG_EXAMPLES]) - 1)])

                self._pos_part_triplet = TripletEvaluator(
                    anchors=anchors,
                    positives=positives,
                    negatives=partially_positives,
                    main_distance_function=self.main_distance_function,
                    name="pos_part",
                    batch_size=self.batch_size,
                    show_progress_bar=self.show_progress_bar,
                    write_csv=self.write_csv,
                )
                self._pos_neg_triplet = TripletEvaluator(
                    anchors=anchors,
                    positives=positives,
                    negatives=negatives,
                    main_distance_function=self.main_distance_function,
                    name="pos_neg",
                    batch_size=self.batch_size,
                    show_progress_bar=self.show_progress_bar,
                    write_csv=self.write_csv,
                )
                self._part_neg_triplet = TripletEvaluator(
                    anchors=anchors,
                    positives=partially_positives,
                    negatives=negatives,
                    main_distance_function=self.main_distance_function,
                    name="part_neg",
                    batch_size=self.batch_size,
                    show_progress_bar=self.show_progress_bar,
                    write_csv=self.write_csv,
                )

            self.anchors = anchors
            self.positives = positives
            self.partially_positives = partially_positives
            self.negatives = negatives

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:

        self._reset_examples()

        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        LOGGER.info("QuadrupletEvaluator: Evaluating the model on " + self.name + " dataset" + out_txt)

        # Compute accuracies
        pos_part_accuracy = self._pos_part_triplet(model, output_path, epoch, steps)
        pos_neg_accuracy = self._pos_neg_triplet(model, output_path, epoch, steps)
        part_neg_accuracy = self._part_neg_triplet(model, output_path, epoch, steps)

        # Compute global accuracy, following the quadruplet loss formula
        glob_accuracy = (((1-self._gamma) * pos_part_accuracy + self._gamma * part_neg_accuracy) + pos_neg_accuracy)/2

        LOGGER.info("Pos-Part Accuracy Distance:   \t{:.2f}".format(pos_part_accuracy * 100))
        LOGGER.info("Pos-Neg Accuracy Distance:   \t{:.2f}".format(pos_neg_accuracy * 100))
        LOGGER.info("Part-Neg Distance:   \t{:.2f}".format(part_neg_accuracy * 100))
        LOGGER.info("Accuracy Distance:   \t{:.2f}".format(glob_accuracy * 100))

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, newline="", mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow(
                        [epoch, steps, pos_part_accuracy, pos_neg_accuracy, part_neg_accuracy, glob_accuracy])

            else:
                with open(csv_path, newline="", mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [epoch, steps, pos_part_accuracy, pos_neg_accuracy, part_neg_accuracy, glob_accuracy])

        return glob_accuracy


def get_sequential_evaluator(dataset: QuadrupletDataset,
                             loss: GammaQuadrupletLoss,
                             evaluation_queries_path: Optional[str] = None,
                             corpus_chunk_size: int = 50000,
                             mrr_at_k: List[int] = [10],
                             ndcg_at_k: List[int] = [10],
                             accuracy_at_k: List[int] = [1, 3, 5, 10],
                             precision_recall_at_k: List[int] = [1, 3, 5, 10],
                             map_at_k: List[int] = [100],
                             show_progress_bar: bool = False,
                             batch_size: int = 32,
                             write_csv: bool = True,
                             score_functions: List[Callable[[Tensor, Tensor], Tensor]] = {'cos_sim': cos_sim,
                                                                                          'dot_score': dot_score},
                             main_score_function: str = None,
                             main_distance_function: SimilarityFunction = None,
                             name: str = '',
                             data_loader_sampler = None,
                             additional_model_kwargs: Optional[List[str]] = None,
                             additional_loss_kwargs: Optional[List[str]] = None,
                             use_amp: bool = False) -> SequentialEvaluator:

    if evaluation_queries_path is not None:
        try:
            with open(evaluation_queries_path, "r") as fp:
                evaluation_queries: Optional[Dict[str, Dict[str, Union[str, List[str], Set[str]]]]] = json.load(fp)

                # Convert the relevant queries to sets as required by the evaluator
                for q in evaluation_queries["relevant"]:
                    evaluation_queries["relevant"][q] = set(evaluation_queries["relevant"])
        except IOError as e:
            evaluation_queries = None
            print(f"Error: {evaluation_queries_path} file could not be opened due to error: {e}")
    else:
        evaluation_queries = None

    evaluators = []
    if evaluation_queries is not None:
        information_retrieval_evaluator = InformationRetrievalEvaluator(
            queries=evaluation_queries["queries"],
            corpus=evaluation_queries["corpus"],
            relevant_docs=evaluation_queries["relevant"],
            corpus_chunk_size=corpus_chunk_size,
            mrr_at_k=mrr_at_k,
            ndcg_at_k=ndcg_at_k,
            accuracy_at_k=accuracy_at_k,
            precision_recall_at_k=precision_recall_at_k,
            map_at_k=map_at_k,
            show_progress_bar=show_progress_bar,
            batch_size=batch_size,
            name=name,
            write_csv=write_csv,
            score_functions=score_functions,
            main_score_function=main_score_function
        )
        evaluators.append(information_retrieval_evaluator)

    quadruplet_evaluator = QuadrupletEvaluator.from_input_examples(
        examples=dataset,
        gamma=loss.gamma,
        main_distance_function=main_distance_function,
        name=name,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        write_csv=write_csv
    )
    evaluators.append(quadruplet_evaluator)

    # Loss will be the main score function, so it must be the last one
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=data_loader_sampler)
    quadruplet_loss_evaluator = QuadrupletLossEvaluator(
        data_loader=data_loader,
        quadruplet_loss=loss,
        additional_model_kwargs=additional_model_kwargs,
        additional_loss_kwargs=additional_loss_kwargs,
        use_amp=use_amp
    )
    evaluators.append(quadruplet_loss_evaluator)

    return SequentialEvaluator(evaluators=evaluators)
