import csv
import json
import logging
import os
import random
from typing import Optional, List, Callable, Dict, Union, Set, final, Final
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, InputExample, CrossEncoder
from sentence_transformers.evaluation import SentenceEvaluator, SimilarityFunction, TripletEvaluator, \
    SequentialEvaluator, InformationRetrievalEvaluator
from sentence_transformers.util import cos_sim, dot_score, batch_to_device
from torch import Tensor
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from dataset.constants import REFERENCE_EXAMPLE, POS_EXAMPLES, PART_POS_EXAMPLES, NEG_EXAMPLES, RANDOM_SEED
from dataset.quadruplet_dataset import QuadrupletDataset
from dataset.sentence_compr_dataset_creation import generate_variations
from models.losses import GammaQuadrupletLoss
from models.losses.losses import QuadrupletLoss
from models.quadruplet_sentence_transformer import QuadrupletSentenceTransformerLossModel


IR_EVALUATION_PATH: Final[str] = os.path.join("data", "ir_evaluation", "ir_evaluation_dataset.json")
N_IR_SAMPLES: Final[int] = 1000  # as in ladder loss
SIMILARITY_THRESHOLD: Final[float] = 0.4  # probably we need to increase this
LOGGER = logging.getLogger(__name__)
# cross-encoder/stsb-roberta-large: 91.47% accuracy, cross-encoder/stsb-roberta-base: 90.17% accuracy,
# cross-encoder/stsb-distilroberta-base: 87.92% accuracy, cross-encoder/stsb-TinyBERT-L-4: 85.50% accuracy
cross_encoder_singleton = CrossEncoder("cross-encoder/stsb-roberta-large")


class QuadrupletLossEvaluator(SentenceEvaluator):
    def __init__(self,
                 quadruplet_dataset: QuadrupletDataset,
                 quadruplet_loss: QuadrupletLoss,
                 batch_size: int = 32,
                 additional_model_kwargs: Optional[List[str]] = None,
                 additional_loss_kwargs: Optional[List[str]] = None,
                 use_amp: bool = False):
        self._quadruplet_dataset = quadruplet_dataset
        self._quadruplet_loss = quadruplet_loss
        self._batch_size = batch_size
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
        smart_batching_dl = DataLoader(
            dataset=self._quadruplet_dataset,
            batch_size=self._batch_size,
            collate_fn=model.smart_batching_collate
        )
        full_out_path = os.path.join(output_path, "_quadruplet_loss_eval.json")
        with torch.no_grad():
            desc = f"Evaluating quadruplet loss. Average loss: {average_loss}"
            progress_bar = tqdm(range(0, len(smart_batching_dl)), desc=desc)
            for i, instance in enumerate(smart_batching_dl):

                # Move features to device if required
                features, labels = instance
                features = list(map(lambda batch: batch_to_device(batch, model.device), features))
                labels = labels.to(model.device)

                if self._use_amp:
                    with autocast():
                        loss_value = loss_model(features, labels)
                else:
                    loss_value = loss_model(features, labels)

                average_loss = average_loss + 1 / (i + 1) * (loss_value - average_loss)

                # Update progress bar
                desc = f"Evaluating quadruplet loss. Average loss: {average_loss}"
                progress_bar.update(n=1)
                progress_bar.set_description(desc=desc)
            progress_bar.close()

        if output_path is not None:
            log_dict = {
                "epoch": epoch,
                "steps": steps,
                "average_loss": average_loss.item()
            }

            if not os.path.exists(full_out_path):
                full_log_dict: dict[str, list] = {}
            else:
                with open(full_out_path, "r") as fp:
                    full_log_dict: dict[str, list] = json.load(fp)

            for k in log_dict:
                if k not in full_log_dict:
                    full_log_dict[k] = []
                full_log_dict[k].append(log_dict[k])

            with open(full_out_path, "w") as fp:
                json.dump(full_log_dict, fp, indent=2)

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
            gamma: float = 0.6,
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

        # noinspection PyTypeChecker
        for example in tqdm(examples, desc="Sampling examples for QuadrupletEvaluator..."):
            if isinstance(example, tuple):
                example, _ = example
            if isinstance(example, InputExample):
                texts = example.texts
                anchors.append(texts[0])
                positives.append(texts[1])
                partially_positives.append(texts[2])
                negatives.append(texts[3])
            else:

                # Get the anchor
                anchors.append(example[REFERENCE_EXAMPLE])

                # Get one random positive, partially positive and negative
                if isinstance(example[POS_EXAMPLES], list):
                    positives.append(example[POS_EXAMPLES][random.randint(0, len(example[POS_EXAMPLES]) - 1)])
                else:
                    positives.append(example[POS_EXAMPLES])

                if isinstance(example[PART_POS_EXAMPLES], list):
                    partially_positives.append(
                        example[PART_POS_EXAMPLES][random.randint(0, len(example[PART_POS_EXAMPLES]) - 1)]
                    )
                else:
                    partially_positives.append(example[PART_POS_EXAMPLES])

                if isinstance(example[NEG_EXAMPLES], list):
                    negatives.append(example[NEG_EXAMPLES][random.randint(0, len(example[NEG_EXAMPLES]) - 1)])
                else:
                    negatives.append(example[NEG_EXAMPLES])

        return cls(anchors, positives, partially_positives, negatives, all_examples=examples, **kwargs)

    def _reset_examples(self):
        self._epoch_counter += 1  # increase epoch counter

        # Every N_EPOCHS_RESET epochs, reset the positive, negative and partially positive examples by resampling them
        if self._all_examples is not None and self._epoch_counter % self.N_EPOCHS_RESET_EXAMPLES == 0:
            anchors = []
            positives = []
            partially_positives = []
            negatives = []

            # noinspection PyTypeChecker
            for example in tqdm(self._all_examples, desc="Re-sampling examples for QuadrupletEvaluator..."):
                if isinstance(example, tuple):
                    example, _ = example
                if isinstance(example, InputExample):
                    # Convert to dictionary
                    texts = example.texts

                    anchors.append(texts[0])
                    positives.append(texts[1])
                    partially_positives.append(texts[2])
                    negatives.append(texts[3])
                else:

                    # Get the anchor
                    anchors.append(example[REFERENCE_EXAMPLE])

                    # Get one random positive, negative and partially positive
                    if isinstance(example[POS_EXAMPLES], list):
                        positives.append(example[POS_EXAMPLES][random.randint(0, len(example[POS_EXAMPLES]) - 1)])
                    else:
                        positives.append(example[POS_EXAMPLES])

                    if isinstance(example[PART_POS_EXAMPLES], list):
                        partially_positives.append(
                            example[PART_POS_EXAMPLES][random.randint(0, len(example[PART_POS_EXAMPLES]) - 1)]
                        )
                    else:
                        partially_positives.append(example[PART_POS_EXAMPLES])

                    if isinstance(example[NEG_EXAMPLES], list):
                        negatives.append(example[NEG_EXAMPLES][random.randint(0, len(example[NEG_EXAMPLES]) - 1)])
                    else:
                        negatives.append(example[NEG_EXAMPLES])

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
        glob_accuracy = (((1 - self._gamma) * pos_part_accuracy + self._gamma * part_neg_accuracy) + pos_neg_accuracy)/2

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


def euclidean_score(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    return 1 / (1 + torch.cdist(a, b, p=2))


def create_ir_evaluation_set(dataset: QuadrupletDataset,
                             n_queries: int = 20,
                             out_path: str = IR_EVALUATION_PATH,
                             use_pos: bool = False,
                             use_part_pos: bool = False,
                             use_cross_encoder: bool = True,
                             add_part_pos_corpus: bool = True) -> Dict[str, Dict[str, Union[str, List[str], Set[str]]]]:
    # If another ir_evaluation_set with the same split exists, then load it without recreating it
    if os.path.exists(out_path):
        with open(out_path, "r") as fp:
            ir_evaluation_set = json.load(fp)
            if ir_evaluation_set["random_seed"] == RANDOM_SEED:
                # Convert the "relevant" lists to sets as required by InformationRetrievalEvaluator
                for key in ir_evaluation_set["relevant"]:
                    ir_evaluation_set["relevant"][key] = set(ir_evaluation_set["relevant"][key])

                # Log info about the found relevant sentences
                n_relevant = []
                for q in ir_evaluation_set["queries"]:
                    n_relevant.append(len(ir_evaluation_set["relevant"][q]))
                n_relevant = np.array(n_relevant)
                print(f"\nNumber of relevant examples for each query: {n_relevant}")
                print(f"Average number of relevant examples for each query: {np.mean(n_relevant)}")
                print(f"Quantiles for the number of relevant examples for each query:"
                      f" {np.quantile(n_relevant, q=[0, 0.25, 0.5, 0.75, 1.0])}")
                return ir_evaluation_set

    # Select random indices to make the queries
    indexes = set(random.choices(list(range(0, len(dataset))), k=n_queries))

    ir_evaluation_set = {
        "queries": {},
        "corpus": {},
        "relevant": {}
    }
    query_idx = 0
    corpus_idx = 0
    progress_bar = tqdm(range(0, len(dataset)), desc="Creating information retrieval evaluation set...")
    # noinspection PyTypeChecker
    for idx, instance in enumerate(dataset):

        # If the current instance has been selected
        if idx in indexes:
            query_example = dataset[idx][REFERENCE_EXAMPLE]
            # Randomly choose if doing variations on the query (to enhance evaluation variability)
            query_example = generate_variations(query_example, n=1)
            if isinstance(query_example, list):
                query_example = query_example[0]

            # Use the reference example as query
            ir_evaluation_set["queries"][str(query_idx)] = query_example

            # Add the positives and partially positives to relevant items
            ir_evaluation_set["relevant"][str(query_idx)] = []
            for example in dataset[idx][POS_EXAMPLES]:
                ir_evaluation_set["corpus"][str(corpus_idx)] = example

                # Add to the relevant set if required explicitly
                if use_pos:
                    ir_evaluation_set["relevant"][str(query_idx)].append(str(corpus_idx))
                corpus_idx += 1
            for example in dataset[idx][PART_POS_EXAMPLES]:
                if add_part_pos_corpus:
                    ir_evaluation_set["corpus"][str(corpus_idx)] = example

                # Add to the relevant set if required explicitly
                if add_part_pos_corpus and use_part_pos:
                    ir_evaluation_set["relevant"][str(query_idx)].append(str(corpus_idx))
                corpus_idx += 1

            # Increment the query index
            query_idx += 1

        else:
            # Add only the reference/positives/partially positives to the corpus
            ir_evaluation_set["corpus"][str(corpus_idx)] = dataset[idx][REFERENCE_EXAMPLE]
            corpus_idx += 1
            for example in dataset[idx][POS_EXAMPLES]:
                ir_evaluation_set["corpus"][str(corpus_idx)] = example
                corpus_idx += 1

            if add_part_pos_corpus:
                for example in dataset[idx][PART_POS_EXAMPLES]:
                    ir_evaluation_set["corpus"][str(corpus_idx)] = example
                    corpus_idx += 1

        # Update the progress bar
        progress_bar.update(n=1)
        desc = f"Creating information retrieval evaluation set. Processed queries: {query_idx}/{n_queries}, " \
               f"Corpus index: {corpus_idx}"
        progress_bar.set_description(desc=desc)
    progress_bar.close()

    if use_cross_encoder:
        for q in tqdm(ir_evaluation_set["queries"], desc="Generating relevant examples using cross-encoder"):
            couples = [
                [ir_evaluation_set["queries"][q], ir_evaluation_set["corpus"][c]] for c in ir_evaluation_set["corpus"]
            ]
            scores = cross_encoder_singleton.predict(couples)
            for i, s in enumerate(scores):
                if s >= SIMILARITY_THRESHOLD:
                    ir_evaluation_set["relevant"][str(q)].append(str(i))

    # Log info about the found relevant sentences
    n_relevant = []
    for q in ir_evaluation_set["queries"]:
        n_relevant.append(len(ir_evaluation_set["relevant"][q]))
    n_relevant = np.array(n_relevant)
    print(f"\nNumber of relevant examples for each query: {n_relevant}")
    print(f"Average number of relevant examples for each query: {np.mean(n_relevant)}")
    print(f"Quantiles for the number of relevant examples for each query:"
          f" {np.quantile(n_relevant, q=[0, 0.25, 0.5, 0.75, 1.0])}")

    ir_evaluation_set["random_seed"] = RANDOM_SEED
    with open(out_path, "w") as fp:
        json.dump(ir_evaluation_set, fp, indent=2)

    for key in ir_evaluation_set["relevant"]:
        # Convert the "relevant" lists to sets as required by InformationRetrievalEvaluator
        ir_evaluation_set["relevant"][key] = set(ir_evaluation_set["relevant"][key])

    return ir_evaluation_set


def get_sequential_evaluator(dataset: QuadrupletDataset,
                             loss: GammaQuadrupletLoss,
                             evaluation_queries_path: Optional[str] = None,
                             no_transform_dataset: Optional[Union[QuadrupletDataset, Subset]] = None,
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
                             additional_model_kwargs: Optional[List[str]] = None,
                             additional_loss_kwargs: Optional[List[str]] = None,
                             use_amp: bool = False) -> SequentialEvaluator:
    evaluation_queries = None
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
    if evaluation_queries is None and no_transform_dataset is not None:
        evaluation_queries = create_ir_evaluation_set(no_transform_dataset)
    # elif evaluation_queries_path is None:
    #    evaluation_queries = None

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
    # data_loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=data_loader_sampler)
    quadruplet_loss_evaluator = QuadrupletLossEvaluator(
        quadruplet_dataset=dataset,
        quadruplet_loss=loss,
        additional_model_kwargs=additional_model_kwargs,
        additional_loss_kwargs=additional_loss_kwargs,
        use_amp=use_amp,
        batch_size=batch_size
    )
    evaluators.append(quadruplet_loss_evaluator)

    return SequentialEvaluator(evaluators=evaluators)
