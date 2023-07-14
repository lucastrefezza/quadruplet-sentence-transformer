import random
from typing import Tuple, Any, Optional, List, Dict, Union
import torch
from sentence_transformers import SentenceTransformer, InputExample
from dataset.constants import REFERENCE_EXAMPLE, POS_EXAMPLES, PART_POS_EXAMPLES, NEG_EXAMPLES
from models.losses.losses import QuadrupletLoss


class QuadrupletSentenceTransformerLossModel(torch.nn.Module):
    def __init__(self,
                 st_model: SentenceTransformer,
                 quadruplet_loss: QuadrupletLoss,
                 additional_model_kwargs: Optional[List[str]] = None,
                 additional_loss_kwargs: Optional[List[str]] = None):
        super().__init__()
        self._st_model = st_model
        self._quadruplet_loss = quadruplet_loss
        self.__additional_model_kwargs = additional_model_kwargs
        self.__additional_loss_kwargs = additional_loss_kwargs

    # noinspection PyUnusedLocal
    def forward(self, features: Union[Dict, List[Dict]], labels=None) -> Tuple[torch.Tensor, ...]:

        if isinstance(features, Dict):
            reference_example = features[REFERENCE_EXAMPLE]
            pos_example = features[POS_EXAMPLES]
            part_pos_example = features[PART_POS_EXAMPLES]
            neg_example = features[NEG_EXAMPLES]
        else:
            reference_example = features[0]
            pos_example = features[1]
            part_pos_example = features[2]
            neg_example = features[3]

        # Additional model arguments
        additional_model_kwargs = {}
        if self.__additional_model_kwargs is not None:
            for kwarg in self.__additional_model_kwargs:
                additional_model_kwargs[kwarg] = features[kwarg]

        # Call the model on the instance examples
        reference_example = self._st_model(
            reference_example,
            **additional_model_kwargs
        )['sentence_embedding']

        pos_example = self._st_model(
            pos_example,
            **additional_model_kwargs
        )['sentence_embedding']

        part_pos_example = self._st_model(
            part_pos_example,
            **additional_model_kwargs
        )['sentence_embedding']

        neg_example = self._st_model(
            neg_example,
            **additional_model_kwargs
        )['sentence_embedding']

        # Additional loss kwargs
        additional_loss_kwargs = {}
        if self.__additional_loss_kwargs is not None:
            for kwarg in self.__additional_loss_kwargs:
                additional_loss_kwargs[kwarg] = features[kwarg]

        # Compute loss
        loss = self._quadruplet_loss(
            x_anchor=reference_example,
            x_pos=pos_example,
            x_part=part_pos_example,
            x_neg=neg_example,
            **additional_loss_kwargs
        )

        return loss


# Ensures compatibility with InputExample smart batching collate function required by SentenceTransformer


def to_input_example(
        instance: Union[Dict[str, Union[str, List[str]]], Tuple[Dict[str, Union[str, List[str]]], torch.Tensor]]
) -> InputExample:
    """
    Transforms the given instance from dictionary of REFERENCE_EXAMPLE, POS_EXAMPLES, PART_POS_EXAMPLES and NEG_EXAMPLES
    to a SentenceTransformer-supported InputExample, where the 'texts' attribute is a list of strings, containing the
    examples in the above order.

    :param instance: the dictionary representing the instance.
    :return: an InputExample representing the instance.
    """
    instance = select_single_example(instance)
    texts = [instance[REFERENCE_EXAMPLE], instance[POS_EXAMPLES], instance[PART_POS_EXAMPLES], instance[NEG_EXAMPLES]]

    return InputExample(texts=texts)


# Ensures compatibility with SentenceTransformer's fit() expecting a label
def add_empty_label(instance: Any) -> Tuple[Any, torch.Tensor]:
    return instance, torch.tensor(0)


def select_single_example(
        instance: Union[Dict[str, Union[str, List[str]]], Tuple[Dict[str, Union[str, List[str]]], torch.Tensor]]
) -> Union[Dict[str, str], Tuple[Dict[str, str], torch.Tensor]]:
    """
    Ensures that a single example is returned for each instance (one positive, one negative, one partially positive),
    selecting a single example for each type.

    :param instance: the instance to select the example from, either a dictionary with strings/lists of strings, or the
        same dictionary coupled with a label tensor.

    :return: the given instance with a single example selected for each example type.
    """
    # Check if any labels are given
    labels = None
    if isinstance(instance, tuple):
        instance, labels = instance

    # Select a single example among all the example types
    if isinstance(instance[POS_EXAMPLES], list):
        instance[POS_EXAMPLES] = instance[POS_EXAMPLES][random.randint(0, len(instance[POS_EXAMPLES]))]
    if isinstance(instance[NEG_EXAMPLES], list):
        instance[NEG_EXAMPLES] = instance[NEG_EXAMPLES][random.randint(0, len(instance[NEG_EXAMPLES]))]
    if isinstance(instance[POS_EXAMPLES], list):
        instance[PART_POS_EXAMPLES] = instance[PART_POS_EXAMPLES][random.randint(0, len(instance[PART_POS_EXAMPLES]))]

    # Return labels if any labels are given
    if labels is not None:
        return instance, labels
    return instance
