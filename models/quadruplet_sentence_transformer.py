import random
from typing import Tuple, Any, Optional, List, Dict, Union
import torch
from sentence_transformers import SentenceTransformer
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
    def forward(self, features: Dict, labels=None) -> Tuple[torch.Tensor, ...]:
        reference_example = features[REFERENCE_EXAMPLE]
        pos_example = features[POS_EXAMPLES]
        part_pos_example = features[PART_POS_EXAMPLES]
        neg_example = features[NEG_EXAMPLES]

        # Additional model arguments
        additional_model_kwargs = {}
        for kwarg in self.__additional_model_kwargs:
            additional_model_kwargs[kwarg] = features[kwarg]

        # TODO: choose one random positive/negative example
        # Call the model on the instance examples
        reference_example = self._st_model(
            reference_example,
            **additional_model_kwargs
        )

        pos_example = self._st_model(
            pos_example,
            **additional_model_kwargs
        )

        part_pos_example = self._st_model(
            part_pos_example,
            **additional_model_kwargs
        )

        neg_example = self._st_model(
            neg_example,
            **additional_model_kwargs
        )

        # Additional loss kwargs
        additional_loss_kwargs = {}
        for kwarg in self.__additional_loss_kwargs:
            additional_loss_kwargs[kwarg] = features[kwarg]

        # Compute loss
        loss = self._quadruplet_loss(
            x_anchor=reference_example,
            x_pos=pos_example,
            x_part_pos=part_pos_example,
            x_neg=neg_example,
            **additional_loss_kwargs
        )

        return loss


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
