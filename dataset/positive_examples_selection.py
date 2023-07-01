import random
import math
from typing import Union, List
import nlpaug.augmenter.word as naw
from dataset.backtranslation import perform_back_translation
from sentence_transformers import SentenceTransformer, util
import torch
from dataset.constants import *

TOP_K_BACKUP: final = 2


def get_embedder(model_name: str = 'all-MiniLM-L6-v2') -> SentenceTransformer:
    return SentenceTransformer(model_name)


def back_translation(captions: List[str]) -> List[str]:
    return perform_back_translation(captions)


def compute_cosine_scores(caption: str,
                          captions: List[str]) -> torch.Tensor:
    embedder = get_embedder()
    captions_embeddings = embedder.encode(captions, convert_to_tensor=True)
    caption_embedding = embedder.encode(caption, convert_to_tensor=True)
    cos_scores = util.cos_sim(caption_embedding, captions_embeddings)[0]
    return cos_scores


def pop_random_caption(captions: List[str],
                       forbidden: set[str] = frozenset([]),
                       max_iterations: int = 50) -> str:
    if max_iterations == 0:
        raise ValueError(f"max_iterations must be > 0 or < 0, {max_iterations} "
                         "given.")

    caption = None
    iter_counter = 0
    indexes = range(0, len(captions))

    # Until a caption is selected
    while caption is None and iter_counter < max_iterations:
        # Choose a random caption
        i = random.choice(indexes)

        # If it is not in the forbidden list, select it and remove from caption list
        if captions[i] not in forbidden:
            caption = captions[i]
            captions[i], captions[-1] = captions[-1], captions[i]  # O(1) remove
            captions.pop()

        if max_iterations > 0:
            iter_counter += 1

    # If no non-duplicated captions are found in max iter, return a duplicated one
    if caption is None:
        i = random.choice(indexes)
        caption = captions[i]

    return caption


def select_positive_examples(captions: List[str],
                             threshold: float = SIMILARITY_THRESHOLD,
                             n_examples: int = N_EXAMPLES,
                             augment: bool = True,
                             augment_insert: bool = False,
                             return_similarities: bool = False,
                             max_attempts: int = MAX_ATTEMPTS) -> \
        Union[List[str], tuple[List[str], str, torch.Tensor]]:
    if not 0 < max_attempts <= len(captions):
        raise ValueError(f"max_attempts must be between 1 and the number of "
                         f"captions {len(captions)}. {max_attempts} given.")

    selected_positive_examples = 0  # selected positive example count
    attempt_count = 0  # attempt count
    captions_tmp = []
    already_attempted = set()  # forbidden list to not attempt the same two times

    # Choose a new reference caption and add it to the forbidden list
    caption = pop_random_caption(captions, forbidden=already_attempted)
    already_attempted.add(caption)

    cos_scores = None
    # While we don't select enough positive samples or finish attempts
    while selected_positive_examples == 0 and attempt_count < max_attempts:

        # Compute cosine similarity between the chosen caption and the other ones
        cos_scores = compute_cosine_scores(caption=caption, captions=captions)

        # Discard too distant captions
        captions_tmp = []
        for i, capt in enumerate(captions):
            if cos_scores[i] >= threshold:
                captions_tmp.append(capt)

        # If no captions have been selected
        selected_positive_examples = len(captions_tmp)
        if selected_positive_examples == 0:
            # Increase attempt count
            attempt_count += 1

            # Choose a new reference caption
            caption_new = pop_random_caption(captions, already_attempted)

            # Re-add the previous one to the caption list
            captions.append(caption)

            # Add the new selected caption to the forbidden list (cannot be selected)
            already_attempted.add(caption_new)
            caption = caption_new

    # If no captions have been selected after max attempts, choose the top-k most
    # similar found
    if len(captions_tmp) == 0:
        _, topk_indexes = torch.topk(cos_scores, k=TOP_K_BACKUP)
        for idx in topk_indexes:
            captions_tmp.append(captions[idx])

    captions = captions_tmp

    # (DO NOT) Add the reference caption to the selected positive examples list
    # captions.append(caption)

    # If there aren't enough positive example captions, replicate some and augment
    n_lacking_captions = n_examples - len(captions)
    if n_lacking_captions > 0:

        # If required, perform backtranslation on each positive example caption
        if augment:
            new_captions = back_translation(captions)
            if augment_insert:
                aug = naw.ContextualWordEmbsAug(
                    model_path='roberta-base',
                    action="insert",
                    aug_min=0,
                    aug_max=2
                )
                new_captions = aug.augment(new_captions)
            aug = naw.SynonymAug(
                aug_src='wordnet',
                aug_min=1,
                aug_max=MAX_WORDS_TO_REPLACE,
                stopwords=NO_REPLACE_WORDS
            )
            new_captions = aug.augment(new_captions)

        # Otherwise just replicate the existing ones
        else:
            new_captions = captions

        # Add the augmented/replicated captions to the positive example captions,
        # sampling without replacement
        new_captions = random.sample(new_captions,
                                     min(n_lacking_captions, len(new_captions)))

        # If not enough captions were selected, repeat augmented/replicated captions
        if len(new_captions) < n_lacking_captions:
            # current_len*x >= target_len -> x >= target_len/current_len > 1
            n_repeats = math.ceil(n_lacking_captions / len(new_captions)) - 1
            repeated = new_captions * n_repeats
            new_captions.extend(repeated)
            new_captions = new_captions[:n_lacking_captions]  # delete excess captions
        captions.extend(new_captions)

    if return_similarities:
        return captions, caption, cos_scores

    return captions
