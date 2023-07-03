from typing import Optional, List
import torch
from transformers import MarianMTModel, MarianTokenizer
from utils.synchronization import synchronized


# MarianMTModel and MarianTokenizer singletons
first_model_singleton: Optional[MarianMTModel] = None
first_model_tkn_singleton: Optional[MarianTokenizer] = None
second_model_singleton: Optional[MarianMTModel] = None
second_model_tkn_singleton: Optional[MarianTokenizer] = None


@synchronized
def get_first_model(first_model_name: str = 'Helsinki-NLP/opus-mt-en-fr') -> tuple[MarianMTModel, MarianTokenizer]:
    """Gets the backtranslation first model singleton instance and corresponding tokenizer."""
    global first_model_singleton
    global first_model_tkn_singleton

    if first_model_singleton is None:
        # Get the device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the pretrained model based on the name
        first_model_singleton = MarianMTModel.from_pretrained(first_model_name).to(device)

        # Get the tokenizer
        first_model_tkn_singleton = MarianTokenizer.from_pretrained(first_model_name)

    return first_model_singleton, first_model_tkn_singleton


@synchronized
def get_second_model(second_model_name: str = 'Helsinki-NLP/opus-mt-fr-en') -> tuple[MarianMTModel, MarianTokenizer]:
    """Gets the backtranslation second model singleton instance and corresponding tokenizer."""
    global second_model_singleton
    global second_model_tkn_singleton

    if second_model_singleton is None:
        # Get the device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the pretrained model based on the name
        second_model_singleton = MarianMTModel.from_pretrained(second_model_name).to(device)

        # Get the tokenizer
        second_model_tkn_singleton = MarianTokenizer.from_pretrained(second_model_name)

    return second_model_singleton, second_model_tkn_singleton


def get_backtranslation_models(first_model_name: str = 'Helsinki-NLP/opus-mt-en-fr',
                               second_model_name: str = 'Helsinki-NLP/opus-mt-fr-en') -> \
        tuple[MarianMTModel, MarianTokenizer, MarianMTModel, MarianTokenizer]:
    """Gets the backtranslation models/tokenizers."""
    # Get the first model/tokenizer
    first_model, first_model_tkn = get_first_model(first_model_name)

    # Get the second model/tokenizer
    second_model, second_model_tkn = get_second_model(second_model_name)

    return first_model, first_model_tkn, second_model, second_model_tkn


def format_batch_texts(language_code, batch_texts) -> List[str]:
    """Formats the texts for the backtranslation adding the language code as a prefix."""
    formated_batch = [">>{}<< {}".format(language_code, text) for text in batch_texts]

    return formated_batch


def perform_translation(batch_texts, model, tokenizer, language="fr") -> List[str]:
    """Translates the given text to the given language with the given model."""
    # Prepare the text data into appropriate format for the model
    formated_batch_texts = format_batch_texts(language, batch_texts)

    '''print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')'''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Generate translation using model
    translated = model.generate(**tokenizer(formated_batch_texts, return_tensors="pt", padding=True).to(device))

    '''print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'B')
    print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'B')'''

    # Convert the generated tokens indices back into text
    translated_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

    return translated_texts


def perform_back_translation(batch_texts, original_language="en", temporary_language="fr") -> List[str]:
    """Performs backtranslation using the given languages on the given texts."""
    first_model, first_model_tkn, second_model, second_model_tkn = get_backtranslation_models()

    # Translate from Original to Temporary Language
    tmp_translated_batch = perform_translation(batch_texts, first_model,
                                               first_model_tkn,
                                               temporary_language)

    # Translate Back to English
    back_translated_batch = perform_translation(tmp_translated_batch,
                                                second_model,
                                                second_model_tkn,
                                                original_language)

    # Return The Final Result
    return back_translated_batch
