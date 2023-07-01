from transformers import MarianMTModel, MarianTokenizer


def get_backtranslation_models(first_model_name: str = 'Helsinki-NLP/opus-mt-en-fr',
                               second_model_name: str = 'Helsinki-NLP/opus-mt-fr-en') -> \
        tuple[MarianMTModel, MarianTokenizer, MarianMTModel, MarianTokenizer]:
    # Load the pretrained model based on the name
    first_model = MarianMTModel.from_pretrained(first_model_name)

    # Get the tokenizer
    first_model_tkn = MarianTokenizer.from_pretrained(first_model_name)

    # Load the pretrained model based on the name
    second_model = MarianMTModel.from_pretrained(second_model_name)

    # Get the tokenizer
    second_model_tkn = MarianTokenizer.from_pretrained(second_model_name)

    return first_model, first_model_tkn, second_model, second_model_tkn


def format_batch_texts(language_code, batch_texts):
    formated_bach = [">>{}<< {}".format(language_code, text) for text in batch_texts]

    return formated_bach


def perform_translation(batch_texts, model, tokenizer, language="fr"):
    # Prepare the text data into appropriate format for the model
    formated_batch_texts = format_batch_texts(language, batch_texts)

    # Generate translation using model
    translated = model.generate(**tokenizer(formated_batch_texts, return_tensors="pt", padding=True))

    # Convert the generated tokens indices back into text
    translated_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

    return translated_texts


def perform_back_translation(batch_texts, original_language="en", temporary_language="fr"):

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
