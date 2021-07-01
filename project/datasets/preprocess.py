from tqdm import tqdm
from transformers import BertTokenizer
import json
import sys

def search_idxs(input_ids, answer):
    """
    Busca los indices donde comienza y termina la respuesta
    dentro de los tokens de entrada. En el caso de no encontrar la respuesta
    devuelve [0, 0] como indices.

    Parameters:
    -----------
    input_ids : list
    answer : list
    """
    spans = [
        (i, i + len(answer))
        for i in range(len(input_ids) - len(answer) + 1)
        if input_ids[i:i + len(answer)] == answer
    ]

    return spans[0] if len(spans) > 0 else [0, 0]

def preprocess_inputs(question, context, answer, tokenizer, max_len, chunk_size=384):
    """
    Devuelve los tokens de pregunta y contexto junto con los
    indices de comienzo y fin de respuesta

    Parameters:
    -----------
    question : str
    context : str
    answer : str
    tokenizer : transformers.BertTokenizer
    max_len : int
    chunk_size : int
    """
    answer = tokenizer(answer, add_special_tokens=False)['input_ids']
    question = tokenizer(question)['input_ids']
    context = tokenizer(context, add_special_tokens=False)['input_ids']

    # Si la [CLS] pregunta [SEP] contexto [SEP] son menores a 384
    # generamos un ejemplo
    if len(question + context + [tokenizer.sep_token_id]) <= chunk_size:
        input_ids = (question + context + [tokenizer.sep_token_id])
        spans = search_idxs(input_ids, answer)

        return {
            'input_ids': input_ids,
            'spans': spans
        }

    # Si superan 384, devolvemos una lista con los chunks preprocesados de 384
    else:
        chunks = []
        for i in range(0, len(question + context + [tokenizer.sep_token_id]), chunk_size - len(question)):
            input_ids = (
                question +
                context[i:i + chunk_size - len(question) - 1] +
                [tokenizer.sep_token_id]
            )

            spans = search_idxs(input_ids, answer)

            chunks.append({
                'input_ids': input_ids,
                'spans': spans
            })

        return chunks


def preprocess_dataset(dataset_path, tokenizer='distilbert-base-cased', max_len=512, save=None):
    """
    Preprocesa un archivo de dataset

    Parameters:
    -----------
    dataset_path : str
    max_len : int
    tokenizer : str
        transformers tokenizer
    save : str
        path/to/save/preprocessed/dataset

    Returns:
    -------
    list
    """
    tokenizer = BertTokenizer.from_pretrained(tokenizer)
    dataset = []

    with open(dataset_path, 'r') as file:
        data = json.load(file)['data']

    for example in tqdm(data, f'Preprocesando archivo {dataset_path.split("/")[-1]}'):
        for paragraph in example['paragraphs']:
            context = paragraph['context']
            qas_list = paragraph['qas']

            for qa in qas_list:
                answers = qa['answers']
                question = qa['question']

                # Si todas las respuestas son iguales, solo procesar un ejemplo
                if all([answers[0]['text'] == answer['text'] for answer in answers]):
                    preprocessed_inputs = preprocess_inputs(
                        question,
                        context,
                        answers[0]['text'],
                        tokenizer,
                        max_len
                    )

                    # Depende de si prepreocess inputs nos dio la lista de
                    # chunks o un solo ejemplo
                    if isinstance(preprocessed_inputs, dict):
                        dataset.append(preprocessed_inputs)
                    else:
                        dataset += preprocessed_inputs

                # De lo contrario, procesar un ejemplo por respuesta
                else:
                    for answer in answers:
                        preprocessed_inputs = preprocess_inputs(
                            question,
                            context,
                            answer['text'],
                            tokenizer,
                            max_len
                        )

                    # Depende de si prepreocess inputs nos dio la lista de
                    # chunks o un solo ejemplo
                    if isinstance(preprocessed_inputs, dict):
                        dataset.append(preprocessed_inputs)
                    else:
                        dataset += preprocessed_inputs

    if save:
        with open(save, 'w') as file:
            json.dump({'data': dataset}, file)

    return dataset
