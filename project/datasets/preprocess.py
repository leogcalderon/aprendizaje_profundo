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

def preprocess_inputs(question, context, answer, tokenizer, max_len):
    """
    Devuelve tensores listos para alimentar el modelo
    Parameters:
    -----------
    inputs_ids : str
    answer : str
    tokenizer : transformers.BertTokenizer
    max_len : int
    """
    answer = tokenizer(answer, add_special_tokens=False)['input_ids']
    
    # Por ahora, si el input tokenizado es > 512 tokens, solo tomamos
    # los 512 primeros tokens, ver si es mejor tirarlos, o hacer ventanas.
    # ya que la respuesta puede estar justo en el corte.
    
    input_ids = tokenizer(question, context)['input_ids'][:max_len]
    spans = search_idxs(input_ids, answer)

    return {
        'input_ids': input_ids,
        'spans': spans
    }

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
                    dataset.append(preprocess_inputs(
                        question,
                        context,
                        answers[0]['text'],
                        tokenizer,
                        max_len
                    ))

                # De lo contrario, procesar un ejemplo por respuesta
                else:
                    for answer in answers:
                        dataset.append(preprocess_inputs(
                            question,
                            context,
                            answer['text'],
                            tokenizer,
                            max_len
                        ))

    if save:
        with open(save, 'w') as file:
            json.dump({'data': dataset}, file)

    return dataset

if __name__ == '__main__':
    dataset_path = sys.argv[1]
    if len(sys.argv) > 2:
        save = sys.argv[2]
    preprocess_dataset(dataset_path, save=save)
