import torch
from tqdm import tqdm
from collections import Counter

def prepare_inputs(example, tokenizer, chunk_size=384):
    """
    Crea los tensores de entrada
    [tokens, mascaras de atencion, spans, tipo de token]
    Parameters:
    -----------
    example : dict
    tokenizer : transformers.BertTokenizer
    chunk_size : int

    Returns:
    --------
    dict
    """
    start_context = example['input_ids'].index(tokenizer.sep_token_id) + 1

    token_type_ids = (
        [0] * start_context +
        [1] * (len(example['input_ids']) - start_context) +
        [0] * (chunk_size - len(example['input_ids']))
    )

    attention_mask = (
        [1] * len(example['input_ids']) +
        [0] * (chunk_size - len(example['input_ids']))
    )

    input_ids = (
        example['input_ids'] +
        [tokenizer.pad_token_id] * (chunk_size - len(example['input_ids']))
    )

    return {
        'input_ids': torch.Tensor(input_ids).int(),
        'start_idx': torch.tensor([example['spans'][0]]).int(),
        'end_idx': torch.tensor([example['spans'][1]]).int(),
        'token_type_ids': torch.Tensor(token_type_ids).int(),
        'attention_mask': torch.Tensor(attention_mask).int()
    }

def get_prediction(model, example, tokenizer, device, max_len=512):
    """
    Predice la respuesta a partir de un
    ejemplo del dataset preprocesado.

    Parameters:
    -----------
    model : nn.Module
    example : dict
    tokenizer : transformers.BertTokenizer
    max_len : int

    Returns:
    --------
    str
    """
    model.to(device)
    model.eval()
    input = prepare_inputs(example, tokenizer)

    start, end = model(
        input['input_ids'].to(device).unsqueeze(0),
        input['token_type_ids'].to(device).unsqueeze(0),
        input['attention_mask'].to(device).unsqueeze(0)
    )

    answer_ids = input['input_ids'][start.argmax(1).item():end.argmax(1).item()]
    return tokenizer.decode(answer_ids)

def compute_exact_match(pred, truth):
    """
    Calcula EM a partir de dos strings
    """
    return int(pred == truth)

def compute_f1(pred, truth):
    """
    Calcula F1 a partir de dos strings
    """
    pred_tokens = pred.split()
    truth_tokens = truth.split()

    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    common_tokens = set(pred_tokens) & set(truth_tokens)

    if len(common_tokens) == 0:
        return 0

    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)

    return 2 * (prec * rec) / (prec + rec)

def get_scores(model, dataset, tokenizer, device):
    """
    Calcula exact match y F1

    Parameters:
    -----------
    model : nn.Module
    dataset : dict
    tokenizer : transformers.BertTokenizer
    device : str

    Returns:
    --------
    tuple
    """
    metrics = {'EM': [], 'F1': []}
    model.eval()

    with torch.no_grad():
        for example in tqdm(dataset, 'Evaluando resultados'):
            s, e = example['spans']
            pred = get_prediction(model, example, tokenizer, device)
            true = tokenizer.decode(example['input_ids'][s:e])
            metrics['EM'].append(compute_exact_match(pred, true))
            metrics['F1'].append(compute_f1(pred, true))

    return (
        sum(metrics['EM']) / len(metrics['EM']),
        sum(metrics['F1']) / len(metrics['F1'])
    )

def calculate_class_weigths(dataset, encoding):
    """
    Calcula los pesos a usar en la funcion
    de perdida para las clases.

    Parameters:
    -----------
    dataset : list
    encoding : dict

    Returns:
    --------
    dict
    """
    encoding_ = {v: k for k, v in encoding.items()}
    class_weigths = []
    for example in dataset:
        class_ = encoding_[example['domain']]
        class_weigths.append(class_)

    class_weigths = dict(Counter(class_weigths))
    max_class_n = class_weigths[max(class_weigths)]
    class_weigths = {k : max_class_n/v for k, v in class_weigths.items()}
    class_weigths = {encoding[k]: v for k, v in class_weigths.items()}

    cw = []
    for i in range(6):
        cw.append(class_weigths[i])

    return torch.Tensor(cw)
