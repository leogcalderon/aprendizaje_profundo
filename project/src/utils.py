import torch
from dataloader import prepare_inputs
from tqdm import tqdm

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
    # TODO: implementar chunks en caso de que
    # la pregunta + contexto supere los 384
    # Casos:
    #   * no hay respuestas en ningun chunk
    #   * solo hay una respuesta (devolver esa)
    #   * mas de una respuesta (verificar si son iguales, y si no ? QUE HACER?)

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

def get_scores(model, dataset, tokenizer):
    """
    Calcula exact match y F1

    Parameters:
    -----------
    model : nn.Module
    dataset : dict
    tokenizer : transformers.BertTokenizer

    Returns:
    --------
    tuple
    """
    metrics = {'EM': [], 'F1': []}
    model.eval()

    with torch.no_grad():
        for example in tqdm(squad_train, 'Evaluando resultados'):
            s, e = example['spans']
            pred = get_prediction(model, example, tokenizer)
            true = tokenizer.decode(example['input_ids'][s:e])
            metrics['EM'].append(compute_exact_match(pred, true))
            metrics['F1'].append(compute_f1(pred, true))

    return (
        sum(metrics['EM']) / len(metrics['EM']),
        sum(metrics['F1']) / len(metrics['F1'])
    )

