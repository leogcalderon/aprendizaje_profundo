import os
import json
import torch
import random
from transformers import BertTokenizer
from torch.utils.data import DataLoader, Dataset

def load_processed_datasets(processed_path):
    """
    Parsea los json presentes en el path
    que contienen los datasets preprocesados

    Parameters:
    -----------
    processed_path : str

    Returns:
    --------
    dict
    """
    datasets = {}
    for file in os.listdir(processed_path):
        with open(os.path.join(processed_path, file), 'r') as f:
            datasets[file.split('.')[0]] = json.load(f)['data']

    return datasets

def create_shuffled_dataset(processed_path, adversarial_flag=False):
    """
    Lee los datasets en formato json y crea
    un dataset con los datos mezclados

    Parameters:
    -----------
    processed_path : str
    adversarial_flag : bool

    Returns:
    --------
    dict
    """
    datasets = load_processed_datasets(processed_path)
    shuffled_dataset = list()

    oodomain_datasets = [k for k, v in datasets.items() if len(v) < 500]

    for name, dataset in datasets.items():
        if adversarial_flag:
            if name in oodomain_datasets:
                for example in dataset:
                    example['oodomain'] = 1
            else:
                for example in dataset:
                    example['oodomain'] = 0

            shuffled_dataset += dataset

        else:
            shuffled_dataset += dataset

    random.shuffle(shuffled_dataset)
    return shuffled_dataset

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

class QADataset(Dataset):
    def __init__(self, data, tokenizer, chunk_size=384):
        self.data = data
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return prepare_inputs(self.data[idx], self.tokenizer)

def get_dataloader(
    dataset,
    batch_size,
    tokenizer_model='distilbert-base-cased',
    **kwargs):
    """
    Devuelve un dataloader de un dataset preprocesado
    Parameters:
    -----------
    dataset : list
    tokenizer_model : str

    Returns:
    --------
    torch.utils.data.DataLoader
    """
    tokenizer = BertTokenizer.from_pretrained(tokenizer_model)
    qa_dataset = QADataset(dataset, tokenizer)
    return DataLoader(qa_dataset, batch_size=batch_size, **kwargs)
