import click
import torch
from torch import nn
from transformers import AdamW
from src import (
    dataloader,
    baseline_model,
    utils
)

@click.command()
@click.option(
    '--train-path',
    help='Directorio con los datasets de entrenamiento preprocesados',
    default='data/train'
)

@click.option(
    '--val-path',
    help='Directorio con los datasets de validación preprocesados',
    default='data/val'
)

@click.option(
    '--batch-size',
    help='Tamaño de batchs',
    default=32
)

@click.option(
    '--epochs',
    help='TÉpocas a entrenar',
    default=3
)

@click.option(
    '--lr',
    help='Tasa de aprendizaje',
    default=3e-5
)

@click.option(
    '--device',
    help='Dispositivo para entrenar modelo',
    default='cpu'
)

@click.option(
    '--save-path',
    help='Directorio para guardar los pesos del modelo',
    default=None
)

def main(train_path, val_path, batch_size, epochs, lr, device, save_path):
    """
    Entrenamiento de baseline

    Parameters:
    -----------
    train_path : str
        Directorio con los json de entrenamiento preprocesados
        por src.preprocess.preprocess_dataset

    val_path : str
        Directorio con los json de validación preprocesados
        por src.preprocess.preprocess_dataset

    epochs : int
    lr : float
    device : str
    save_path : str

    Returns:
    ---------
    tuple
    """
    shuffled_train_dataset = dataloader.create_shuffled_dataset(train_path)
    shuffled_val_dataset = dataloader.create_shuffled_dataset(val_path)
    train_dataloader = dataloader.get_dataloader(shuffled_train_dataset, batch_size=32)
    val_dataloader = dataloader.get_dataloader(shuffled_val_dataset, batch_size=32)

    model = baseline_model.BaseModel()
    loss = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)

    model, history = baseline.train_baseline(
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        loss,
        epochs,
        device,
    )

    if save_path:
        torch.save(model.state_dict(), save_path)

    return model, history

if __name__ == '__main__':
    _, history = main()
    print(history)
