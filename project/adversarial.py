import click
import torch
from torch import nn
from transformers import AdamW
from src import (
    dataloader,
    adversarial_model,
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
    '--device',
    help='Dispositivo para entrenar modelo',
    default='cpu'
)

@click.option(
    '--save-path',
    help='Directorio para guardar los pesos del modelo',
    default=None
)


@click.option(
    '--qa-lr',
    help='Tasa de aprendizaje de QA',
    default=3e-5
)


@click.option(
    '--d-lr',
    help='Tasa de aprendizaje de discirminador',
    default=5e-5
)

def main(train_path, val_path, batch_size, epochs, device, save_path, qa_lr, d_lr):
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
    shuffled_train_dataset, _ = dataloader.create_shuffled_dataset(train_path, adversarial_flag=True)
    shuffled_val_dataset, _ = dataloader.create_shuffled_dataset(val_path, adversarial_flag=True)
    train_dataloader = dataloader.get_dataloader(shuffled_train_dataset, batch_size=32)
    val_dataloader = dataloader.get_dataloader(shuffled_val_dataset, batch_size=32)

    adv_lambda = 0.01
    qa = adversarial_model.BaseModel()
    discriminator = adversarial_model.DomainDiscriminator()

    qa_loss = nn.CrossEntropyLoss()
    adv_qa_loss = nn.KLDivLoss(reduction='batchmean')
    adv_d_loss = nn.NLLLoss()

    qa_optimizer = AdamW(qa.parameters(), lr=qa_lr)
    d_optimizer = AdamW(discriminator.parameters(), lr=d_lr)

    qa, discriminator, history = adversarial_model.train_adversarial(
        qa, discriminator, train_dataloader, val_dataloader,
        qa_optimizer, d_optimizer, qa_loss, adv_qa_loss,
        adv_d_loss, epochs, adv_lambda, device, print_every=100
    )

    if save_path:
        torch.save(qa.state_dict(), save_path + '_qa')
        torch.save(discriminator.state_dict(), save_path + '_d')

    return qa, discriminator, history

if __name__ == '__main__':
    _, _, history = main()
    print(history)
