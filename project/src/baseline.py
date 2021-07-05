import torch
import time
from torch import nn
from transformers import DistilBertModel

class BaseModel(nn.Module):
    """
    Baseline con Bert-base-cased pre entrenado y dos capas lineares
    para calcular el comienzo y final de la respuesta.
    """
    def __init__(self, bert='distilbert-base-cased', dropout=0.3):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(bert)
        self.emb_dim = self.bert.embeddings.word_embeddings.embedding_dim
        self.max_len = self.bert.embeddings.position_embeddings.num_embeddings

        self.start_mlp = nn.Linear(self.emb_dim, 1)
        self.end_mlp = nn.Linear(self.emb_dim, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, token_type_ids, attention_mask):

        # input_ids, token_type_ids, attention_mask = [batch_size, chunk_size]
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_state = bert_output.last_hidden_state
        # last_hidden_state = [batch_size, chunk_size, emb_dim]

        # Tirar a 0 todos los valores que son pregunta y padding.
        context_hidden_state = last_hidden_state * token_type_ids.unsqueeze(-1).repeat(1, 1, self.emb_dim)
        # context_hidden_state = [batch_size, chunk_size, emb_dim]

        # MLP para softmax del token de entrada y de salida
        start_logits = self.start_mlp(self.dropout(context_hidden_state))
        end_logits = self.end_mlp(self.dropout(context_hidden_state))
        # start_logits, end_logits = [batch_size, chunk_size, 1]

        return start_logits, end_logits

def train_baseline(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    loss,
    epochs,
    device,
    print_every=200):
    """
    Bucle de entrenamiento para el baseline

    Parameters:
    -----------
    model : nn.Module
    train_dataloader : torch.utils.data.DataLoader
    val_dataloader : torch.utils.data.DataLoader
    optimizer : torch.optim.Optimizer
    loss : nn._Loss
    epochs : int
    device : str
    print_every : int

    Returns:
    --------
    nn.Module, dict
    """
    model.to(device)
    history = {'train': [], 'val': []}

    for epoch in range(epochs):
        train_epoch_loss = 0
        val_epoch_loss = 0
        start_time = time.time()

        print('Entrenando modelo')
        for i, example in enumerate(train_dataloader):
            model.train()
            optimizer.zero_grad()

            input_ids, token_type_ids, attention_mask = (
                example['input_ids'].to(device),
                example['token_type_ids'].to(device),
                example['attention_mask'].to(device)
            )

            start_idx, end_idx = (
                example['start_idx'].long().to(device),
                example['end_idx'].long().to(device)
            )

            # Salida del modelo baseline
            start_logits, end_logits = model(input_ids, token_type_ids, attention_mask)

            # Promedio de las funciones de perdida
            # de los tokens de entrada y de salida
            l = (loss(start_logits, start_idx) + loss(end_logits, end_idx)) / 2

            l.backward()
            optimizer.step()
            train_epoch_loss += l.item()

            if i % print_every == (print_every - 1):
                current_time = time.time()
                elapsed_time = epoch_time(start_time, current_time)
                train_loss = train_epoch_loss / (i * train_dataloader.batch_size)
                print(f'[EPOCH: {epoch + 1} - BATCH: {i + 1}/{len(train_dataloader)}]')
                print(f'Perdida de entrenamiento actual: {train_loss}')
                print(f'Tiempo entrenando: {elapsed_time[0]}:{elapsed_time[1]} minutos')
                print('================================')

        history['train'].append(train_epoch_loss / (i * train_dataloader.batch_size))

        print('Evaluando modelo')
        for i, example in enumerate(val_dataloader):
            model.eval()
            start_time = time.time()

            with torch.no_grad():
                input_ids, token_type_ids, attention_mask = (
                    example['input_ids'].to(device),
                    example['token_type_ids'].to(device),
                    example['attention_mask'].to(device)
                )

                start_idx, end_idx = (
                    example['start_idx'].long().to(device),
                    example['end_idx'].long().to(device)
                )

                # Salida del modelo baseline
                start_logits, end_logits = model(input_ids, token_type_ids, attention_mask)
                l = (loss(start_logits, start_idx) + loss(end_logits, end_idx)) / 2
                val_epoch_loss += l.item()

                if i % print_every == (print_every - 1):
                    val_loss = val_epoch_loss / (i * val_dataloader.batch_size)
                    current_time = time.time()
                    elapsed_time = epoch_time(start_time, current_time)
                    print(f'[EPOCH: {epoch + 1} - BATCH: {i + 1}/{len(train_dataloader)}]')
                    print(f'Perdida de validacion actual: {val_loss}')
                    print(f'Tiempo validando: {elapsed_time[0]}:{elapsed_time[1]} minutos')
                    print('================================')

            history['val'].append(val_epoch_loss / (i * val_dataloader.batch_size))

    return model, history
