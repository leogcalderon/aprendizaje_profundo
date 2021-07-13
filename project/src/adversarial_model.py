import time
import torch
from torch import nn
from torch.nn import functional as F
from transformers import DistilBertModel

def epoch_time(start_time, end_time):
    """
    Calcula el tiempo transcurrido entre los
    dos argumentos.
    Snippet: TP8

    Parameters:
    -----------
    start_time : float
    end_time : float

    Returns:
    --------
    tuple
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

class DomainDiscriminator(nn.Module):
    """
    Discriminador para entrenamiento adversario.
    Misma arquitectura que el paper original
    """
    def __init__(self, input_dim=768, hidden_sizes=[768, 512, 512], dropout=0.1, classes=6):
        super().__init__()
        self.classes = classes
        layers = []
        for hidden_size in hidden_sizes:
            layers.append(nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
            input_dim = hidden_size
        layers.append(nn.Linear(hidden_size, classes))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
        x = self.layers[-1](x)
        return F.log_softmax(x, dim=1)

class BaseModel(nn.Module):
    """
    Baseline con Bert-base-cased pre entrenado y dos capas lineares
    para calcular el comienzo y final de la respuesta. Ademas
    devuelve el token cls para el entrenamiento adversario.
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

        # [CLS] token
        cls = last_hidden_state[:, 0]

        return cls, start_logits, end_logits

def train_adversarial(
    qa, discriminator, train_dataloader, val_dataloader,
    qa_optimizer, d_optimizer, qa_loss, adv_qa_loss,
    adv_d_loss, epochs, adv_lambda, device, print_every=200):
    """
    Bucle de entrenamiento adversario para el baseline
    """
    qa.to(device)
    discriminator.to(device)
    history = {
        'train': {
            'spans_loss': [],
            'qa_adv_loss': [],
            'discriminator_loss': []
        },
        'val': {
            'spans_loss': [],
            'qa_adv_loss': [],
            'discriminator_loss': []
        }
    }

    for epoch in range(epochs):
        train_qa_epoch_loss = train_d_adv_epoch_loss = train_qa_adv_epoch_loss = 0
        val_qa_epoch_loss = val_d_adv_epoch_loss = val_qa_adv_epoch_loss = 0

        start_time = time.time()

        print('Entrenando modelo')
        for i, example in enumerate(train_dataloader):

            qa.train()
            discriminator.train()
            qa_optimizer.zero_grad()
            d_optimizer.zero_grad()

            input_ids, token_type_ids, attention_mask = (
                example['input_ids'].to(device),
                example['token_type_ids'].to(device),
                example['attention_mask'].to(device)
            )

            start_idx, end_idx, domains = (
                example['start_idx'].long().to(device),
                example['end_idx'].long().to(device),
                example['domain'].long().to(device)
            )

            # Primero se entrena el sistema QA y luego el discriminador
            # Output de BERT. [CLS] token y logits de los MLP de comienzo y fin
            cls, start_logits, end_logits = qa(input_ids, token_type_ids, attention_mask)

            # Output del discriminador
            discriminator_pred = discriminator(cls)

            # Promedio de las funciones de perdida QA
            # de los tokens de entrada y de salida
            qa_l = (qa_loss(start_logits, start_idx) + qa_loss(end_logits, end_idx)) / 2

            # Perdida adversaria QA
            qa_adv_l = adv_qa_loss(discriminator_pred, torch.ones_like(discriminator_pred) / discriminator.classes)

            # Perdida total del QA
            total_l = qa_l + adv_lambda * qa_adv_l

            # Guardar valores de perdida
            train_qa_epoch_loss += qa_l.item()
            train_qa_adv_epoch_loss += qa_adv_l.item()
            history['train']['spans_loss'].append(qa_l.item())
            history['train']['qa_adv_loss'].append(qa_adv_l.item())

            total_l.backward()
            qa_optimizer.step()

            # Perdida del discriminador
            with torch.no_grad():
                cls, _, _ = qa(input_ids, token_type_ids, attention_mask)

            discriminator_pred = discriminator(cls)
            d_adv_l = adv_d_loss(discriminator_pred, domains.squeeze(-1))

            # Guardar valores de perdida
            history['train']['discriminator_loss'].append(d_adv_l.item())
            train_d_adv_epoch_loss += d_adv_l.item()

            d_adv_l.backward()
            d_optimizer.step()

            if i % print_every == (print_every - 1):
                current_time = time.time()
                elapsed_time = epoch_time(start_time, current_time)

                train_qa_loss = train_qa_epoch_loss / (i * train_dataloader.batch_size)
                train_qa_adv_loss = train_qa_adv_epoch_loss / (i * train_dataloader.batch_size)
                train_d_adv_loss = train_d_adv_epoch_loss / (i * train_dataloader.batch_size)

                print(f'[EPOCH: {epoch + 1} - BATCH: {i + 1}/{len(train_dataloader)}]')
                print(f'Perdida QA spans: {train_qa_loss}')
                print(f'Perdida QA Adv: {train_qa_adv_loss}')
                print(f'Perdida Discriminador: {train_d_adv_loss}')
                print(f'Tiempo entrenando: {elapsed_time[0]}:{elapsed_time[1]} minutos')
                print('================================')

        print('Validando modelo')
        for i, example in enumerate(val_dataloader):
            qa.eval()
            discriminator.eval()

            with torch.no_grad():
                input_ids, token_type_ids, attention_mask = (
                    example['input_ids'].to(device),
                    example['token_type_ids'].to(device),
                    example['attention_mask'].to(device)
                )

                start_idx, end_idx, domains = (
                    example['start_idx'].long().to(device),
                    example['end_idx'].long().to(device),
                    example['domain'].long().to(device)
                )

                # Salidas de modelos
                cls, start_logits, end_logits = qa(input_ids, token_type_ids, attention_mask)
                discriminator_pred = discriminator(cls)

                # Perdida total del QA
                qa_l = (qa_loss(start_logits, start_idx) + qa_loss(end_logits, end_idx)) / 2
                qa_adv_l = adv_qa_loss(discriminator_pred, torch.ones_like(discriminator_pred) / discriminator.classes)
                total_l = qa_l + adv_lambda * qa_adv_l

                val_qa_epoch_loss += qa_l.item()
                val_qa_adv_epoch_loss += qa_adv_l.item()
                history['val']['spans_loss'].append(qa_l.item())
                history['val']['qa_adv_loss'].append(qa_adv_l.item())

                # Perdida del discriminador
                cls, _, _ = qa(input_ids, token_type_ids, attention_mask)
                discriminator_pred = discriminator(cls)
                d_adv_l = adv_d_loss(discriminator_pred, domains.squeeze(-1))

                history['val']['discriminator_loss'].append(d_adv_l.item())
                val_d_adv_epoch_loss += d_adv_l.item()

                if i % print_every == (print_every - 1):
                    current_time = time.time()
                    elapsed_time = epoch_time(start_time, current_time)

                    val_qa_loss = val_qa_epoch_loss / (i * val_dataloader.batch_size)
                    val_qa_adv_loss = val_qa_adv_epoch_loss / (i * val_dataloader.batch_size)
                    val_d_adv_loss = val_d_adv_epoch_loss / (i * val_dataloader.batch_size)

                    print(f'[EPOCH: {epoch + 1} - BATCH: {i + 1}/{len(val_dataloader)}]')
                    print(f'Perdida QA spans: {val_qa_loss}')
                    print(f'Perdida QA Adv: {val_qa_adv_loss}')
                    print(f'Perdida Discriminador: {val_d_adv_loss}')
                    print(f'Tiempo entrenando: {elapsed_time[0]}:{elapsed_time[1]} minutos')
                    print('================================')

    return qa, discriminator, history
