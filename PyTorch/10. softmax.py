import torch
from torch import nn

# Y_true has class labels [NO ONE-HOT]
# Y_pred has raw scores (logits) [NO SOFTMAX]
loss = nn.CrossEntropyLoss()

# 3 samples
y_true = torch.tensor([2, 0, 1])
y_pred_good = torch.tensor([[2.0, 1.0, 3.1], [1.1, 1.0, 1.0], [1.0, 2.0, 1.8]])
predictions_good = torch.argmax(y_pred_good, axis=1)

y_pred_bad = torch.tensor([[4.1, 1.0, 2.0], [1.0, 2.0, 2.1], [1.0, 0.1, 0.1]])
predictions_bad = torch.argmax(y_pred_bad, axis=1)

print('y_true', y_true)
print('Good pred:', loss(y_pred_good, y_true), 'pred:', predictions_good)
print('Bad pred:', loss(y_pred_bad, y_true), 'pred:', predictions_bad)

# With a binary classification, use nn.BCELoss()
