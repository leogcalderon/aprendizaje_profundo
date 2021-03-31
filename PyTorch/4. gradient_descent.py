import numpy as np
import torch

# NUMPY IMPLEMENTATION
print('Numpy IMPLEMENTATION')
X = np.array([1, 2, 3, 4], dtype=np.float32)
y = np.array([2, 4, 6, 8], dtype=np.float32)
w = 0.0

# Predictions
def forward(x):
    return w * x

# Loss
def loss(y, y_pred):
    return np.sum((y - y_pred)**2).mean()

# Gradient
def gradient(x, y, y_pred):
    return np.dot(2*x, y_pred - y).mean()

print(f'Prediction before training: f(5) = {forward(5)}')

learning_rate = 0.01
iterations = 10

for it in range(iterations):
    y_pred = forward(X)
    l = loss(y, y_pred)
    w -= learning_rate * gradient(X, y, y_pred)
    print(f'Epoch: {it} - loss: {l} - w = {w}')

print(f'Prediction after training: f(5) = {forward(5)}')

# TORCH IMPLEMENTATION
print('\nPyTorch IMPLEMENTATION')
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# Predictions
def forward(x):
    return w * x

# Loss
def loss(y, y_pred):
    return torch.sum((y - y_pred)**2).mean()

print(f'Prediction before training: f(5) = {forward(5)}')

for it in range(iterations):
    y_pred = forward(X)
    l = loss(y, y_pred)
    l.backward()
    # Because we dont need to have this in the computational graph
    with torch.no_grad():
        w -= learning_rate * w.grad

    w.grad.zero_()

    print(f'Epoch: {it} - loss: {l} - w = {w}')

print(f'Prediction after training: f(5) = {forward(5)}')
