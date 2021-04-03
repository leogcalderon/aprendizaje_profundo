import torch
from torch import nn
import numpy as np
from sklearn import datasets

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=27)
X = torch.from_numpy(X.astype(np.float32))
y = torch.from_numpy(y.astype(np.float32)).view(y.shape[0], 1)

N, d = X.shape

liner_regression = nn.Linear(d, 1)
loss = nn.MSELoss()
optimizer = torch.optim.SGD(liner_regression.parameters(), lr=0.01)

epochs = 100

for epoch in range(epochs):
    pred = liner_regression(X)
    l = loss(pred, y)
    l.backward()
    optimizer.step()
    optimizer.zero_grad()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch: {epoch + 1} - Loss: {l}')
