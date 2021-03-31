import torch
from torch import nn


X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
X_test = torch.tensor([[5]], dtype=torch.float32)
n, d = X.shape

iterations = 100
learning_rate = 0.01
model = nn.Linear(d, d)
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

print(f'f(5)=', model(X_test).item())

for it in range(iterations):
    y_pred = model(X)
    l = loss(y, y_pred)
    l.backward()
    optimizer.step()
    optimizer.zero_grad()

    if it % 10 == 0:
        [w, b] = model.parameters()
        print(f'Epoch: {it} - loss: {l} - w = {w[0][0]}')

print(f'f(5)=', model(X_test).item())
