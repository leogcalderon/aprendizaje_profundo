import torch
from torch import nn
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = datasets.load_breast_cancer()
X, y = data.data, data.target
N, d = X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

sc = StandardScaler()
X_train, X_test = sc.fit_transform(X_train), sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))

y_train = torch.from_numpy(y_train.astype(np.float32)).view(y_train.shape[0], 1)
y_test = torch.from_numpy(y_test.astype(np.float32)).view(y_test.shape[0], 1)

class LR(nn.Module):
    def __init__(self, d):
        super(LR, self).__init__()
        self.linear = nn.Linear(d, 1)

    def forward(self, X):
        return torch.sigmoid(self.linear(X))

model = LR(d)
loss = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

epochs = 100

for epoch in range(epochs):
    pred = model(X_train)
    l = loss(pred, y_train)
    l.backward()
    optimizer.step()
    optimizer.zero_grad()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch: {epoch + 1} - Loss: {l}')
