import torch
from torch import nn
import torchvision
import matplotlib.pyplot as plt

device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

FEATURES = 28*28
HIDDEN = 128
CLASSES = 10
EPOCHS = 5
BATCH_SIZE = 128
LEARNING_RATE = 0.001

train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    transform=torchvision.transforms.ToTensor(),
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, classes):
        super(Model, self).__init__()
        self.layer_1 = nn.Linear(input_size, hidden_size)
        self.activation_1 = nn.ReLU()
        self.layer_2 = nn.Linear(hidden_size, classes)

    def forward(self, X):
        X = self.layer_1(X)
        X = self.activation_1(X)
        return self.layer_2(X)

model = Model(FEATURES, HIDDEN, CLASSES)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    for i, (X, y) in enumerate(train_loader):

        X = X.reshape(-1, FEATURES).to(device)
        y = y.to(device)

        y_pred = model(X)
        l = loss(y_pred, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

    print(f'Epoch {epoch} - loss {l}')

    with torch.no_grad():
        correct = 0
        for X, y in test_loader:
            X = X.reshape(-1, FEATURES).to(device)
            y = y.to(device)
            y_pred = torch.argmax(model(X), axis=1)
            correct += torch.sum(y_pred == y)

    print('Test accuracy:', (correct/(len(test_loader)*BATCH_SIZE)).item())
