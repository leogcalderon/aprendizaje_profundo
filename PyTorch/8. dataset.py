import torch
import torchvision
import numpy as np

class WineDataset(torch.utils.data.Dataset):

    def __init__(self):

        data = np.loadtxt(
            'data/wine.csv',
            delimiter=',',
            dtype=np.float32,
            skiprows=1
        )

        self.X = data[:, 1:]
        self.y = data[:, 0]

        self.N = data.shape[0]
        self.D = self.X.shape[1]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.N

dataset = WineDataset()

dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=4,
    shuffle=True,
    num_workers=2
)

epochs = 5
n = len(dataset)
iterations = int(n / 4)

for epoch in range(epochs):
    for i, (X, y) in enumerate(dataloader):
        print(epoch)
        print(i, X.shape, y.shape)
