import torchvision
import torch
import numpy as np

class WineDataset(torch.utils.data.Dataset):

    def __init__(self, transform=None):

        data = np.loadtxt(
            'data/wine.csv',
            delimiter=',',
            dtype=np.float32,
            skiprows=1
        )

        self.X = data[:, 1:]
        self.y = data[:, [0]]

        self.N = data.shape[0]
        self.D = self.X.shape[1]

        self.transform = transform

    def __getitem__(self, index):

        sample = (self.X[index], self.y[index])

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.N

class ToTensor:
    def __call__(self, sample):
        X, y = sample
        return (
            torch.from_numpy(X),
            torch.from_numpy(y)
        )

class MulTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        X, y = sample
        return self.factor*X, y

composed = torchvision.transforms.Compose([
    ToTensor(),
    MulTransform(2)
])

print('No transformers')
dataset = WineDataset()
example = dataset[0]
X, y = example
print(type(X), type(y))
print(X)

print('\nWith transformers')
dataset = WineDataset(transform=composed)
example = dataset[0]
X, y = example
print(type(X), type(y))
print(X)
