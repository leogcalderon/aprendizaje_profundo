# 5.1 Layers and blocks

To implement these complex networks, we introduce the concept of a neural network block. A block could describe a single layer, a component consisting of multiple layers, or the entire model itself.

From a programing standpoint, a block is represented by a class. Any subclass of it must define a forward propagation function that transforms its input into output and must store any necessary parameters. Finally a block must possess a backpropagation function, for purposes of calculating gradients.

## 5.1.1 A custom block

**basic functionality that each block must provide:**

- Ingest input data as arguments to its forward propagation function.

- Generate an output by having the forward propagation function return a value. Note that the output may have a different shape from the input.

- Calculate the gradient of its output with respect to its input, which can be accessed via its backpropagation function. Typically this happens automatically.

- Store and provide access to those parameters necessary to execute the forward propagation computation.

- Initialize model parameters as needed.

```python
from torch import nn
from torch.nn import functional as F

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))
```

### 5.1.2 The sequential block

In the __init__ method, we add every module to the ordered dictionary _modules one by one.

```python
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            self._modules[str(idx)] = module

    def forward(self, X):
        for block in self._modules.values():
            X = block(X)
        return X
```

## 5.2 Parameter Management

Parameter access:
```python 
print(net[2].state_dict()))

>>> OrderedDict([('weight', tensor([[-0.0879,  0.1419, -0.1770,  0.1744,  0.2785, -0.2827, -0.1250,  0.2618]])), ('bias', tensor([-0.2401]))])

print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)

>>> <class 'torch.nn.parameter.Parameter'>
>>> Parameter containing:
>>> tensor([-0.2401], requires_grad=True)
>>> tensor([-0.2401])
```

Parameter init:
```python
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)

net.apply(init_normal)
```
