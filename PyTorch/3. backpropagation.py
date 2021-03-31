import torch

# Forward pass: compute loss
# Compute local gradients
# Backward pass: compute input gradients with chain rule

x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad=True)

# Forward pass
y_pred = w * x
loss = (y_pred - y)**2

# Backward pass
loss.backward()
print(w.grad)
