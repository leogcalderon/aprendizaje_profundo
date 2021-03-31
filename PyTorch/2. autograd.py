import torch

# We must specified the variables that we want to have grad
x = torch.randn(3, requires_grad=True)
print('x:', x)

# This will create a computational graph
# For each operation will be having inputs and outputs
# This graph allow us to get the gradients automatically
# y will be have a grad_fn called AddBackward that calculates dy/dx
y = x + 2
z = y * y * 2
z = z.mean()
print('z:', z)

# Gets the gradient of dz/dx
# and saves it in the grad attribute of the input variable (x).
z.backward()
print('dz/dx:', x.grad)

# To prevent gradient tracking
# - x.requires_grad_(False))
# - x.detach()
# - with torch.no_grad():

# To prevent the gradient accumulation
# variable.grad.zero_()
# optimizer.grad_zero()
