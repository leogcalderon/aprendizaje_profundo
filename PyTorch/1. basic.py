import torch

if torch.cuda.is_available():

    device = torch.device('cuda')

    # Write to device
    x = torch.ones(5, device=device)

    # It writes to cpu for default
    y = torch.ones(5)

    # Send it to cuda
    y = y.to(device)
    z = x * y

    # If z is in the gpu, we cannot convert it to numpy
    # So we send it to cpu first
    z = z.to('cpu')
    print(z.numpy())

else:
    print('No cuda')
