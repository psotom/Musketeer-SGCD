import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.func import functional_call, jvp
import functorch as ft
import random

# Define a simple MLP with one hidden layer.
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # Flatten MNIST images (28x28) into vectors of size 784.
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Prepare MNIST data.
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Choose device.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the model and loss function.
model = SimpleMLP().to(device)
criterion = nn.CrossEntropyLoss()

# For coordinate descent we need to work with the model parameters.
# We extract them into a list so that we can update them manually.
params = model.parameters()
params_list = list(model.parameters())
func, params, buffers = ft.make_functional_with_buffers(model)

# Learning rate and number of coordinate updates.
lr = 0.01
num_iterations = 1000

# Create a data iterator to cycle through the dataset.
data_iter = iter(train_loader)

# Training loop with coordinate descent.
for iter_num in range(num_iterations):
    # Get a new batch; if the iterator is exhausted, reinitialize it.
    try:
        x_batch, y_batch = next(data_iter)
    except StopIteration:
        data_iter = iter(train_loader)
        x_batch, y_batch = next(data_iter)
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    
    # Define a loss function that depends on the parameters.
    # We use functorch.functional_call so that parameters are passed explicitly.
    func, params, buffers = ft.make_functional_with_buffers(model)
    def loss_fn(p):
        # p is a tuple of parameters.
        out = func(params, buffers, x_batch)
        loss = criterion(out, y_batch)
        return loss
    
    # Randomly select one parameter tensor and a coordinate (entry) within it.
    param_idx = random.randrange(len(params_list))
    param_tensor = params_list[param_idx]
    numel = param_tensor.numel()
    coord_idx = random.randrange(numel)
    
    # Build a tangent vector (seed) with the same structure as params.
    # Only the selected parameter has a nonzero entry (a one-hot vector).
    tangent = []
    for i, p in enumerate(params_list):
        print(p)
        if i == param_idx:
            t = torch.zeros_like(p)
            # Set the chosen coordinate to 1.
            t.view(-1)[coord_idx] = 1.0
            tangent.append(t)
        else:
            tangent.append(torch.zeros_like(p))
    tangent = tuple(tangent)
    tangents = tuple([
        torch.ones_like(p) if coord_idx == i 
        else torch.ones_like(p) 
        for i, p in enumerate(model.parameters())
    ])
    tangents = tuple([torch.rand_like(p) for p in model.parameters()])
    # Compute the function value and the directional derivative (JVP) in the direction of tangent.
    #params_tuple = tuple(params)
    loss_val, directional_deriv = jvp(loss_fn, (params, ), (tangents,))
    print(loss_val, directional_deriv)
    
    # Extract the derivative for the selected coordinate.
    grad_component = directional_deriv[param_idx].view(-1)[coord_idx].item()
    
    # Update only the selected coordinate.
    # Since we are updating in place, we flatten the tensor view, update the coordinate,
    # and the change will reflect in the parameter used by the model.
    p_flat = params[param_idx].view(-1)
    p_flat[coord_idx] = p_flat[coord_idx] - lr * grad_component
    
    if iter_num % 100 == 0:
        print(f"Iteration {iter_num:4d}, Loss: {loss_val.item():.4f}, Grad[{param_idx}][{coord_idx}] = {grad_component:.4f}")

# After training, you can evaluate the model on a test set (not shown here)
# to see how well it learned.
