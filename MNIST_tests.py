import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from methods import uniform_coordinate_descent, gradient_descent, musketeer

# -------------------------
# Network Architecture
# -------------------------
# We build a simple fully-connected network with:
#   - Input dimension: 28*28 = 784
#   - Hidden dimension: 69 (chosen so that total parameters are ≃55,050)
#   - Output dimension: 10
#
# With biases in both layers, the total number of parameters is:
#   p = (784*69 + 69) + (69*10 + 10) = 795*69 + 10 ≃ 54,865.
# This is close enough to 55,050 so that √p ≃ 234.

class SimpleNet(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=69, output_dim=10):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=True)
        
    def forward(self, x):
        # x is assumed to be of shape [N, 1, 28, 28]; flatten it.
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -------------------------
# Helper Functions: Parameter Handling
# -------------------------

def get_parameters_vector(model):
    """Flatten all model parameters into a single 1D numpy array."""
    return np.concatenate([p.detach().cpu().numpy().ravel() for p in model.parameters()])

def set_parameters_vector(model, theta):
    """Set the model parameters from a flattened vector theta."""
    pointer = 0
    for p in model.parameters():
        numel = p.numel()
        new_vals = theta[pointer:pointer+numel].reshape(p.size())
        p.data.copy_(torch.tensor(new_vals, dtype=p.dtype, device=p.device))
        pointer += numel

# -------------------------
# Objective and Gradient Functions
# -------------------------
# Here we use CrossEntropyLoss. The network outputs raw logits and
# the labels are integer class labels.

def f_net(theta, model, X, y, loss_fn):
    """Compute loss over the full dataset."""
    set_parameters_vector(model, theta)
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        loss = loss_fn(outputs, y)
    return loss.item()

def grad_net(theta, model, X, y, loss_fn, k=None):
    """Compute full gradient and optionally return the k-th coordinate."""
    set_parameters_vector(model, theta)
    model.train()
    model.zero_grad()
    outputs = model(X)
    loss = loss_fn(outputs, y)
    loss.backward()
    grads = []
    for p in model.parameters():
        grads.append(p.grad.detach().cpu().numpy().ravel())
    full_grad = np.concatenate(grads)
    if k is None:
        return full_grad
    else:
        return full_grad[k]

# -------------------------
# Experiment Function
# -------------------------
def run_experiment(dataset_name):
    """
    Runs the optimization tests on either MNIST or Fashion-MNIST.
    """
    # Define transformation: convert images to tensors.
    transform = transforms.ToTensor()
    
    if dataset_name == 'MNIST':
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    elif dataset_name == 'FashionMNIST':
        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    else:
        raise ValueError("Dataset must be 'MNIST' or 'FashionMNIST'.")
    
    # For simplicity, we load the entire training set into memory.
    X_list = []
    y_list = []
    for img, label in trainset:
        X_list.append(img)  # shape: [1, 28, 28]
        y_list.append(label)
    X = torch.stack(X_list)  # [N, 1, 28, 28]
    y = torch.tensor(y_list, dtype=torch.long)
    
    # Create the network and loss function.
    model = SimpleNet(input_dim=784, hidden_dim=69, output_dim=10)
    loss_fn = nn.CrossEntropyLoss()
    
    # Get the initial parameter vector and determine its dimension.
    theta0 = get_parameters_vector(model)
    p = len(theta0)
    print(f"{dataset_name}: Network parameter dimension p = {p}")
    
    # According to the test specification, set T = 234 (≈ √p).
    T = 234
    
    # Define a budget for coordinate evaluations.
    total_evals = 100000  
    steps_cd = total_evals                  # one coordinate evaluation per step in UCD
    steps_gd = total_evals // p             # full gradient descent: one step counts as p evaluations
    epochs_musketeer = total_evals // T     # each epoch of MUSKETEER uses T coordinate evaluations
    
    # Learning rate (this may require tuning)
    gamma = 0.01
    
    # Define function handles wrapping f_net and grad_net.
    f = lambda theta: f_net(theta, model, X, y, loss_fn)
    grad = lambda theta, k=None: grad_net(theta, model, X, y, loss_fn, k)
    
    # Run Uniform Coordinate Descent (UCD)
    theta_cd, history_cd, evals_cd = uniform_coordinate_descent(theta0, f, grad, steps_cd, gamma)
    print(f"{dataset_name} - UCD final loss: {f(theta_cd):.4f}")
    
    # Reset model to initial parameters.
    set_parameters_vector(model, theta0)
    theta_gd, history_gd, evals_gd = gradient_descent(theta0, f, grad, steps_gd, gamma)
    print(f"{dataset_name} - GD final loss: {f(theta_gd):.4f}")
    
    # Reset model to initial parameters.
    set_parameters_vector(model, theta0)
    lambda_seq = 0.1  # constant mixing parameter
    eta = 1.0         # softmax temperature parameter
    gain_type = 'abs'
    theta_musk, history_musk, evals_musk = musketeer(theta0, f, grad, epochs_musketeer, T, gamma, lambda_seq, eta, gain_type)
    print(f"{dataset_name} - MUSKETEER final loss: {f(theta_musk):.4f}")
    
    # Plot the evolution of the loss.
    plt.figure(figsize=(8, 5))
    plt.plot(evals_cd, history_cd, label='Uniform Coordinate Descent')
    plt.plot(evals_gd, history_gd, label='Full Gradient Descent')
    plt.plot(evals_musk, history_musk, label='MUSKETEER')
    plt.xlabel('Coordinate Evaluations')
    plt.ylabel('Training Loss')
    plt.title(f'{dataset_name}: Optimization Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

# -------------------------
# Run Experiments on MNIST and Fashion-MNIST
# -------------------------

if __name__ == '__main__':
    # Run on MNIST
    run_experiment('MNIST')
    
    # Run on Fashion-MNIST
    run_experiment('FashionMNIST')
