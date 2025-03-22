import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from methods import uniform_coordinate_descent, gradient_descent, musketeer, musketeer2
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_grad(z):
    s = sigmoid(z)
    return s * (1 - s)

def softmax(z):
    # Numerically stable softmax for both vectors and matrices.
    exps = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return exps / np.sum(exps, axis=-1, keepdims=True)

class SimpleMLP:
    def __init__(self, input_size, hidden_size, num_classes):
        """
        Initialize the network parameters.
        """
        self.W1 = np.random.randn(hidden_size, input_size) * 0.1
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(num_classes, hidden_size) * 0.1
        self.b2 = np.zeros(num_classes)
        self.total_params = self.W1.size + self.b1.size + self.W2.size + self.b2.size

    def forward(self, x, return_logits=True):
        """
        Forward pass.
        If return_logits is True, returns (z1, a1, z2) where z2 are the raw logits.
        If False, returns (z1, a1, z2, a2) with a2 = softmax(z2).
        """
        if x.ndim == 1:
            z1 = np.dot(self.W1, x) + self.b1
            a1 = sigmoid(z1)
            z2 = np.dot(self.W2, a1) + self.b2
        else:
            z1 = np.dot(x, self.W1.T) + self.b1  # (batch_size, hidden_size)
            a1 = sigmoid(z1)
            z2 = np.dot(a1, self.W2.T) + self.b2   # (batch_size, num_classes)
        if return_logits:
            return z1, a1, z2
        else:
            a2 = softmax(z2)
            return z1, a1, z2, a2

    def compute_gradient_single_param(self, x, y, param_layer, i, j=None):
        """
        Compute the gradient of the loss (for one training sample) with respect to one parameter.
        """
        # Use the logits mode, then compute softmax manually.
        z1, a1, z2 = self.forward(x, return_logits=True)
        a2 = softmax(z2)
        delta2 = a2.copy()
        delta2[y] -= 1  # derivative of cross-entropy loss wrt z2
        
        if param_layer == 'W2':
            if j is None:
                raise ValueError("Index j must be provided for weight matrices.")
            grad = delta2[i] * a1[j]
        elif param_layer == 'b2':
            grad = delta2[i]
        elif param_layer == 'W1':
            if j is None:
                raise ValueError("Index j must be provided for weight matrices.")
            # For W1, backpropagate the error through the hidden layer.
            hidden_error = np.dot(self.W2[:, i], delta2)
            grad = hidden_error * sigmoid_grad(z1[i]) * x[j]
        elif param_layer == 'b1':
            hidden_error = np.dot(self.W2[:, i], delta2)
            grad = hidden_error * sigmoid_grad(z1[i])
        else:
            raise ValueError("param_layer must be one of 'W1', 'b1', 'W2', or 'b2'")
        return grad

    def coordinate_update(self, x, y, param_layer, i, j=None, learning_rate=0.1):
        """
        Performs a coordinate update on a single parameter.
        """
        grad = self.compute_gradient_single_param(x, y, param_layer, i, j)
        if param_layer in ['W1', 'W2']:
            if j is None:
                raise ValueError("Index j must be provided for weight matrices.")
            if param_layer == 'W1':
                self.W1[i, j] -= learning_rate * grad
            else:
                self.W2[i, j] -= learning_rate * grad
        elif param_layer in ['b1', 'b2']:
            if param_layer == 'b1':
                self.b1[i] -= learning_rate * grad
            else:
                self.b2[i] -= learning_rate * grad
        else:
            raise ValueError("Invalid parameter layer.")
        return grad

    def flat_coordinate_update(self, x, y, k, learning_rate=0.1):
        """
        Selects a coordinate (flattened index k) and applies a coordinate update.
        """
        if k < self.W1.size:
            param_layer = 'W1'
            i = k // self.W1.shape[1]
            j = k % self.W1.shape[1]
        elif k < self.W1.size + self.b1.size:
            param_layer = 'b1'
            i = k - self.W1.size
            j = None
        elif k < self.W1.size + self.b1.size + self.W2.size:
            param_layer = 'W2'
            k_adjusted = k - (self.W1.size + self.b1.size)
            i = k_adjusted // self.W2.shape[1]
            j = k_adjusted % self.W2.shape[1]
        else:
            param_layer = 'b2'
            i = k - (self.W1.size + self.b1.size + self.W2.size)
            j = None

        grad = self.coordinate_update(x, y, param_layer, i, j, learning_rate)
        return param_layer, i, j, grad

def get_parameters_vector(model):
    """
    Returns a flat numpy array containing all parameters of the model.
    """
    return np.concatenate([model.W1.flatten(), model.b1.flatten(),
                           model.W2.flatten(), model.b2.flatten()])

def set_parameters_vector(model, theta):
    """
    Sets the model parameters from the flat vector theta.
    """
    W1_size = model.W1.size
    b1_size = model.b1.size
    W2_size = model.W2.size
    model.W1 = theta[0:W1_size].reshape(model.W1.shape)
    model.b1 = theta[W1_size:W1_size+b1_size]
    model.W2 = theta[W1_size+b1_size:W1_size+b1_size+W2_size].reshape(model.W2.shape)
    model.b2 = theta[W1_size+b1_size+W2_size:]

def f_net(theta, model, X, y, loss_fn, batch_size=32):
    """
    Computes the average loss over a mini-batch from (X, y) using a vectorized forward pass.
    """
    set_parameters_vector(model, theta)
    N = len(X)
    batch_indices = np.random.choice(N, batch_size, replace=False)
    # Convert the mini-batch to a 2D NumPy array.
    X_batch = torch.stack([X[idx] for idx in batch_indices]).view(batch_size, -1).numpy()  # shape: (batch_size, input_size)
    # Forward pass in logits mode.
    _, _, z2 = model.forward(X_batch, return_logits=True)
    logits = torch.tensor(z2, dtype=torch.float32)
    labels = torch.tensor([y[idx] for idx in batch_indices], dtype=torch.long)
    loss_val = loss_fn(logits, labels)
    return loss_val.item()

def grad_net(theta, model, X, y, loss_fn, k=None, batch_size=32):
    """
    Computes the gradient with respect to theta over a mini-batch.
    
    If k is provided, computes the average gradient for that coordinate over the batch;
    otherwise, computes the full gradient vector.
    (Note: The per-sample gradient is still computed in a loop.)
    """
    set_parameters_vector(model, theta)
    N = len(X)
    batch_indices = np.random.choice(N, batch_size, replace=False)
    
    if k is None:
        grad_vector = np.zeros_like(theta)
        p = theta.shape[0]
        for coord in range(p):
            grad_total = 0.0
            if coord < model.W1.size:
                param_layer = 'W1'
                i = coord // model.W1.shape[1]
                j = coord % model.W1.shape[1]
            elif coord < model.W1.size + model.b1.size:
                param_layer = 'b1'
                i = coord - model.W1.size
                j = None
            elif coord < model.W1.size + model.b1.size + model.W2.size:
                param_layer = 'W2'
                k_adjusted = coord - (model.W1.size + model.b1.size)
                i = k_adjusted // model.W2.shape[1]
                j = k_adjusted % model.W2.shape[1]
            else:
                param_layer = 'b2'
                i = coord - (model.W1.size + model.b1.size + model.W2.size)
                j = None
            for idx in batch_indices:
                x_i = X[idx].view(-1).numpy()
                grad_total += model.compute_gradient_single_param(x_i, y[idx], param_layer, i, j)
            grad_vector[coord] = grad_total / batch_size
        return grad_vector
    else:
        grad_total = 0.0
        if k < model.W1.size:
            param_layer = 'W1'
            i = k // model.W1.shape[1]
            j = k % model.W1.shape[1]
        elif k < model.W1.size + model.b1.size:
            param_layer = 'b1'
            i = k - model.W1.size
            j = None
        elif k < model.W1.size + model.b1.size + model.W2.size:
            param_layer = 'W2'
            k_adjusted = k - (model.W1.size + model.b1.size)
            i = k_adjusted // model.W2.shape[1]
            j = k_adjusted % model.W2.shape[1]
        else:
            param_layer = 'b2'
            i = k - (model.W1.size + model.b1.size + model.W2.size)
            j = None
        for idx in batch_indices:
            x_i = X[idx].view(-1).numpy()
            grad_total += model.compute_gradient_single_param(x_i, y[idx], param_layer, i, j)
        return grad_total / batch_size

def compute_accuracy(theta, model, X, y):
    """
    Computes the classification accuracy over the dataset (X, y) using a vectorized forward pass.
    """
    set_parameters_vector(model, theta)
    # Convert all data into a 2D NumPy array.
    X_np = X.view(X.size(0), -1).numpy()  # shape: (N, input_size)
    # Get logits and then compute softmax for probabilities.
    _, _, z2 = model.forward(X_np, return_logits=True)
    a2 = softmax(z2)
    preds = np.argmax(a2, axis=1)
    # Ensure y is a flat array.
    y_np = y.numpy().flatten() if isinstance(y, torch.Tensor) else np.array(y).flatten()
    return np.mean(preds == y_np)

def run_experiment(dataset_name, batch_size=128):
    """
    Runs the optimization tests on either MNIST or Fashion-MNIST using mini-batches.
    Additionally, records both loss and accuracy during optimization.
    """
    transform = transforms.ToTensor()
    
    if dataset_name == 'MNIST':
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset_name == 'FashionMNIST':
        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    else:
        raise ValueError("Dataset must be 'MNIST' or 'FashionMNIST'.")
    
    # Load the dataset.
    X_list, y_list = [], []
    for img, label in trainset:
        X_list.append(img)
        y_list.append(label)
    X = torch.stack(X_list)  # shape: (N, 1, 28, 28)
    y = torch.tensor(y_list, dtype=torch.long)

    X_list, y_list = [], []
    for img, label in testset:
        X_list.append(img)
        y_list.append(label)
    X_test = torch.stack(X_list)  # shape: (N, 1, 28, 28)
    y_test = torch.tensor(y_list, dtype=torch.long)
    print("hay ", len(X_test), " tests")
    
    # Create the network and loss function.
    model = SimpleMLP(784, 64, 10)
    loss_fn = nn.CrossEntropyLoss()
    
    # Initial parameter vector.
    theta0 = get_parameters_vector(model)
    p = len(theta0)
    print(f"{dataset_name}: Network parameter dimension p = {p}")
    
    # Evaluation budgets.
    T = 234                     # For MUSKETEER (≈ √p)
    total_evals = 100000
    steps_cd = total_evals                  # one coordinate evaluation per step in UCD
    steps_gd = total_evals // p             # full gradient descent (each step counts as p evaluations)
    epochs_musketeer = total_evals // T     # each epoch of MUSKETEER uses T coordinate evaluations
    
    # Learning rate.
    gamma = 10

    # Accuracy history lists.
    acc_history_cd = []
    acc_history_gd = []
    acc_history_musk = []

    test_size = 128
    N_test = len(X_test)
    # Callback functions.
    def callback_cd(theta, eval_count):
        batch_indices = np.random.choice(N_test, test_size, replace=False)
        X_batch = torch.stack([X[idx] for idx in batch_indices]).view(test_size, -1)
        # Use a flat tensor for labels.
        y_batch = torch.tensor([y[idx] for idx in batch_indices])
        acc = compute_accuracy(theta, model, X_batch, y_batch)
        acc_history_cd.append((eval_count, acc))

    def callback_gd(theta, eval_count):
        batch_indices = np.random.choice(N_test, test_size, replace=False)
        X_batch = torch.stack([X_test[idx] for idx in batch_indices]).view(test_size, -1)
        y_batch = torch.tensor([y_test[idx] for idx in batch_indices])
        acc = compute_accuracy(theta, model, X_batch, y_batch)
        acc_history_gd.append((eval_count, acc))

    def callback_musk(theta, eval_count):
        batch_indices = np.random.choice(N_test, test_size, replace=False)
        X_batch = torch.stack([X_test[idx] for idx in batch_indices]).view(test_size, -1)
        y_batch = torch.tensor([y_test[idx] for idx in batch_indices])
        acc = compute_accuracy(theta, model, X_batch, y_batch)
        acc_history_musk.append((eval_count, acc))
    
    # Function handles with mini-batch evaluations.
    f = lambda theta: f_net(theta, model, X, y, loss_fn, batch_size)
    grad = lambda theta, k=None: grad_net(theta, model, X, y, loss_fn, k, batch_size)
    
    # Run the optimization methods.
    # Uncomment the ones you want to run.
    # set_parameters_vector(model, theta0)
    # theta_gd, history_gd, evals_gd = gradient_descent(theta0, f, grad, steps_gd, gamma, callback=callback_gd)
    # print(f"{dataset_name} - GD final loss: {f(theta_gd):.4f}")
    
    set_parameters_vector(model, theta0)
    lambda_seq = 0.2
    eta = 5.0         
    gain_type = 'abs'
    theta_musk, history_musk, evals_musk = musketeer2(theta0, f, grad, epochs_musketeer, T, gamma, lambda_seq, eta, gain_type, callback=callback_musk, n_calls=10)
    print(f"{dataset_name} - MUSKETEER final loss: {f(theta_musk):.4f}")

    set_parameters_vector(model, theta0)
    theta_cd, history_cd, evals_cd = uniform_coordinate_descent(theta0, f, grad, steps_cd, gamma, eval_each=T, callback=callback_cd, n_calls=10)
    print(f"{dataset_name} - UCD final loss: {f(theta_cd):.4f}")
    
    # Plot loss and accuracy evolution.
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves.
    axs[0].plot(evals_cd, history_cd, label='UCD')
    #axs[0].plot(evals_gd, history_gd, label='GD')
    axs[0].plot(evals_musk, history_musk, label='MUSKETEER')
    axs[0].set_xlabel('Coordinate Evaluations')
    axs[0].set_ylabel('Training Loss')
    axs[0].set_title(f'{dataset_name}: Loss Evolution')
    axs[0].legend()
    axs[0].grid(True)
    
    # Accuracy curves.
    if acc_history_cd:
        evals_acc_cd, acc_cd = zip(*acc_history_cd)
        axs[1].plot(evals_acc_cd, acc_cd, label='UCD')
    if acc_history_gd:
        evals_acc_gd, acc_gd = zip(*acc_history_gd)
        axs[1].plot(evals_acc_gd, acc_gd, label='GD')
    if acc_history_musk:
        evals_acc_musk, acc_musk = zip(*acc_history_musk)
        axs[1].plot(evals_acc_musk, acc_musk, label='MUSKETEER')
    axs[1].set_xlabel('Coordinate Evaluations')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_title(f'{dataset_name}: Accuracy Evolution')
    axs[1].legend()
    axs[1].grid(True)
    
    plt.show()

# -----------------------------
# Example usage:
# -----------------------------
if __name__ == "__main__":
    run_experiment('MNIST', batch_size=32)
