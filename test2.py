import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Set a random seed for reproducibility
np.random.seed(42)

# ----- Data Loading and Preprocessing -----
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data.astype(np.float32) / 255.0  # normalize pixel values
y = mnist.target.astype(np.int32)

# Split dataset into training and test sets (e.g., 60k train, 10k test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=10000, random_state=42
)
print("Dataset loaded. Training samples:", X_train.shape[0], "Test samples:", X_test.shape[0])


# ----- Helper Functions -----
def softmax(z):
    # subtract max for numerical stability
    expz = np.exp(z - np.max(z, axis=1, keepdims=True))
    return expz / np.sum(expz, axis=1, keepdims=True)

def cross_entropy_loss(probs, y):
    # Avoid log(0) by adding a small constant
    n = y.shape[0]
    loss = -np.log(probs[np.arange(n), y] + 1e-8).mean()
    return loss


# ----- MLP Model with One Hidden Layer -----
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # Initialize weights with small random numbers
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros(output_dim)
        
    def forward(self, X):
        # First layer: Linear + ReLU activation
        z1 = X.dot(self.W1) + self.b1
        a1 = np.maximum(0, z1)  # ReLU
        # Second layer: Linear (logits)
        logits = a1.dot(self.W2) + self.b2
        return logits, a1
    
    def predict(self, X):
        logits, _ = self.forward(X)
        probs = softmax(logits)
        return np.argmax(probs, axis=1)
    
    def compute_loss(self, X, y):
        logits, _ = self.forward(X)
        probs = softmax(logits)
        return cross_entropy_loss(probs, y)
    
    def get_params(self):
        # Flatten all parameters into a 1D vector
        params = np.concatenate([
            self.W1.flatten(),
            self.b1,
            self.W2.flatten(),
            self.b2
        ])
        return params
    
    def set_params(self, params):
        # Recover parameters from the flattened vector
        s1 = self.input_dim * self.hidden_dim
        s2 = self.hidden_dim
        s3 = self.hidden_dim * self.output_dim
        s4 = self.output_dim
        
        self.W1 = params[0:s1].reshape(self.input_dim, self.hidden_dim)
        self.b1 = params[s1:s1+s2]
        self.W2 = params[s1+s2:s1+s2+s3].reshape(self.hidden_dim, self.output_dim)
        self.b2 = params[s1+s2+s3:s1+s2+s3+s4]


# ----- Zeroth Order Gradient Descent (SPSA) Training -----
# Hyperparameters
input_dim = 784       # MNIST images are 28x28
hidden_dim = 100      # You can change this value
output_dim = 10       # 10 classes for digits 0-9
num_iterations = 1000
batch_size = 64
learning_rate = 0.01
epsilon = 1e-3  # finite difference step size

# Initialize the model
mlp = MLP(input_dim, hidden_dim, output_dim)
n_train = X_train.shape[0]

X_train = X_train.to_numpy()
y_train = y_train.to_numpy()

print("\nStarting training using zeroth order (SPSA) gradient estimation...")
for i in range(num_iterations):
    # Sample a mini-batch of data
    indices = np.random.choice(n_train, batch_size, replace=False)
    X_batch = X_train[indices]
    y_batch = y_train[indices]
    
    # Get current parameters
    theta = mlp.get_params()
    
    # Sample a random perturbation vector (each element is +1 or -1)
    delta = np.random.choice([-1, 1], size=theta.shape)
    
    # Evaluate the loss at theta + epsilon*delta
    mlp.set_params(theta + epsilon * delta)
    loss_plus = mlp.compute_loss(X_batch, y_batch)
    
    # Evaluate the loss at theta - epsilon*delta
    mlp.set_params(theta - epsilon * delta)
    loss_minus = mlp.compute_loss(X_batch, y_batch)
    
    # Estimate the gradient (SPSA estimate)
    grad_est = (loss_plus - loss_minus) / (2 * epsilon) * delta
    
    # Update the parameter vector using gradient descent
    theta_new = theta - learning_rate * grad_est
    mlp.set_params(theta_new)
    
    # Optionally, print training loss every 100 iterations
    if i % 100 == 0:
        current_loss = mlp.compute_loss(X_batch, y_batch)
        print(f"Iteration {i}: batch loss = {current_loss:.4f}")

# ----- Evaluation on Test Set -----
predictions = mlp.predict(X_test)
test_accuracy = np.mean(predictions == y_test)
print(f"\nTest accuracy: {test_accuracy * 100:.2f}%")
