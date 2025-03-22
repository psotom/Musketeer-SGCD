import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from methods import uniform_coordinate_descent, gradient_descent, musketeer, musketeer2

# Ridge Regression
def f_ridge(theta, X, y, mu):
    n = X.shape[0]
    residual = y - X.dot(theta)
    return np.sum(residual ** 2) / (2 * n) + (mu / 2) * np.sum(theta ** 2)

def grad_ridge(theta, X, y, mu, k=None):
    n = X.shape[0]
    if k is None:
        return (-1/n) * X.T.dot(y - X.dot(theta)) + mu * theta
    else:
        # derivative with respect to coordinate k
        return (-1/n) * np.dot(X[:, k], y - X.dot(theta)) + mu * theta[k]

# Logistic Regression
def f_logistic(theta, X, y, mu):
    n = X.shape[0]
    Xtheta = X.dot(theta)
    # use stable computation for log(1+exp(-z))
    loss = np.log(1 + np.exp(-y * Xtheta))
    return (1/n) * np.sum(loss) + mu * np.sum(theta**2)

def grad_logistic(theta, X, y, mu, k=None):
    n = X.shape[0]
    Xtheta = X.dot(theta)
    # derivative: -y/(1+exp(y*(Xtheta)))
    factor = - y / (1 + np.exp(y * Xtheta))
    if k is None:
        return (1/n) * X.T.dot(factor) + 2 * mu * theta
    else:
        return (1/n) * np.dot(X[:, k], factor) + 2 * mu * theta[k]

# -------------------------
# Data Generation with Block Structure
# -------------------------

def generate_data(n, p, alpha, model='ridge'):
    """
    Generate data with block structure.
    Each column j is drawn from N(0, sigma_j^2) with sigma_j^2 = (j+1)^{-alpha}.
    """
    X = np.zeros((n, p))
    for j in range(p):
        sigma = (j + 1) ** (-alpha / 2)
        X[:, j] = np.random.randn(n) * sigma

    # generate a ground truth parameter vector
    theta_true = np.random.randn(p) * 3
    
    if model == 'ridge':
        noise = 0.1 * np.random.randn(n)
        y = X.dot(theta_true) + noise
    elif model == 'logistic':
        logits = X.dot(theta_true)
        prob = 1 / (1 + np.exp(-logits))
        # assign labels in {-1, 1}
        y = np.where(prob > 0.5, 1, -1)
    else:
        raise ValueError("Model type must be 'ridge' or 'logistic'.")
    return X, y, theta_true

# -------------------------
# Main Experiment
# -------------------------

if __name__ == '__main__':
    # Settings
    model_type = 'logistic'  # choose 'ridge' or 'logistic'
    n = 10000      # number of samples
    p = 250        # number of features
    alpha = 5.0    # block decay parameter for feature variances
    mu = 1 / n       # regularization parameter
    np.random.seed(0)
    
    # Generate data
    X, y, theta_true = generate_data(n, p, alpha, model=model_type)
    
    # Initialization
    theta0 = np.zeros(p)

    # Set up objective and gradient functions for the chosen model.
    if model_type == 'ridge':
        f = lambda theta: f_ridge(theta, X, y, mu)
        grad = lambda theta, k=None: grad_ridge(theta, X, y, mu, k)
        theta_star = np.linalg.solve((X.T @ X)/n + mu * np.eye(p), (X.T @ y)/n)
    elif model_type == 'logistic':
        f = lambda theta: f_logistic(theta, X, y, mu)
        grad = lambda theta, k=None: grad_logistic(theta, X, y, mu, k)

        res = minimize(f, theta0, jac=grad, method='L-BFGS-B')
        theta_star = res.x
 
    
    # Total coordinate evaluations for fair comparison
    total_evals = 20000
    # For uniform coordinate descent, one step = one coordinate evaluation.
    steps_cd = total_evals
    # For full gradient descent, one iteration uses p evaluations.
    steps_gd = total_evals // p  # integer division
    # For MUSKETEER, let T ≈ sqrt(p) and epochs such that total evaluations ≈ total_evals.
    T = int(np.sqrt(p))
    epochs_musketeer = total_evals // T
    # Choose learning rate (may need tuning)
    gamma_0 = 10 # 3 for ridge, 10 for logistic
    k0 = 5 # 10 for ridge, 5 for logistic
    gamma = gamma_0 * p / (gamma_0 * p * k0 + np.arange(steps_cd))   # this is one possible choice; tuning may be required.
    gamma_gd = gamma_0 / (gamma_0 * k0 + np.arange(steps_gd))
    print(gamma[0], gamma[-1], gamma_gd[0], gamma_gd[-1])
    gamma = 0.1
    gamma_gd = 0.1

    # For MUSKETEER, set lambda sequence and eta parameter.
    lambda_val = 100
    lambda_base = 3
    lambda_seq = 1 / np.log(lambda_base + lambda_val * np.arange(epochs_musketeer))
    print(lambda_seq[0], lambda_seq[-1])
    lambda_seq = 0.5 # for softmax
    eta = 1.0
    gain_type = 'abs'  # could be 'avg' or 'sqr' as well

    # Run Uniform Coordinate Descent
    theta_cd, history_cd, evals_cd = uniform_coordinate_descent(theta0, f, grad, steps_cd, gamma)
    print("Uniform Coordinate Descent final difference:", f(theta_cd) - f(theta_star))
    
    # Run Full Gradient Descent
    theta_gd, history_gd, evals_gd = gradient_descent(theta0, f, grad, steps_gd, gamma_gd)
    print("Gradient Descent final difference:", f(theta_gd) - f(theta_star))
    
    # Run MUSKETEER
    theta_musk, history_musk, evals_musk = musketeer(theta0, f, grad, epochs_musketeer, T, gamma, lambda_seq, eta, gain_type)
    print("MUSKETEER final difference:", f(theta_musk) - f(theta_star))
    
    # -------------------------
    # Plotting Results
    # -------------------------
    plt.figure(figsize=(8, 5))
    plt.semilogy(evals_cd, history_cd - f(theta_star), label='Uniform Coordinate Descent')
    plt.semilogy(evals_gd, history_gd - f(theta_star), label='Full Gradient Descent')
    plt.semilogy(evals_musk, history_musk - f(theta_star), label='MUSKETEER')
    plt.xlabel('Coordinate Evaluations')
    plt.ylabel('Objective Value')
    plt.title(f"Regularized {model_type.capitalize()} Regression Comparison, with " + r"$\alpha$ =" + str(int(alpha)))
    plt.legend()
    plt.grid(True)
    plt.savefig(model_type)
    plt.show()