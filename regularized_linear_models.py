import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from methods import uniform_coordinate_descent, gradient_descent, musketeer, musketeer2, grad_first, grad_zeroth

# Ridge Regression
def f_ridge(theta, X, y, mu):
    n = X.shape[0]
    residual = y - X.dot(theta)
    return np.sum(residual ** 2) / (2 * n) + (mu / 2) * np.sum(theta ** 2)

# Original analytic gradient (unused when using finite differences)
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

# Original analytic gradient (unused when using finite differences)
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
    alpha = 2.0    # block decay parameter for feature variances
    mu = 1 / n       # regularization parameter
    np.random.seed(2)
    
    # Choose the gradient estimator order: 'zeroth' or 'first'
    order = 'first'  # change to 'first' to use coordinate-wise finite differences
    
    # Generate data
    X, y, theta_true = generate_data(n, p, alpha, model=model_type)
    
    # Initialization
    theta0 = np.zeros(p)
    
    # Set up objective and gradient functions for the chosen model and gradient estimator.

    if model_type == 'ridge':
            f = lambda theta: f_ridge(theta, X, y, mu)
            theta_star = np.linalg.solve((X.T @ X) / n + mu * np.eye(p), (X.T @ y) / n)

    elif model_type == 'logistic':
        f = lambda theta: f_logistic(theta, X, y, mu)
        grad = lambda theta, k=None: grad_logistic(theta, X, y, mu, k)
        res = minimize(f, theta0, jac=grad, method='L-BFGS-B')
        theta_star = res.x

    if order == 'zeroth':
        grad = lambda theta, k=None: (
            grad_zeroth(f, theta, h=1e-5, num_samples=1)[k]
            if k is not None else grad_zeroth(f, theta, h=1e-5, num_samples=1)
        )
    
    elif order == 'first':
        grad = lambda theta, k=None: grad_first(f, theta, h=1e-5, k=k)
    else:
        raise ValueError("Unknown gradient order. Choose 'zeroth' or 'first'.")

    # Total coordinate evaluations for fair comparison
    total_evals = 100000
    # For uniform coordinate descent, one step = one coordinate evaluation.
    steps_cd = total_evals
    # For full gradient descent, one iteration uses p evaluations.
    steps_gd = total_evals // p  # integer division
    # For MUSKETEER, let T ≈ sqrt(p) and epochs such that total evaluations ≈ total_evals.
    T = int(np.sqrt(p))
    epochs_musketeer = total_evals // T
    # Choose learning rate (may need tuning)
    gamma_0 = 10  # 3 for ridge, 10 for logistic
    k0 = 5    # 10 for ridge, 5 for logistic
    gamma = gamma_0 * p / (gamma_0 * p * k0 + np.arange(steps_cd))  # possible schedule; tuning may be required.
    gamma_gd = gamma_0 / (gamma_0 * k0 + np.arange(steps_gd))
    print(gamma[0], gamma[-1], gamma_gd[0], gamma_gd[-1])
    gamma = 0.1
    gamma_gd = 0.1

    # For MUSKETEER, set lambda sequence and eta parameter.
    lambda_val = 1 #00
    lambda_base = 3
    lambda_seq = 1 / np.log(lambda_base + lambda_val * np.arange(epochs_musketeer))
    print(lambda_seq[0], lambda_seq[-1])
    #lambda_seq = 0.5  # for softmax
    eta = 1.0
    gain_type = 'abs'  # could be 'avg' or 'sqr' as well

    # Run Full Gradient Descent
    theta_gd, history_gd, evals_gd = gradient_descent(theta0, f, grad, steps_gd, gamma_gd)
    print("Gradient Descent final difference:", f(theta_gd) - f(theta_star))

    # Run Uniform Coordinate Descent
    theta_cd, history_cd, evals_cd = uniform_coordinate_descent(theta0, f, grad, steps_cd, gamma)
    print("Uniform Coordinate Descent final difference:", f(theta_cd) - f(theta_star))    

    # Run MUSKETEER
    theta_musk, history_musk, evals_musk = musketeer(theta0, f, grad, epochs_musketeer, T, gamma, lambda_seq, eta, gain_type, norm='l1')
    print("MUSKETEER final difference:", f(theta_musk) - f(theta_star))

    gain_type = 'avg'
    theta_musk, history_musk_avg, evals_musk = musketeer(theta0, f, grad, epochs_musketeer, T, gamma, lambda_seq, eta, gain_type, norm='l1')
    print("MUSKETEER final difference:", f(theta_musk) - f(theta_star))

    gain_type = 'sqr'
    theta_musk, history_musk_sqr, evals_musk = musketeer(theta0, f, grad, epochs_musketeer, T, gamma, lambda_seq, eta, gain_type, norm='l1')
    print("MUSKETEER final difference:", f(theta_musk) - f(theta_star))
    
    # -------------------------
    # Plotting Results
    # -------------------------
    
    def moving_average(data, window_size):
        """Compute the moving average of a 1D array with boundary adjustments.
        
        The moving average at each index i is computed over the window:
        [max(0, i - (window_size-1)//2) : min(len(data), i + window_size//2 + 1)]
        so that near the edges the average is taken over a smaller set of values.
        """
        n = len(data)
        result = np.empty(n)
        # Compute the cumulative sum with a zero prepended
        cumsum = np.cumsum(np.insert(data, 0, 0))
        half1 = (window_size - 1) // 2
        half2 = window_size // 2
        for i in range(n):
            start = max(0, i - half1)
            end = min(n, i + half2 + 1)
            window_sum = cumsum[end] - cumsum[start]
            result[i] = window_sum / (end - start)
        return result

    # Set the window size for the moving average (adjust as desired)
    window_size = 20

    y_cd = np.array(history_cd) - f(theta_star)
    smooth_y_cd = moving_average(y_cd, window_size)
    y_gd = np.array(history_gd) - f(theta_star)
    smooth_y_gd = moving_average(y_gd, window_size)
    y_musk = np.array(history_musk) - f(theta_star)
    smooth_y_musk = moving_average(y_musk, window_size)
    y_musk_avg = np.array(history_musk_avg) - f(theta_star)
    smooth_y_musk_avg = moving_average(y_musk_avg, window_size)
    y_musk_sqr = np.array(history_musk_sqr) - f(theta_star)
    smooth_y_musk_sqr = moving_average(y_musk_sqr, window_size)

    plt.figure(figsize=(8, 5))
    plt.semilogy(evals_cd, smooth_y_cd, label='Uniform coordinate descent')
    plt.semilogy(evals_gd, smooth_y_gd, label='Gradient descent')
    plt.semilogy(evals_musk, smooth_y_musk, label='MUSKETEER abs')
    plt.semilogy(evals_musk, smooth_y_musk_avg, label='MUSKETEER avg')
    plt.semilogy(evals_musk, smooth_y_musk_sqr, label='MUSKETEER sqr')
    plt.xlabel('Coordinate Evaluations')
    plt.ylabel('Objective Value')
    plt.title(f"Regularized {model_type.capitalize()} Regression, " + r"$\alpha=$" + str(int(alpha)))
    plt.legend()
    plt.grid(True)
    plt.savefig(model_type + "_alpha_" + str(int(alpha)))
    plt.show()
