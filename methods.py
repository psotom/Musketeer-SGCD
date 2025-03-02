import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Quadratic Example
# -------------------------
def quadratic_function(x, A, b):
    """
    Quadratic objective: f(x) = 0.5 * x^T A x - b^T x.
    """
    return 0.5 * np.dot(x, A @ x) - np.dot(b, x)

def quadratic_gradient(x, A, b):
    """
    Gradient of the quadratic function: ∇f(x) = A x - b.
    """
    return A @ x - b

# -------------------------
# Descent methods
# -------------------------

def uniform_coordinate_descent(x0, f, grad_f, steps, gamma):
    """
    Uniform coordinate descent.
    At each iteration one coordinate is sampled uniformly and updated.
    """
    x = x0.copy()
    p = len(x)
    history = []
    coord_evals = []
    eval_count = 0
    for t in range(steps):
        k = np.random.randint(0, p)
        g = grad_f(x, k)
        x[k] = x[k] - gamma * g
        eval_count += 1  # one coordinate evaluation
        history.append(f(x))
        coord_evals.append(eval_count)
    return x, np.array(history), np.array(coord_evals)

def gradient_descent(x0, f, grad_f, steps, gamma):
    """
    Full gradient descent.
    Each iteration uses the full gradient (p coordinate evaluations).
    """
    x = x0.copy()
    history = []
    coord_evals = []
    eval_count = 0
    for t in range(steps):
        g = grad_f(x)
        x = x - gamma * g
        eval_count += len(x)  # full gradient evaluation counts as p evaluations
        history.append(f(x))
        coord_evals.append(eval_count)
    return x, np.array(history), np.array(coord_evals)

# -------------------------
# MUSKETEER
# -------------------------

def explore_phase(x, d, T, gamma, grad_f, gain_type):
    """
    Exploration phase of MUSKETEER.
    
    For T steps:
      - Sample a coordinate k according to distribution d.
      - Compute the gradient at coordinate k.
      - Update x at coordinate k with an descent step: x[k] = x[k] - gamma * g.
      - Accumulate the gain for coordinate k.
      
    The gain update rule is selected by gain_type:
      - 'avg': raw gradient value,
      - 'abs': absolute value,
      - 'sqr': squared value.
      
    Returns:
      x        : updated iterate after exploration
      gain_vec : vector of averaged gains computed over T steps.
      evals    : number of coordinate evaluations (equals T here).
    """
    p = len(x)
    gain_vec = np.zeros(p)
    evals = 0
    for t in range(T):
        # Sample a coordinate index k according to distribution d.
        k = np.random.choice(p, p=d)
        g = grad_f(x, k)
        x[k] = x[k] - gamma * g
        
        # Accumulate gain for coordinate k.
        # Importance sampling update: scale the coordinate update by 1/d[k]
        if gain_type == 'abs':
            gain_vec[k] += abs(g) / d[k]
        elif gain_type == 'sqr':
            gain_vec[k] += (g**2) / d[k]
        elif gain_type == 'avg':
            gain_vec[k] += g / d[k]
        else:
            raise ValueError("Invalid gain_type. Use 'avg', 'abs', or 'sqr'.")
        evals += 1
    
    gain_vec /= T
    return x, gain_vec, evals

def exploit_phase(G, gain_phase, n, lambda_n, eta):
    """
    Exploitation phase: update cumulative gains and the sampling distribution.
    
    - Update the cumulative gain G with a running average:
         G_new = G + (gain_phase - G) / (n + 1)
    - Compute a normalized version of G. We use a softmax operator with parameter eta.
    - Mix the softmax probabilities with a uniform distribution using parameter lambda_n:
         d_new = (1 - lambda_n) * softmax(eta * G_new) + lambda_n * (1/p)
    
    Returns:
      d_new : updated coordinate sampling distribution.
      G_new : updated cumulative gain vector.
    """

    G_new = G + (gain_phase - G) / (n + 1)
    exp_etaG = np.exp(eta * G_new)
    softmax = exp_etaG / np.sum(exp_etaG)
    p = len(G_new)
    d_new = (1 - lambda_n) * softmax + lambda_n * (np.ones(p) / p)
    return d_new, G_new

def musketeer(x0, f, grad_f, epochs, T, gamma, lambda_seq, eta, gain_type='abs'):
    """
    Implementation of MUSKETEER algorithm.
    
    Parameters:
      x0        : initial point in R^p.
      f         : objective function.
      grad_f    : function to compute gradient.
      epochs    : number of outer iterations (each with an exploration phase).
      T         : number of exploration steps per epoch (e.g., T ≈ sqrt(p)).
      gamma     : step size for coordinate update.
      lambda_seq: sequence (or constant) controlling the closeness of the
                sampling distribution to the uniform one.
      eta       : softmax parameter for normalizing cumulative gains.
      gain_type : type of gain update ('avg', 'abs', or 'sqr').
      
    Returns:
      x         : final point.
      history   : list of f(x) values recorded after every coordinate evaluation.
      evals     : list of cumulative coordinate evaluations.
    """
    p = len(x0)
    d = np.ones(p) / p          # d0 = (1/p,...,1/p)
    G = np.zeros(p)             # G0 = (0,...,0)
    x = x0.copy()
    
    history = []
    evals = []
    eval_count = 0

    for n in range(epochs):
        x, gain_phase, evals_phase = explore_phase(x, d, T, gamma, grad_f, gain_type)
        eval_count += evals_phase
        history.append(f(x))
        evals.append(eval_count)
        
        lambda_n = lambda_seq[n] if n < len(lambda_seq) else lambda_seq[-1]
        d, G = exploit_phase(G, gain_phase, n, lambda_n, eta)

    return x, history, evals

# -------------------------
# Testing on a Quadratic Problem
# -------------------------
def run_musketeer_experiment():
    np.random.seed(0)
    p = 50
    # Create a diagonal matrix A with entries linearly spaced between 1 and 10.
    A = np.diag(np.linspace(1, 10, p))
    b = np.random.randn(p)
    
    # Compute the optimum: for a quadratic f(x)=0.5*x^T A x - b^T x, optimum is x* = A^{-1}b.
    x_star = np.linalg.solve(A, b)
    f_star = quadratic_function(x_star, A, b)
    
    # Define the function gap: f(x) - f*.
    f_gap = lambda x: quadratic_function(x, A, b) - f_star
    grad = lambda x: quadratic_gradient(x, A, b)
    grad_coord = lambda x, k: quadratic_gradient(x, A, b)[k]
    
    x0 = np.zeros(p)
    L_max = np.max(np.diag(A))
    # For our quadratic, a full gradient descent would use gamma = 1/L_max.
    # MUSKETEER is a coordinate descent algorithm so we need to devide by p
    gamma = 1.0 / (L_max * p)   # this is one possible choice; tuning may be required.
    
    epochs = 1000              # number of outer iterations (each with T exploration steps)
    T = int(np.sqrt(p))       # exploration phase length (as suggested in the paper)
    lambda_seq = 0.1 * np.ones(epochs)  # fixed exploration weight (can also be scheduled, e.g., 1/log(n))
    eta = 1.0                 # softmax parameter for probability update
    gain_type = 'abs'         # choose from 'avg', 'abs', 'sqr'
    
    x_final, history, evals = musketeer(x0, f_gap, grad_coord, epochs, T, gamma, lambda_seq, eta, gain_type)
    x_final, history_sqr, evals_sqr = musketeer(x0, f_gap, grad_coord, epochs, T, gamma, lambda_seq, eta, 'sqr')
    x_final, history_avg, evals_avg = musketeer(x0, f_gap, grad_coord, epochs, T, gamma, lambda_seq, eta, 'avg')
    x_ucd, history_ucd, evals_ucd = uniform_coordinate_descent(x0, f_gap, grad_coord, epochs * T, gamma)
    x_gcd, history_gcd, evals_gcd = gradient_descent(x0, f_gap, grad, int(epochs * T / p), gamma)
    
    print("Final optimality gap: {:.3e}".format(history[-1]))
    plt.figure(figsize=(8, 5))
    plt.semilogy(evals, history, label='MUSKETEER abs')
    plt.semilogy(evals_sqr, history_sqr, label='MUSKETEER sqr')
    plt.semilogy(evals_avg, history_avg, label='MUSKETEER avg')
    plt.semilogy(evals_ucd, history_ucd, label='Uniform Coordinate Descent')
    plt.semilogy(evals_gcd, history_gcd, label='Gradient Descent')
    plt.xlabel('Coordinate Evaluations')
    plt.ylabel('Optimality Gap f(x) - f(x*)')
    plt.title('MUSKETEER Convergence on Quadratic Problem')
    plt.legend()
    plt.grid(True)
    plt.show()
    
if __name__ == "__main__":
    run_musketeer_experiment()
