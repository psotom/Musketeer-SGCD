import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


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

# Zeroth order: Random direction finite-difference estimator.
# Only uses function evaluations.
def grad_zeroth(f, theta, h=1e-5, num_samples=1):
    grad_est = np.zeros_like(theta)
    for _ in range(num_samples):
        u = np.random.randn(*theta.shape)
        u /= np.linalg.norm(u)
        grad_est += (f(theta + h * u) - f(theta)) / h * u
    return grad_est / num_samples

# First order: Coordinate-wise central finite-difference estimator.
def grad_first(f, theta, h=1e-5, k=None):
    if k is not None:
        e = np.zeros_like(theta)
        e[k] = 1
        grad = (f(theta + h * e) - f(theta - h * e)) / (2 * h)
        if np.isnan(grad).any():
            print("NAAAAAAANS first")
            print(f(theta + h * e), f(theta - h * e), h)
        return grad

    grad_est = np.zeros_like(theta)
    for i in range(len(theta)):
        e = np.zeros_like(theta)
        e[i] = 1
        grad_est[i] = (f(theta + h * e) - f(theta - h * e)) / (2 * h)
    return grad_est

def adam_grad_first(f, theta, m, v, t, h=1e-5, k=None, 
                      beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Compute finite-difference gradient estimates and update Adam moment estimates
    for a given coordinate or for all coordinates.
    
    Parameters:
      f       : function, the objective function to evaluate at theta
      theta   : np.array, the current parameter vector
      m       : np.array, first moment estimates (should have same shape as theta)
      v       : np.array, second moment estimates (should have same shape as theta)
      t       : int, current time step (>=1) for bias correction
      h       : float, finite difference step size
      k       : int or None, if int then update only coordinate k, else update all coordinates
      beta1   : float, exponential decay rate for the first moment estimates
      beta2   : float, exponential decay rate for the second moment estimates
      epsilon : float, a small constant for numerical stability
      
    Returns:
      If k is provided: the Adam-adjusted gradient for coordinate k.
      If k is None: an array of Adam-adjusted gradients for all coordinates.
      
    Note:
      In your optimization loop you would typically update theta as:
          theta[k] = theta[k] - lr * (m_hat / (sqrt(v_hat) + epsilon))
      where m_hat and v_hat are the bias-corrected moment estimates.
    """
    
    if k is not None:
        # Create a basis vector for coordinate k
        e = np.zeros_like(theta)
        e[k] = 1
        # Compute finite difference for coordinate k
        grad = (f(theta + h * e) - f(theta - h * e)) / (2 * h)
        if np.isnan(grad):
            print(f"NaN encountered for coordinate {k} with f(theta+h*e)={f(theta + h * e)} and f(theta-h*e)={f(theta - h * e)}")
            return None
        
        # Update the first and second moments for coordinate k
        m[k] = beta1 * m[k] + (1 - beta1) * grad
        v[k] = beta2 * v[k] + (1 - beta2) * grad**2
        # Compute bias-corrected moment estimates
        m_hat = m[k] / (1 - beta1**t[k])
        v_hat = v[k] / (1 - beta2**t[k])
        t[k] += 1
        # Return the coordinate-wise Adam gradient (the step factor)
        return m_hat / (np.sqrt(v_hat) + epsilon)
    
    else:
        # Compute for all coordinates
        adam_grad = np.zeros_like(theta)
        for i in range(len(theta)):
            e = np.zeros_like(theta)
            e[i] = 1
            grad = (f(theta + h * e) - f(theta - h * e)) / (2 * h)
            if np.isnan(grad):
                print(f"NaN encountered for coordinate {i}")
                continue  # or handle as needed
            # Update moments for coordinate i
            m[i] = beta1 * m[i] + (1 - beta1) * grad
            v[i] = beta2 * v[i] + (1 - beta2) * grad**2
            # Bias correction
            m_hat = m[i] / (1 - beta1**t)
            v_hat = v[i] / (1 - beta2**t)
            adam_grad[i] = m_hat / (np.sqrt(v_hat) + epsilon)

        return adam_grad


# -------------------------
# Descent methods
# -------------------------

def uniform_coordinate_descent(x0, f, grad_f, steps, gamma, eval_each=1, callback=None, n_calls=1):
    """
    Uniform coordinate descent.
    At each iteration one coordinate is sampled uniformly and updated.
    """

    x = x0.copy()
    gamma = np.atleast_1d(gamma)
    p = len(x)
    history = []
    coord_evals = []
    eval_count = 0
    evals = steps // eval_each
    for ev in tqdm(range(evals)):
        history.append(f(x))
        for t in range(ev * eval_each, (ev + 1) * eval_each):
            k = np.random.randint(0, p)
            g = grad_f(x, k)
            gamma_t = gamma[t] if t < len(gamma) else gamma[-1]
            x[k] = x[k] - gamma_t * g
        eval_count += eval_each
        coord_evals.append(eval_count)

        if callback is not None and ev % n_calls == 0:
            callback(x, eval_count)

    return x, np.array(history), np.array(coord_evals)

def gradient_descent(x0, f, grad_f, steps, gamma, callback=None):
    """
    Full gradient descent.
    Each iteration uses the full gradient (p coordinate evaluations).
    """
    x = x0.copy()
    history = []
    coord_evals = []
    eval_count = 0
    gamma = np.atleast_1d(gamma)
    for t in tqdm(range(steps)):
        history.append(f(x))
        g = grad_f(x)
        gamma_t = gamma[t] if t < len(gamma) else gamma[-1]
        x = x - gamma_t * g
        eval_count += len(x)  # full gradient evaluation counts as p evaluations
        coord_evals.append(eval_count)

        if callback is not None:
            callback(x, eval_count)

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
    gamma = np.atleast_1d(gamma)
    p = len(x)
    gain_vec = np.zeros(p)
    evals = 0
    for t in range(T):
        # Sample a coordinate index k according to distribution d.
        k = np.random.choice(p, p=d)
        g = grad_f(x, k)
        gamma_t = gamma[t] if t < len(gamma) else gamma[-1]
        x[k] = x[k] - gamma_t * g
        
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

def exploit_phase(G, gain_phase, n, lambda_n, eta, r=0.8, norm='softmax'):
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

    #G_new = G + (gain_phase - G) / (n + 1)
    G_new = G + (gain_phase - G) / (r * n + 1)

    if norm == 'l1':
        d_g = np.abs(G_new) / np.sum(np.abs(G_new))
    elif norm == 'softmax':
        exp_etaG = np.exp(eta * G_new)
        d_g = exp_etaG / np.sum(exp_etaG)
    else:
        raise ValueError

    p = len(G_new)
    d_new = (1 - lambda_n) * d_g + lambda_n * (np.ones(p) / p)
    return d_new, G_new

def musketeer(x0, f, grad_f, epochs, T, gamma, lambda_seq, eta, gain_type='abs', norm='softmax', callback=None, n_calls=1):
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
    gamma = np.atleast_1d(gamma)
    lambda_seq = np.atleast_1d(lambda_seq)


    history = []
    evals = []
    eval_count = 0

    for n in tqdm(range(epochs)):
        gamma_n = gamma[n * T: (n + 1) * T] if (n + 1) * T < len(gamma) else np.atleast_1d(gamma[-1])
        x, gain_phase, evals_phase = explore_phase(x, d, T, gamma_n, grad_f, gain_type)
        eval_count += evals_phase
        history.append(f(x))
        evals.append(eval_count)
        
        if np.isnan(gain_phase).any():
            print("NANS GAIN PHASE")

        lambda_n = lambda_seq[n] if n < len(lambda_seq) else lambda_seq[-1]
        d, G = exploit_phase(G, gain_phase, n, lambda_n, eta, norm=norm)

        #print("iter: ", n + 1, " max diff is :", np.max(d) - np.min(d), "from coordinate ", np.argmax(d), "max gain is: ", np.max(G))

        if callback is not None and n % n_calls == 0:
            callback(x, eval_count)

    return x, history, evals

def explore_phase2(x, d, T, gamma, grad_f, gain_type):
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
    gamma = np.atleast_1d(gamma)
    p = len(x)
    gain_vec = np.zeros(p)
    evals = 0
    gain_app = np.zeros(p)

    for t in range(T):
        # Sample a coordinate index k according to distribution d.
        k = np.random.choice(p, p=d)
        g = grad_f(x, k)
        gamma_t = gamma[t] if t < len(gamma) else gamma[-1]
        x[k] = x[k] - gamma_t * g
        gain_app[k] += 1

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
    
    gain_vec /= p
    return x, gain_vec, gain_app, evals

def exploit_phase2(G, gain_phase, app, gain_app, lambda_n, eta=1.0, r=0.8, norm='softmax'):
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

    #G_new = G + (gain_phase - G) / (n + 1)
    G_new = (G * app + gain_phase) / (app + gain_app)

    if norm == 'l1':
        d_g = np.abs(G_new) / np.sum(np.abs(G_new))
    elif norm == 'softmax':
        exp_etaG = np.exp(eta * G_new)
        d_g = exp_etaG / np.sum(exp_etaG)
    else:
        raise ValueError
    
    p = len(G_new)
    d_new = (1 - lambda_n) * d_g + lambda_n * (np.ones(p) / p)
    return d_new, G_new

def musketeer2(x0, f, grad_f, epochs, T, gamma, lambda_seq, eta, gain_type='abs', callback=None, n_calls=1):
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
    gamma = np.atleast_1d(gamma)
    lambda_seq = np.atleast_1d(lambda_seq)
    appearences = np.ones(p)

    history = []
    evals = []
    eval_count = 0

    for n in tqdm(range(epochs)):
        gamma_n = gamma[n * T: (n + 1) * T] if (n + 1) * T < len(gamma) else np.atleast_1d(gamma[-1])
        x, gain_phase, gain_app, evals_phase = explore_phase2(x, d, T, gamma_n, grad_f, gain_type)
        eval_count += evals_phase
        history.append(f(x))
        evals.append(eval_count)
        
        lambda_n = lambda_seq[n] if n < len(lambda_seq) else lambda_seq[-1]
        d, G = exploit_phase2(G, gain_phase, appearences, gain_app, lambda_n, eta)
        appearences += gain_app

        print("iter: ", n + 1, " max diff is :", np.max(d) - np.min(d), "from coordinate ", np.argmax(d), "max gain is: ", np.max(G))
        print("appears: ", appearences[:5], np.max(appearences))
        if callback is not None and n % n_calls == 0:
            callback(x, eval_count)

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
    epochs = 2000              # number of outer iterations (each with T exploration steps)
    T = int(np.sqrt(p))
    steps = epochs * T
    gamma_0 = 10
    gamma = gamma_0 * p / (gamma_0 * p * L_max + np.arange(steps))   # this is one possible choice; tuning may be required.
    gamma_gd = gamma_0 / (gamma_0 * L_max + np.arange(epochs))
    #gamma = 1.0 / (L_max * p + np.zeros(epochs))

    T = int(np.sqrt(p))       # exploration phase length (as suggested in the paper)
    lambda_seq = 0.1 * np.ones(epochs)  # fixed exploration weight (can also be scheduled, e.g., 1/log(n))
    eta = 1.0                 # softmax parameter for probability update
    gain_type = 'abs'         # choose from 'avg', 'abs', 'sqr'
    
    x_final, history, evals = musketeer(x0, f_gap, grad_coord, epochs, T, gamma, lambda_seq, eta, gain_type)
    x_final, history_sqr, evals_sqr = musketeer(x0, f_gap, grad_coord, epochs, T, gamma, lambda_seq, eta, 'sqr')
    x_final, history_avg, evals_avg = musketeer(x0, f_gap, grad_coord, epochs, T, gamma, lambda_seq, eta, 'avg')
    x_ucd, history_ucd, evals_ucd = uniform_coordinate_descent(x0, f_gap, grad_coord, epochs * T, gamma)
    x_gcd, history_gcd, evals_gcd = gradient_descent(x0, f_gap, grad, int(epochs * T / p), gamma_gd)
    
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
