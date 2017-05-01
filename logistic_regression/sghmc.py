import numpy as np

def logistic(x):
    '''
    logistic function
    '''
    return 1/(1+np.exp(-x))

def U_logistic(theta, Y, X, phi):
    '''
    Potential energy function for logistic.
    Equal to the negative of the log posterior.
    Inputs:
        - theta: coefficients (p x 1)
        - Y: labels (n x p) 
        - X: design matrix (n x p) 
        - phi: prior precision
    Output:
        - energy
    '''
    return - (Y.T @ X @ theta - np.sum(np.log(1+np.exp(X @ theta))) - 0.5 * phi * np.sum(theta**2))

def gradU_logistic(theta, Y, X, phi):
    '''
    Gradient of the potential energy function for logistic.
    Inputs:
        - theta: coefficients (p x 1)
        - Y: labels (n x p) 
        - X: design matrix (n x p) 
        - phi: prior precision
    Output:
        - gradient (p x 1)
    '''
    n = X.shape[0]
    
    Y_pred = logistic(X @ theta)
    epsilon = (Y[:,np.newaxis] - Y_pred[:,np.newaxis])
    grad = X.T @ epsilon - phi * theta[:, np.newaxis]

    return -grad/n


def hmc(Y, X, U, gradU, M, eps, m, theta0, phi):
    '''
    Hamiltonian Monte Carlo
    Inputs:
        - Y: labels (n x p) 
        - X: design matrix (n x p)
        - U: potential energy function
        - gradU: gradient of potential energy function
        - M: Mass matrix
        - eps: time step rate
        - m: number of time steps in Hamiltonian dynamics
        - theta0: coefficients
        - phi: prior precision
    Output:
        - theta: estimated coefficients (p x 1)
        - accept: whether or not step accepted
        - rho: acceptance probability
        - H: Total energy after time steps
    '''
    theta = theta0.copy()
    n, p = X.shape
    
    # Precompute
    Minv = np.linalg.inv(M)
    
    # Randomly sample momentum
    r = np.random.multivariate_normal(np.zeros(p),M)[:,np.newaxis]
    
    # Intial energy
    H0 = U(theta0, Y, X, phi) + 0.5 * np.asscalar(r.T @ Minv @ r)
    
    # Hamiltonian dynamics
    r -= (eps/2)*gradU(theta, Y, X, phi)
    for i in range(m):
        theta += (eps*Minv@r).ravel()
        r -= eps*gradU(theta, Y, X, phi)
    r -= (eps/2)*gradU(theta, Y, X, phi)
    
    # Final energy
    H1 = U(theta, Y, X, phi) + np.asscalar(0.5 * r.T @ Minv @ r)
    
    # MH step
    u = np.random.uniform()
    rho = np.exp(H0 - H1) # Acceptance probability
    
    if u < np.min((1, rho)):
        # accept
        accept = True
        H = H1
    else:
        # reject
        theta = theta0
        accept = False
        H = H0

    return theta, accept, rho, H


def run_hmc(Y, X, U, gradU, M, eps, m, theta, phi, nsample):
    '''
    Wrapper function for hmc. All inputs/outputs are the same,
    except nsample, which is the number of hmc steps.
    '''

    n, p = X.shape
    
    # Allocate space
    samples = np.zeros((nsample, p))
    accept = np.zeros(nsample)
    rho = np.zeros(nsample)
    H = np.zeros(nsample)
    
    # Run hmc
    for i in range(nsample):
        theta, accept[i], rho[i], H[i] = hmc(Y, X, U, gradU, M, eps, m, theta, phi)
        samples[i] = theta
        
    return samples, accept, rho, H  

def stogradU_logistic(theta, Y, X, nbatch, phi):
    '''
    Negative of the stochastic gradient for logistic
    Inputs:
        - theta: coefficients
        - Y: labels (n x p) 
        - X: design matrix (n x p)
        - nbatch: number of observations used to calculate the gradient
        - phi: prior precision
    Output:
        - stochastic gradient
    '''
    n, p = X.shape
    
    # Sample minibatch
    batch_id = np.random.choice(np.arange(n),nbatch,replace=False)
    
    Y_pred = logistic(X[batch_id,:] @ theta[:,np.newaxis])
    epsilon = (Y[batch_id,np.newaxis] - Y_pred)
    grad = n/nbatch * X[batch_id,:].T @ epsilon - phi * theta[:, np.newaxis]

    return -grad

def sghmc(Y, X, U, gradU, M, Minv, eps, m, theta, C, B, D, phi, nbatch):
    '''
    Stochastic Gradient Descent Hamiltonian Monte Carlo
    Inputs:
        - Y: labels (n x p) 
        - X: design matrix (n x p)
        - U: potential energy function
        - gradU: gradient of potential energy function
        - M: Mass matrix
        - M: inverse of M
        - eps: time step rate
        - m: number of time steps in Hamiltonian dynamics
        - theta: coefficients
        - C: user specified friction
        - B: estimated noise
        - D: 2*(C-B)*eps (precalculated)
        - phi: prior precision
        - nbatch: number of observations used for the gradient
    Output:
        - theta: estimated coefficients (p x 1)
        - accept: whether or not step accepted
        - rho: acceptance probability
        - H: Total energy after time steps
    '''
    n, p = X.shape
    
    # Randomly sample momentum
    r = np.random.multivariate_normal(np.zeros(p),M)[:,np.newaxis]
    
    # Hamiltonian dynamics
    for i in range(m):        
        theta += (eps*Minv@r).ravel()
        r -= eps*gradU(theta, Y, X, nbatch,phi) - eps*C @ Minv @ r \
            + np.random.multivariate_normal(np.zeros(p),D)[:,np.newaxis] 
    
    # Record the energy
    H = U(theta, Y, X, phi) + np.asscalar(0.5 * r.T @ Minv @ r)
    
    return theta, H

def run_sghmc(Y, X, U, gradU, M, eps, m, theta, C, V, phi, nsample, nbatch):
    '''
    Wrapper function for sghmc. All inputs/outputs are the same,
    except nsample, which is the number of hmc steps.
    '''
    n, p = X.shape
    
    # Precompute
    Minv = np.linalg.inv(M)
    B = 0.5 * V * eps
    D = 2*(C-B)*eps
    
    # Allocate space
    samples = np.zeros((nsample, p))
    H = np.zeros(nsample)
    
    # Run sghmc
    for i in range(nsample):
        theta, H[i] = sghmc(Y, X, U, gradU, M, Minv, eps, m, theta, C, B, D, phi, nbatch)
        samples[i] = theta
        
    return samples, H
    
    
def gd(Y, X, gradU, eps, nsample, theta, phi):
    '''
    Gradient Descent
    Inputs:
        - Y: labels (n x p) 
        - X: design matrix (n x p)
        - gradU: gradient of potential energy function
        - eps: time step rate
        - m: number of time steps
        - theta: coefficients
        - phi: prior precision
    Output:
        - theta after gradient descent
    '''
    p = X.shape[1]
    samples = np.zeros((nsample, p))
    
    for i in range(nsample):
        theta -= eps*gradU(theta, Y, X, phi).ravel()
      
    return theta     