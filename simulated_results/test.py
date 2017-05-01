import numpy as np

### HMC version
def logistic(x):
    return 1/(1+np.exp(-x))

def U(theta, Y, X):
    return - (Y.T @ X @ theta - np.sum(np.log(1+np.exp(X @ theta))) - 0.5 * phi * np.sum(theta**2))

def gradU(theta, Y, X, nbatch):
    '''A function that returns the stochastic gradient. Adapted from Eq. 5.
    Inputs are:
        theta, the parameters
        Y, the response
        X, the covariates
        nbatch, the number of samples to take from the full data
    '''
    n = X.shape[0]
    
    Y_pred = logistic(X @ theta)
    epsilon = (Y[:,np.newaxis] - Y_pred[:,np.newaxis])
    grad = X.T @ epsilon - phi * theta[:, np.newaxis]

    return -grad/n
    #temp = -grad/n
    #return temp / np.linalg.norm(temp)


def hmc(Y, X, gradU, M, eps, m, theta, C, V):
    theta0 = theta.copy()
    
    # This is just HMC for testing
    n = X.shape[0]
    p = X.shape[1]
    
    # Precompute
    Minv = np.linalg.inv(M)
    
    # Randomly sample momentum
    r = np.random.multivariate_normal(np.zeros(p),M)[:,np.newaxis]
    
    # Intial energy
    H0 = U(theta, Y, X) + 0.5 * np.asscalar(r.T @ Minv @ r)
    
    # Hamiltonian dynamics
    r = r - (eps/2)*gradU(theta, Y, X, nbatch)
    for i in range(m):
        theta = theta + (eps*Minv@r).ravel()
        r = r - eps*gradU(theta, Y, X, nbatch)
    theta = theta + (eps*Minv@r).ravel()
    r = r - (eps/2)*gradU(theta, Y, X, nbatch)  
    
    # Final energy
    H1 = U(theta, Y, X) + np.asscalar(0.5 * r.T @ Minv @ r)
    
    # MH step
    u = np.random.uniform()
    #rho = np.exp(H1 - H0)
    rho = np.exp(H0 - H1)
    #print('(H0, H1, rho): %s,%s,%s' % (H0, H1, rho))
    
    if u < np.min((1, rho)):
        return theta.copy()
    else:
        return theta0.copy() # reject
        
        
    return theta

def my_gd(Y, X, gradU, M, eps, m, theta, C, V):
    # gradient descent
    n = X.shape[0]
    p = X.shape[1]
    
    for i in range(m):
        theta = theta - eps*gradU(theta, Y, X, nbatch).ravel()
        
    return theta