
import numpy as np
import matplotlib.pyplot as plt
import sghmc
import timeit
import pandas as pd

# Create data

n = 500
p = 50

beta = np.random.normal(0, 1, p+1)

Sigma = np.zeros((p, p))
Sigma_diags = np.array([25, 5, 0.2**2])
distribution = np.random.multinomial(p, pvals=[.05, .05, .9], size=1).tolist()
np.fill_diagonal(Sigma, np.repeat(Sigma_diags, distribution[0], axis=0))

X = np.random.multivariate_normal(np.zeros(p), Sigma, n)
X = np.hstack((np.ones((n, 1)), X))
p = np.exp(X @ beta)/np.exp(1 + np.exp(X @ beta))
Y = np.random.binomial(1, p, n)

# Scale data

Xs = (X - np.mean(X, axis=0))/np.concatenate((np.ones(1),np.std(X[:,1:], axis=0)))
Xs = Xs[:,1:]
p = Xs.shape[1]






# ### Regression


from sklearn.linear_model import LogisticRegression


# Unscaled
mod_logis = LogisticRegression(fit_intercept=False, C=1e50)
mod_logis.fit(X,Y)
beta_true_unscale = mod_logis.coef_.ravel()
beta_true_unscale


# In[6]:

# Scaled
mod_logis = LogisticRegression(fit_intercept=False, C=1e50)
mod_logis.fit(Xs,Y)
beta_true_scale = mod_logis.coef_.ravel()
beta_true_scale






# ### HMC


# HMC - Scaled
nsample = 1000
m = 20
eps = .001
theta = np.zeros(p)
#theta = beta_true_unscale.copy()
phi = 5
M = np.identity(p)

samples, accept, rho, H = sghmc.run_hmc(Y, Xs, sghmc.U_logistic, sghmc.gradU_logistic, M, eps, m, theta, phi, nsample)

beta_est_hmc = np.mean(samples, axis=0)
beta_est_hmc - beta_true_scale



plt.plot((samples - beta_true_scale)[:,0])
plt.tight_layout()
plt.savefig('hmc-trace-sim.pdf')


fig, ax = plt.subplots(figsize=(4,3))
ax.plot(H)
ax.set_title("Total energy")
ax.set_xlabel("Number of samples")
plt.tight_layout()
plt.savefig('hmc-energy-sim.pdf')







# ### SGHMC



# HMC - Scaled (no intercept)
nsample = 1000
m = 20
eps = .002
theta = np.zeros(p)
#theta = beta_true_scale.copy()
phi = 5
nbatch = 500
C = 1 * np.identity(p)
V = 0 * np.identity(p)
M = np.identity(p)

samples_sghmc, H_sghmc = sghmc.run_sghmc(Y, Xs, sghmc.U_logistic, sghmc.stogradU_logistic, M, eps, m, theta, C, V, phi, nsample, nbatch)
beta_est_sghmc = np.mean(samples_sghmc, axis=0)
np.mean(samples_sghmc, axis=0) - beta_true_scale



plt.plot((samples_sghmc - beta_true_scale)[:,0])
plt.tight_layout()
plt.savefig('sghmc-trace-sim.pdf')



plt.plot(H_sghmc)
plt.tight_layout()
plt.savefig('sghmc-energy-sim.pdf')




# ### Gradient Descent

# Gradient descent - Scaled
np.random.seed(2)
phi = .1

beta_est_gd = sghmc.gd(Y, Xs, sghmc.gradU_logistic, .1, 10000, np.zeros(p), phi)

beta_est_gd - beta_true_scale


df = pd.DataFrame(np.vstack((beta_true_scale, 
                  beta_est_hmc, 
                  beta_est_sghmc, 
                  beta_est_gd)).T,
                  columns=['MLE','HMC','SGHMC','GD'])
df.plot()
ax.set_title("Coefficient Estimates")
ax.set_xlabel("Coefficient")
plt.tight_layout()
plt.savefig('coefs-sim.pdf')


