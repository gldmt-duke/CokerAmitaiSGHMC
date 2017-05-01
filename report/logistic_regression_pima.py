
import numpy as np
import matplotlib.pyplot as plt
import sghmc

pima = np.genfromtxt('pima-indians-diabetes.data', delimiter=',')
names = ["Number of times pregnant",
         "Plasma glucose concentration",
         "Diastolic blood pressure (mm Hg)",
         "Triceps skin fold thickness (mm)",
         "2-Hour serum insulin (mu U/ml)",
         "Body mass index (weight in kg/(height in m)^2)",
         "Diabetes pedigree function",
         "Age (years)",
         "Class variable (0 or 1)"]

# Load data
X = np.concatenate((np.ones((pima.shape[0],1)),pima[:,0:8]), axis=1)
Y = pima[:,8]

Xs = (X - np.mean(X, axis=0))/np.concatenate((np.ones(1),np.std(X[:,1:], axis=0)))
Xs = Xs[:,1:]

n, p = Xs.shape

# ### Regression


from sklearn.linear_model import LogisticRegression



# Unscaled
mod_logis = LogisticRegression(fit_intercept=False, C=1e50)
mod_logis.fit(X,Y)
beta_true_unscale = mod_logis.coef_.ravel()
beta_true_unscale


# Scaled
mod_logis = LogisticRegression(fit_intercept=False, C=1e50)
mod_logis.fit(Xs,Y)
beta_true_scale = mod_logis.coef_.ravel()
beta_true_scale


# ### HMC



# HMC - Scaled
nsample = 10000
m = 20
eps = .001
theta = np.zeros(p)
#theta = beta_true_unscale.copy()
phi = 5
M = np.identity(p)

samples, accept, rho, H = sghmc.run_hmc(Y, Xs, sghmc.U_logistic, sghmc.gradU_logistic, M, eps, m, theta, phi, nsample)

beta_est_hmc = np.mean(samples, axis=0)
beta_est_hmc - beta_true_scale

fig, ax = plt.subplots(figsize=(4,3))
ax.plot(samples[:,0])
ax.set_title("Trace of First Coefficient")
ax.set_xlabel("Index of Samples")
plt.tight_layout()
plt.savefig('hmc-trace-pima.pdf')

fig, ax = plt.subplots(figsize=(4,3))
ax.plot(H)
ax.set_title("Total energy")
ax.set_xlabel("Index of Samples")
plt.tight_layout()
plt.savefig('hmc-energy-pima.pdf')



# ### SGHMC



# HMC - Scaled (no intercept)
nsample = 10000
m = 20
eps = .002
theta = np.zeros(p)
#theta = beta_true_scale.copy()
phi = 5
nbatch = 500
C = 1 * np.identity(p)
V = 0 * np.identity(p)
M = np.identity(p)

samples, H = sghmc.run_sghmc(Y, Xs, sghmc.U_logistic, sghmc.stogradU_logistic, M, eps, m, theta, C, V, phi, nsample, nbatch)

beta_est_sghmc = np.mean(samples, axis=0)
np.mean(samples, axis=0) - beta_true_scale



fig, ax = plt.subplots(figsize=(4,3))
ax.plot(samples[:,0])
ax.set_title("Trace of First Coefficient")
ax.set_xlabel("Index of Samples")
plt.tight_layout()
plt.savefig('sghmc-trace-pima.pdf')



fig, ax = plt.subplots(figsize=(4,3))
ax.plot(H)
ax.set_title("Total energy")
ax.set_xlabel("Index of Samples")
plt.tight_layout()
plt.savefig('sghmc-energy-pima.pdf')


import pandas as pd
np.random.seed(2)
phi = .1

beta_est_gd = sghmc.gd(Y, Xs, sghmc.gradU_logistic, .1, 10000, np.zeros(p), phi)

beta_est_gd - beta_true_scale


df = pd.DataFrame(np.vstack((beta_true_scale, 
                  beta_est_hmc, 
                  beta_est_sghmc, 
                  beta_est_gd)).T,
                  columns=['MLE','HMC','SGHMC','GD'])
fig, ax = plt.subplots(figsize=(4,3))
df.plot(ax=ax, fig=fig)
ax.set_title("Coefficient Estimates")
ax.set_xlabel("Coefficient")
plt.tight_layout()
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig('coefs-pima.pdf')