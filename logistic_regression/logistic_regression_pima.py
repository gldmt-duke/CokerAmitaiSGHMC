
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import sghmc


# In[2]:

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


# In[3]:

# Load data
X = np.concatenate((np.ones((pima.shape[0],1)),pima[:,0:8]), axis=1)
Y = pima[:,8]

Xs = (X - np.mean(X, axis=0))/np.concatenate((np.ones(1),np.std(X[:,1:], axis=0)))
Xs = Xs[:,1:]

n, p = Xs.shape


# In[ ]:




# In[ ]:




# In[ ]:




# ### Regression

# In[4]:

from sklearn.linear_model import LogisticRegression


# In[5]:

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


# In[ ]:




# In[ ]:




# In[ ]:




# ### HMC

# In[7]:

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


# In[9]:

plt.plot((samples - beta_true_scale)[:,1])
plt.savefig('hmc-trace.pdf')


# In[10]:

fig, ax = plt.subplots(figsize=(4,3))
ax.plot(H)
ax.set_title("Total energy")
ax.set_xlabel("Number of samples")
plt.savefig('hmc-energy.pdf')


# In[11]:

fig, ax = plt.subplots(3,1, figsize=(6,10))

i = 0
ax[0].plot((samples - beta_true_scale)[:,i])
ax[0].set_title(names[i])


i = 1
ax[1].plot((samples - beta_true_scale)[:,i])
ax[1].set_title(names[i])


i = 2
ax[2].plot((samples - beta_true_scale)[:,i])
ax[2].set_title(names[i])


# In[ ]:

fig, ax = plt.subplots()

ax.bar()


# In[ ]:




# ### SGHMC

# In[11]:

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

samples, H = sghmc.run_sghmc(Y, Xs, sghmc.U_logistic, sghmc.stogradU_logistic, M, eps, m, theta, C, V, phi, nsample, nbatch)

np.mean(samples, axis=0) - beta_true_scale


# In[12]:

plt.plot((samples - beta_true_scale)[:,0])
plt.savefig('sghmc-trace.pdf')


# In[13]:

plt.plot(H)
plt.savefig('sghmc-energy.pdf')


# In[28]:

fig, ax = plt.subplots(3,1, figsize=(6,10))

i = 0
ax[0].plot((samples - beta_true_scale)[:,i])
ax[0].set_title(names[i])


i = 1
ax[1].plot((samples - beta_true_scale)[:,i])
ax[1].set_title(names[i])


i = 2
ax[2].plot((samples - beta_true_scale)[:,i])
ax[2].set_title(names[i])


# ### Gradient Descent

# In[8]:

# Gradient descent - Scaled
np.random.seed(2)
phi = .1

res = sghmc.gd(Y, Xs, sghmc.gradU_logistic, .1, 10000, np.zeros(p), phi)

res - beta_true_scale


# In[ ]:




# In[ ]:




# In[ ]:



