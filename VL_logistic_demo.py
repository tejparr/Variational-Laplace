import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import auxiliaryfunctions as aux
import VL_logistic
from auxiliaryfunctions import softmax
from BayesianModelReduction import reduce_gaussian

# Logistic regression demo for VL routine
#==============================================================================
# This script illustrates logistic regression (formulated to generalise to 
# multiple discrete outomes) using Variational Laplace.This uses a traditional 
# multinomial regression model. This suffers from a violation of the Laplace 
# assumption due to the nonlinear softmax function:
# 
# beta ~ N(pE,pC)
# X    = U@beta
# V    = softmax(X)
# Y    ~ Cat(V)
#
# Despite the violation of the Laplace assumption, this scheme successfully
# classifies the data-points

np.random.seed(1)  # for reproducibility

# Parameters and design matrix (U)
#------------------------------------------------------------------------------
U    = np.concatenate((np.ones((32,1)),5*np.random.randn(32,2)),axis = 1)
beta = np.array([[1],[-1]])
P    = {'beta':beta}  # Parameters
     
# Generate (categorical) data
#-----------------------------------------------------------------------------
b = np.concatenate((np.array([[0]]),P['beta']),axis = 0)
B = np.concatenate((np.zeros((np.size(b,0),1)),b),axis = 1)
B[0,0] = 1
X = U@B
V = softmax(X,1)
V = np.cumsum(V,axis = 1)
r = np.random.rand(np.size(U,0),1)
Y = np.zeros((np.size(U,0),1))
for i in range(np.size(V,0)):
    Y[i,0] = np.nonzero(r[i,0]<V[i,:])[0][0]

Ep,Cp,F = VL_logistic.invert(U,Y)

# Overlay true values on confidence intervals
#------------------------------------------------------------------------------
Fsi = plt.gcf()
ax  = Fsi.get_axes()
ax[-1].bar(range(aux.length(P)),np.array(aux.vec(P)).T[0],width = 1/4,color = [0.8,0.1,0.1],zorder = 3)
x0,x1 = ax[-1].get_xlim()
y0,y1 = ax[-1].get_ylim()
ax[-1].set_aspect(abs(x1-x0)/abs(y1-y0))

# Plot classification
#------------------------------------------------------------------------------
Flr = aux.figure('Classification')
Flr.clf()
ax = Flr.subplots(2,2)

col = [[0.8,0.5,0.5],[0.5,0.5,0.8]]

# Plot colour-coded data-points

for i in range(np.size(P['beta'],1)+1):
    a = np.nonzero(Y == i)
    ax[0,0].plot(U[a[0],1],U[a[0],2],'.',color = col[i],markersize = 16)
    ax[0,0].grid(True)
    ax[0,1].plot(U[a[0],1],U[a[0],2],'.',color = col[i],markersize = 16)
    ax[0,1].grid(True)
    ax[1,1].plot(U[a[0],1],U[a[0],2],'.',color = col[i],markersize = 16)
    ax[1,1].grid(True)

x0,x1 = ax[0,0].get_xlim()
y0,y1 = ax[0,0].get_ylim()

# Plot classification under 'true' parameters

m = -P['beta'][0,0]/P['beta'][1,0]
c = -1/P['beta'][1,0]
ax[0,0].plot(2*np.r_[x0:x1],2*np.r_[x0:x1]*m + c,'--',color = [0.5,0.5,0.5])
ax[0,0].set(xlim = (x0,x1), ylim = (y0,y1))
ax[0,0].set_aspect(abs(x1-x0)/abs(y1-y0))
ax[0,0].set(title = 'Simulated data')

# Plot classification under estimated parameters

m = -Ep['beta'][0,0]/Ep['beta'][1,0]
c = -1/Ep['beta'][1,0]
ax[0,1].plot(2*np.r_[x0:x1],2*np.r_[x0:x1]*m + c,'--',color = [0.5,0.5,0.5])
ax[0,1].set(xlim = (x0,x1), ylim = (y0,y1))
ax[0,1].set_aspect(abs(x1-x0)/abs(y1-y0))
ax[0,1].set(title = 'Classification')

# Plot the likelihood in parameter space

pE = aux.unvec(np.zeros((aux.length(P),1)),P)
pC = np.eye(aux.length(P))*512
M = {'L':VL_logistic.L,'pE':pE,'pC':pC}

d = 0.25
x = np.arange(x0,x1,d)
y = np.arange(y0,y1,d)
Z = np.zeros((np.size(x),np.size(y)))
for i in range(np.size(Z,0)):
    for j in range(np.size(Z,1)):
        p = {'beta':np.array([[x[i]],[y[j]]])}
        Z[i,j]    = VL_logistic.L(p,M,U,Y)
ax[1,0].imshow(Z, extent=[x0, x1, y0, y1],origin = 'lower', cmap = cm.gray, aspect = abs(x1-x0)/abs(y1-y0))
ax[1,0].set_aspect(abs(x1-x0)/abs(y1-y0))
ax[1,0].set(title = 'Likelihood map')


# Plot probabilistic classification

# See section 4.5.2 of Bishop for details of use of posterior covariance in
# computing predictive uncertainty.
x, y = np.meshgrid(x, y)
sigma  = Cp[0,0]*x**2 + 2*Cp[1,0]*x*y + Cp[1,1]*y**2
z      = Ep['beta'][0,0]*x + Ep['beta'][1,0]*y
Z      = np.exp(z/np.sqrt(1 + np.pi*sigma/8))
Z      = np.exp(1)/(Z + np.exp(1))
ax[1,1].imshow(Z, extent=[x0, x1, y0, y1],origin = 'lower',cmap = cm.gray, aspect = abs(x1-x0)/abs(y1-y0))
ax[1,1].set_aspect(abs(x1-x0)/abs(y1-y0))
ax[1,1].set(title = 'Probability map')

# Bayesian model reduction
#------------------------------------------------------------------------------
pE,pC = VL_logistic.default_priors(U,Y)
M = {'pE':pE,'pC':pC}
D = {'M':M,'Ep':Ep,'Cp':Cp}
BMR,BMA = reduce_gaussian(D,'all')