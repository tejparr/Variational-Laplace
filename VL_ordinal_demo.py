import numpy as np
import matplotlib.pyplot as plt
import auxiliaryfunctions as aux
import VL_ordinal
from auxiliaryfunctions import softmax
from BayesianModelReduction import reduce_gaussian

# Ordinal data model demonstration for VL routine
#==============================================================================
# This demo illustrates the use of Variational Laplace to infer the causes of
# ordinal data (e.g., questionare scores). This uses two sorts of latent cause.
# The first is a set of coefficients (beta) that weight the columns of a design
# matrix, just as in linear models. The second set (theta) are the parameters 
# that divide up the real number line into categories. The first theta value is
# assigns everything below it to the first outcome. Subsequent values are the
# logs of the intervals separating subsequent outcomes.

np.random.seed(1) # for reproducibility

# Parameters and design matrix (U)
#------------------------------------------------------------------------------
N     = 128  # Number of measurements
K     = 4    # Number of categories - 1
U     = 5*np.random.randn(N,2)
beta  = np.array([[1],[-1]])         # Coefficients of linear model
theta = np.random.randn(K,1)         # Thresholds
P     = {'beta':beta,'theta':theta}  # Parameters

# Generate (ordinal) data
#-----------------------------------------------------------------------------

X  = U@P['beta']

# Convert intervals to thresholds
dX      = np.concatenate((np.array(P['theta'][0]),np.exp(P['theta'][1:K,0])),axis = 0)
dX      = np.cumsum(dX)
r = np.random.rand(np.size(U,0),1)
Y = np.zeros((np.size(U,0),1))
for i in range(np.size(X,0)):
    Z  = np.array(dX - X[i,0])
    V  = np.zeros((np.size(dX)+1,1))
    for j in range(np.size(Z,0)+1):
        if j == 0:
            V[j,0] = softmax(np.append(np.array([1]),Z[j]),0)[1]
        elif j == np.size(Z,0):
            V[j,0] = 1 - softmax(np.append(np.array([1]),Z[j-1]),0)[1]
        else:
            V[j,0] = softmax(np.append(np.array([[1]]),Z[j]),0)[1] - softmax(np.append(np.array([[1]]),Z[j-1]),0)[1]
    V = np.cumsum(V,axis = 0)
    Y[i,0] = np.nonzero(r[i,0]<V)[0][0]
    

# Infer causes of data
#------------------------------------------------------------------------------

Ep,Cp,F = VL_ordinal.invert(U,Y,K)

# Overlay true values on confidence intervals
#------------------------------------------------------------------------------
Fsi = plt.gcf()
ax  = Fsi.get_axes()
ax[-1].bar(range(aux.length(P)),np.array(aux.vec(P)).T[0],width = 1/4,color = [0.8,0.1,0.1],zorder = 3)
x0,x1 = ax[-1].get_xlim()
y0,y1 = ax[-1].get_ylim()
ax[-1].set_aspect(abs(x1-x0)/abs(y1-y0))

# Bayesian model reduction
#------------------------------------------------------------------------------
pE,pC = VL_ordinal.default_priors(U,Y,K)
M = {'pE':pE,'pC':pC}
D = {'M':M,'Ep':Ep,'Cp':Cp}
BMR,BMA = reduce_gaussian(D,'all')
plt.show(block=True)