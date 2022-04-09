import numpy as np
import matplotlib.pyplot as plt
import auxiliaryfunctions as aux
import VL_linear
from numpy import exp
from BayesianModelReduction import reduce_gaussian

# Linear model demonstration for VL routine
#==============================================================================

np.random.seed(1) # for reproducibility

# Parameters and design matrix (U)
#------------------------------------------------------------------------------
U    = np.random.randn(32,2)                   # Design matrix
beta = np.array([[4],[2]])
P    = {'beta':beta,'pi':2}                    # Parameters
     
# Generate data
#------------------------------------------------------------------------------
Y = U@P['beta'] + exp(-P['pi']/2)*np.random.randn(32,1)

# Infer causes of data
#------------------------------------------------------------------------------
Ep,Cp,F = VL_linear.invert(U,Y)

# Overlay true values on confidence intervals
#------------------------------------------------------------------------------
Fsi = plt.gcf()
ax  = Fsi.get_axes()
ax[-1].bar(range(aux.length(P)),np.array(aux.vec(P)).T[0],width = 1/4,color = [0.8,0.1,0.1],zorder = 3)

# Bayesian model reduction
#------------------------------------------------------------------------------
pE,pC = VL_linear.default_priors(U,Y)
M = {'pE':pE,'pC':pC}
D = {'M':M,'Ep':Ep,'Cp':Cp}
BMR,BMA = reduce_gaussian(D,'all')