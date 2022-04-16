import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import auxiliaryfunctions as aux
import VL_logistic
import VL_linear
from auxiliaryfunctions import softmax

# Demo comparing supervised learning from labels to data and vice versa
#==============================================================================
# This script illustrates logistic regression and a linear model with 
# categorical inputs, using the same simulated data. 

np.random.seed(1)  # for reproducibility
N = 64             # number of datapoints

# Parameters and design matrix (U) for logistic model
#------------------------------------------------------------------------------
U    = np.concatenate((np.ones((N,1)),5*np.random.randn(N,2)),axis = 1)
beta = np.array([[1],[-1]])
P    = {'beta':beta}  # Parameters
     
# Generate (categorical) data
#------------------------------------------------------------------------------
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

# Invert - treating categorical variables as independent variables
#------------------------------------------------------------------------------
Ep1,Cp1,F1 = VL_logistic.invert(U,Y)

# Invert - treating continuous variables as independent variables
#------------------------------------------------------------------------------
W = np.zeros((np.size(U,0),np.size(U,1)-1))
for i in range(np.size(W,0)):
    W[i,int(Y[i,0])] = 1
    
Ep2,Cp2,F2 = VL_linear.invert(W,U[:][:,np.r_[1,2]])

# Plotting
#------------------------------------------------------------------------------

Fcl = aux.figure('Classification')
Fcl.clf()
ax = Fcl.subplots(2,2)

# Plot colour-coded data-points

col = [[0.8,0.5,0.5],[0.5,0.5,0.8]]

for i in range(np.size(P['beta'],1)+1):
    a = np.nonzero(Y == i)
    ax[0,0].plot(U[a[0],1],U[a[0],2],'.',color = col[i],markersize = 16)
    ax[0,0].grid(True)
    ax[0,1].plot(U[a[0],1],U[a[0],2],'.',color = col[i],markersize = 16)
    ax[0,1].grid(True)
    ax[1,0].plot(U[a[0],1],U[a[0],2],'.',color = col[i],markersize = 16)
    ax[1,0].grid(True)
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

# Plot probabilistic classification (colour given location)

# See section 4.5.2 of Bishop for details of use of posterior covariance in
# computing predictive uncertainty.
d = 0.25
x = np.arange(x0,x1,d)
y = np.arange(y0,y1,d)

x, y = np.meshgrid(x, y)

sigma  = Cp1[0,0]*x**2 + 2*Cp1[1,0]*x*y + Cp1[1,1]*y**2
z      = Ep1['beta'][0,0]*x + Ep1['beta'][1,0]*y
Z      = np.exp(z/np.sqrt(1 + np.pi*sigma/8))
Z      = np.exp(1)/(Z + np.exp(1))
ax[1,0].imshow(Z, extent=[x0, x1, y0, y1],origin = 'lower',cmap = cm.gray, aspect = abs(x1-x0)/abs(y1-y0))
ax[1,0].set_aspect(abs(x1-x0)/abs(y1-y0))
ax[1,0].set(title = 'Probability map P(s|x)')

# Plot probability map of location given colour

z = np.exp( - np.exp(Ep2['pi'])*((x - Ep2['beta'][0,0])**2 
           + (y - Ep2['beta'][1,0])**2))
ax[0,1].imshow(-z, extent=[x0, x1, y0, y1],origin = 'lower',cmap = cm.gray, aspect = abs(x1-x0)/abs(y1-y0))
ax[0,1].set_aspect(abs(x1-x0)/abs(y1-y0))
ax[0,1].set(title = 'Probability map P(x|s = red)')

z = np.exp( - np.exp(Ep2['pi'])*((x - Ep2['beta'][0,1])**2 
           + (y - Ep2['beta'][1,1])**2))
ax[1,1].imshow(-z, extent=[x0, x1, y0, y1],origin = 'lower',cmap = cm.gray, aspect = abs(x1-x0)/abs(y1-y0))
ax[1,1].set_aspect(abs(x1-x0)/abs(y1-y0))
ax[1,1].set(title = 'Probability map P(x|s = blue)')
plt.show(block=True)


