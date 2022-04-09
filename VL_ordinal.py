import numpy as np
import auxiliaryfunctions as aux
from VariationalLaplace import VL_Newton

# Likelihood model
#------------------------------------------------------------------------------
def L(P,M,U,Y):
    K = np.size(P['theta'])
    X = U@P['beta']
    theta = np.array(P['theta'])
    dX      = np.concatenate((np.array(theta[0]),np.exp(theta[1:K,0])),axis = 0)
    dX      = np.cumsum(dX)
    v  = np.zeros((np.shape(Y)))
    for i in range(np.size(Y)):
        Z  = np.array(dX - X[i,0])
        V  = np.zeros((np.size(dX)+1,1))
        for j in range(np.size(Z,0)+1):
            if j == 0:
                V[j,0] = aux.softmax(np.append(np.array([1]),Z[j]),0)[1]
            elif j == np.size(Z,0):
                V[j,0] = 1 - aux.softmax(np.append(np.array([1]),Z[j-1]),0)[1]
            else:
                V[j,0] = aux.softmax(np.append(np.array([[1]]),Z[j]),0)[1] - aux.softmax(np.append(np.array([[1]]),Z[j-1]),0)[1]
        v[i,0] = V[Y.astype(int)[i]]        
    LL = (sum(np.log(v + np.finfo(float).eps)))
    return LL

# Default priors
#------------------------------------------------------------------------------
def default_priors(U,Y,K):
    """
    Default priors for parameters of logistic model

    Parameters
    ----------
    U : numpy.ndarray
        Design matrix.
    Y : numpy.ndarray
        Data matrix.
    K : float
        Number of categories.
        
    Returns
    -------
    pE : Dict
        Prior mode.
    pC : numpy.ndarray
        Prior covariance.

    """
    beta  = np.zeros((np.size(U,1),np.size(Y,1)))
    theta = np.zeros((K,1))
    P    = {'beta':beta,'theta':theta}
    pE = aux.unvec(np.zeros((aux.length(P),1)),P)
    pC = np.eye(aux.length(P))
    return pE, pC

# Model inversion
#------------------------------------------------------------------------------
def invert(U,Y,K,*args):
    """
    

     Parameters
    ----------
    U : numpy.ndarray
        Design matrix.
    Y : numpy.ndarray
        Data matrix.
    K : float
        Number of categories.

    Returns
    -------
    Ep : Dict
        Posterior mode.
    Cp : numpy.ndarray
        Posterior covariance.
    F : Float
        Log evidence.

    """
    # Define model
    #--------------------------------------------------------------------------
    try:
        pE = args[0]
        pC = args[1]
    except:
         pE,pC = default_priors(U,Y,K)
    M = {'L':L,'pE':pE,'pC':pC}

    # Variational Laplace
    #--------------------------------------------------------------------------
    Ep,Cp,F = VL_Newton(M,U,Y)
    return Ep,Cp,F