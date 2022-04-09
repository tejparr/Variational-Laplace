import numpy as np
import auxiliaryfunctions as aux
from VariationalLaplace import VL_Newton
from auxiliaryfunctions import softmax

# Likelihood model
#------------------------------------------------------------------------------
def L(P,M,U,Y):
    b = np.concatenate((np.array([[0]]),P['beta']),axis = 0)
    B = np.concatenate((np.zeros((np.size(b,0),1)),b),axis = 1)
    B[0,0] = 1
    X = U@B
    V  = softmax(X,1)
    v  = np.zeros((np.shape(Y)))
    for i in range(np.size(Y)):
        v[i,0] = V[i,Y.astype(int)[i]]
    LL = (sum(np.log(v + np.finfo(float).eps)))
    return LL

# Default priors
#------------------------------------------------------------------------------
def default_priors(U,Y):
    """
    Default priors for parameters of logistic model

    Parameters
    ----------
    U : numpy.ndarray
        Design matrix.
    Y : numpy.ndarray
        Data matrix.
        
    Returns
    -------
    pE : Dict
        Prior mode.
    pC : numpy.ndarray
        Prior covariance.

    """
    beta = np.zeros((np.size(U,1)-1,np.size(Y,1)))
    P    = {'beta':beta}

    pE = aux.unvec(np.zeros((aux.length(P),1)),P)
    pC = np.eye(aux.length(P))
    return pE, pC

# Model inversion
#------------------------------------------------------------------------------
def invert(U,Y,*args):
    """
    Variational inversion of multinomial logistic model with design matrix U, and data Y
    
     Parameters
    ----------
    U : numpy.ndarray
        Design matrix.
    Y : numpy.ndarray
        Data matrix.

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
         pE,pC = default_priors(U,Y)
    M = {'L':L,'pE':pE,'pC':pC}

    # Variational Laplace
    #--------------------------------------------------------------------------
    Ep,Cp,F = VL_Newton(M,U,Y)
    return Ep,Cp,F