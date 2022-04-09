import numpy as np
from numpy import exp
import auxiliaryfunctions as aux
from VariationalLaplace import VL_Newton


# Likelihood model
#------------------------------------------------------------------------------
def L(P,M,U,Y):
    k  = np.size(Y,0)
    LL = 0
    for i in range(np.size(Y,1)):    
        Pi = exp(P['pi'][0,0])*np.eye(k)
        LL = LL + sum( - np.array([(Y[:,i] - U@P['beta'][:,i])])@Pi@np.array([Y[:,i] - U@P['beta'][:,i]]).T/2) - k*np.log(2*np.pi)/2 + np.log(np.linalg.det(Pi))/2
    return LL

# Priors
#------------------------------------------------------------------------------
def default_priors(U,Y):
    """
    Default priors for parameters of linear model

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
    beta = np.zeros((np.size(U,1),np.size(Y,1)))
    P    = {'beta':beta,'pi':2}

    pE = aux.unvec(np.zeros((aux.length(P),1)),P)
    pC = np.eye(aux.length(P))
    return pE, pC

# Model inversion
#------------------------------------------------------------------------------
def invert(U,Y,*args,**kwargs):
    """
    Variational inversion of linear model with design matrix U, and data Y
    
     Parameters
    ----------
    U : numpy.ndarray
        Design matrix.
    Y : numpy.ndarray
        Data matrix.
    pE: Dict
        Prior mode.
    pC: Dict
        Prior covariance.

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