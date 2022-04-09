from scipy import sparse
from scipy.linalg import expm
from numpy import exp
from numpy.linalg import inv, det
from copy import deepcopy
import numpy as np
import cmath
import matplotlib.pyplot as plt
import scipy.special as sc

# SPECIAL FUNCTIONS
#==============================================================================

def softmax(X,n):
    """
    Returns the softmax (normalised exponential) of a matrix X along dimension n

    Parameters
    ----------
    X : numpy.ndarray
        Matrix/vector.
    n : int
        Dimension.

    Returns
    -------
    Z : numpy.ndarray
        Normalised exponential.

    """
    Z = np.array(np.exp(X))
    V = np.sum(Z,axis = n,keepdims = True)
    V = np.tile(V,tuple(1 + np.array(np.shape(X))-np.array(np.shape(V))))
    return Z/V

def beta(a):
    """
    Multivariate beta function

    Parameters
    ----------
    a : numpy.ndarray
        Beta function arguments.

    Returns
    -------
    B : float or numpy.ndarray
        Beta function evaluated at a.

    """
    B = np.exp(sum(sc.gammaln(a)) - sc.gammaln(sum(a)))
    if np.size(B) == 1:
        B = float(B)
    return B

# INFORMATION THEORETIC
#==============================================================================
def KL_Dir(q,p):
    """
    KL Divergence between two Dirichlet distributions

    Parameters
    ----------
    q : numpy.ndarray
        Dirichlet parameters of distribution Q.
    p : numpy.ndarray
        Dirichlet parameters of distribution P.

    Returns
    -------
    KL : Float
        KL Divergence between Q and P.

    """
    
    # Account for zero elements
    j       = 1 - (q == 0)
    p[q==0] = 1
    q[q==0] = 1
    
    # Beta functions
    betalnq = sum(sc.gammaln(q)) - sc.gammaln(sum(q))
    betalnp = sum(sc.gammaln(p)) - sc.gammaln(sum(p))
                 
    # Calculate expected log
    Elnq    = j*(sc.digamma(q) - sc.digamma(sum(q)))
    
    # KL-Divergence
    KL = betalnp - betalnq - sum((p - q)*Elnq)
    KL = float(sum(KL.ravel()))
    
    return KL

# ARRAY OPERATIONS
#==============================================================================

def length(P):
    """
    Length of vectorised array (based on spm_length.m)

    Parameters
    ----------
    P : Various
        Variable containing numerical data.

    Returns
    -------
    n : int
        Length of vectorised array

    """
    
    if isinstance(P, dict):
        # If dictionary or set, loop through fields
        n = 0
        for i in P:
            n = n + length(P[i])
    elif isinstance(P, np.ndarray):
        # If numerical array, return number of elements
        n = np.size(P) 
    elif isinstance(P,int):
        n = 1
    elif isinstance(P,list):
        # If list, loop through elements
        n = 0
        for i in P:
            n = n + length(P[i])
    return n

def vec(X):
    """
    Vectorise X (based on spm_vec.m)

    Parameters
    ----------
    X : Various
        Variable containing numerical data.

    Returns
    -------
    Y : numpy.ndarray
        Vectorised array.

    """
    if isinstance(X, dict):
        # If dictionary, loop through fields
        Y = np.array([[1]])
        for i in X:
            Y = np.concatenate([Y,vec(X[i])])
        Y = np.delete(Y,0,0)
    elif isinstance(X, np.ndarray):
        # If numerical array, return column vector of elements
        Y = X.reshape((np.size(X),1))
    elif isinstance(X,int):
        Y = np.array([[X]])
    elif isinstance(X,list):
        # If list, loop through elements
        Y = np.array([[1]])
        for i in X:
            Y = np.concatenate([Y,vec(X[i])])
        Y = np.delete(Y,0,0)
    Y = np.array(Y)
    return Y

def perm_mtx(n):
    """
    Based upon spm_perm_mtx.m

    Parameters
    ----------
    n : int
        Permutation index.

    Returns
    -------
    K : numpy.ndarray
        Permutation matrix.

    """
    N = 2**n
    K = sparse.lil_matrix((N,n))
    x = np.array([[1],[0]])
    for i in range(n):
        y = np.ones((int(N/np.size(x)),1))
        K[:,i] = np.kron(x,y)
        x      = np.vstack((x,x))
    K = np.array(K.toarray(),dtype = bool)
    return K        

# LINEAR ALGEBRA
#==============================================================================

def en(ox):
    """
    Euclidean normalisation (based on spm_en.m)

    Parameters
    ----------
    ox : np.ndarray
        vector or matrix of column vectors.

    Returns
    -------
    x : np.ndarray
        Column-wise Euclidean normalised vector.

    """

    x = deepcopy(ox)
    d = np.size(x,1) 
    for i in range(0,d-1):
        if np.size(np.nonzero(x[:,i]>0)) > 0:
            x[:,i] = x[:,i]/np.sqrt(sum(x[:,i]**2))
    return x

def svd(oX,oU):
    """
    Singular value decomposition with threshold U (based on spm_svd.m) 

    Parameters
    ----------
    oX : np.ndarray
        Matrix.
    oU : float
        Threshold.

    Returns
    -------
        Tuple
        Contains left singular vectors, singular values, right singular vectors.

    """
    
    X   = deepcopy(oX) # Ensure original X (oX) not changed by function
    U   = deepcopy(oU)
    eps = np.finfo(float).eps
    # set thresholds
    if U >= 1:
        U = U - 1e-6
    elif U <= 0:
        U = 64*eps
        
    # For sparse matrices
    #--------------------------------------------------------------------------    
    d   = np.shape(X)
    M   = d[0]
    try:
        N   = d[1]
    except:
        N   = 1    
    p,q   = np.nonzero(X>0)
    p     = np.unique(p)
    q     = np.unique(q)
    X     = X[(np.ix_(p,q))]

    # SVD
    #--------------------------------------------------------------------------
    i,j = np.nonzero(X)
    s   = np.array(X[X>0])
    s   = np.squeeze(s)
    d   = np.shape(X)
    m   = d[0]
    try:
        n   = d[1]
    except:
        n   = 1 
    if np.size(np.nonzero((i-j)>0)) > 0:
        try:
            X = X.toarray()
        except:
            X = X
        if m > n:
            v,s,v = np.linalg.svd(X.T@X, full_matrices = False)
            # Note that this SVD function returns the transpose of the right
            # singular vectors, so that u,S,v = svd(A) <=> A = u@diag(S)@v. In 
            # Matlab the corresponding function gives [u,S,v] = svd(A) <=>
            # A = u*S*v'
            S     = np.diag(s)
            j     = np.nonzero(s*np.size(s)/sum(s) > U)
            j     = np.squeeze(j)
            v     = v.T
            v     = v[:,j]
            u     = en(X@v)
            S     = np.sqrt(S[np.ix_(j,j)])
        elif m < n:
            u,s,u = np.linalg.svd(X@X.T, full_matrices = False)
            S     = np.diag(s)
            j     = np.nonzero(s*np.size(s)/sum(s) > U)
            j     = np.squeeze(j)
            u     = u.T
            u     = u[:,j]
            v     = en(X.T@u)
            S     = np.sqrt(S[np.ix_(j,j)])
        else:
            u,s,v = np.linalg.svd(X, full_matrices = False)
            S     = np.diag(s)
            s     = S.diagonal()**2
            j     = np.nonzero(s*np.size(s)/sum(s) > U)
            j     = np.squeeze(j)
            v     = v.T
            v     = v[:,j]
            u     = u[:,j]
            S     = S[np.ix_(j,j)]
    else:
        S = sparse.coo_matrix((s,(np.r_[:n],np.r_[:n])), shape = (m,n))
        S = sparse.csr_matrix(S)
        u = sparse.eye(m,n)
        u = sparse.csr_matrix(u)
        v = sparse.eye(m,n)
        v = sparse.csr_matrix(v)
        j = np.argsort(-s)
        j = np.squeeze(j)
        S = S[np.ix_(j,j)]
        v = v[:,j]
        u = u[:,j]
        s = S.diagonal()**2
        j     = np.nonzero(s*np.size(s)/sum(s) > U)
        j     = np.squeeze(j)
        v     = v[:,j]
        u     = u[:,j]
        S     = S[np.ix_(j,j)]
        
    j = np.size(j)
    U = np.zeros((M,j))
    V = np.zeros((N,j))
    if j:
        try:                # Try non-sparse
            U[p,:] = u
            V[q,:] = v
        except:
            U[p,:] = u.toarray()
            V[q,:] = v.toarray()
        
    return sparse.csr_matrix(U),sparse.csr_matrix(S),sparse.csr_matrix(V)

def unvec(ovX,oX):
    """
    This undoes the VL_vec function above (based on spm_unvec.m)

    Parameters
    ----------
    ovX : np.ndarray
        Vectorised array.
    oX : Various
        Template of unvectorised variable.

    Returns
    -------
    X : Various
        Unvectorised array.

    """
    
    vX = deepcopy(ovX)
    X  = deepcopy(oX)
    vX = vec(vX) # ensure vX is a numerical vector
    if isinstance(X, dict):
        # If dictionary or set, loop through fields
        for i in X:
            c = X[i]
            try:
                n = np.size(X[i])
            except:
                n = length(X[i])  
            if n > 0:
                c = unvec(vX[0:n],c)   
                vX = np.delete(vX,range(0,n),0)
            X[i] = c
    elif isinstance(X, np.ndarray):
        # If numerical array, reshape
        try:
            X = vX.toarray().reshape((np.shape(X)))
        except:
            X = vX.reshape((np.shape(X)))
    elif isinstance(X, int):
        X = vX
    else:
        X = np.array([[]])
    return X

# NUMERICAL DERIVATIVES
#==============================================================================

def dfdx(of,of0,dx):
    """
    Evaluate numerical derivative (based on spm_dfdx)

    Parameters
    ----------
    of : Various
        Function to be differentiated evaluated at some point.
    of0 : Various
        Function to be differentiated evaluated dx away from previous point.
    dx : Float
        Small difference over which finite difference derivative is evaluated.

    Returns
    -------
    dfdx : numpy.ndarray
        Derivative of f with respect to x.

    """
    f  = deepcopy(of)
    f0 = deepcopy(of0)
    if isinstance(f,list):
        dfdx = f
        for i in range(np.size(f)):
            dfdx[i] = dfdx(f[i],f0[i],dx)
    elif isinstance(f,dict):
        dfdx = (vec(f) - vec(f0))/dx
    else:
        dfdx = (f - f0)/dx    
    return dfdx

def dfdx_cat(oJ):
    """
    Concatenate derivatives (based on spm_dfdx_cat)

    Parameters
    ----------
    oJ : Various
        Derivatives.

    Returns
    -------
    J : numpy.ndarray
        Concatenated derviatives.

    """
    J = deepcopy(oJ)
    for i in range(len(J)):
        if np.ndim(J[i]) == 1:
            J[i] = np.array([J[i]])
        elif np.ndim(J[i]) == 0:
            J[i] = np.array([[J[i]]])
    if isinstance(J[0],np.ndarray):
        if np.size(J[0],1) == 1:
            J = np.concatenate(tuple(J),1)
        else:
            J = np.concatenate(tuple(J),0)
    return J

def diff(*oargs):
    """
    Calculate finite difference derivatives (based on spm_diff.m)

    Parameters
    ----------
    *oargs : variable inputs
        Tuple of input arguments including function, function arguments,
        an array/list of arguments to differentiate with respect to, and
        a transform matrix (or matrices).

    Returns
    -------
    Dervatives of the function with respect to function arguments.

    """
        
    args = deepcopy(oargs)
    dx   = exp(-8)
    N    = len(args)

# function
#------------------------------------------------------------------------- ----   
    f = args[0]
    
# Parse input arguments
#------------------------------------------------------------------------------    
    if isinstance(args[N-1],list):
        x = args[1:(N-2)]
        n = np.array(args[N-2])
        V = args[N-1]
    elif isinstance(args[N-1],tuple):
        x = args[1:(N-1)]
        n = np.array(args[N-1])
        V = [np.array([])]*np.size(x)
    
# Check transform matrices (V = dxdy)
#------------------------------------------------------------------------------
    for i in range(len(x)):
        try:
            V[i]
        except:
            V.append(np.array([]))
        if np.size(np.nonzero(n == i+1)) > 0 and not np.size(V[i]):
            V[i] = sparse.csr_matrix(sparse.eye(length(x[i])))
    
# Initialise
#------------------------------------------------------------------------------
    m  = n[-1]
    xm = vec(x[m-1])
    J  = [np.array([])]*np.size(V[m-1],1)
    
# Compute derivatives
#------------------------------------------------------------------------------ 
    if np.size(n) == 1:
        # dfdx
        f0 = f(*x)
        for i in range(len(J)):
            xi      = list(x)
            xi[m-1] = unvec(xm + V[m-1][:,i]*dx,x[m-1])
            J[i]    = dfdx(f(*xi),f0,dx)[0]
        xi = tuple(xi)
        
        # Return numeric array for first order derivatives
        f = vec(f0)      # vectorise f
        
        # If there are no arguments to differentiate w.r.t:
        if np.size(xm) == 0:
            J = sparse.csr_matrix((np.size(f),0))
        # If there are no arguments to be differentiated    
        elif np.size(f) == 0:
            J = sparse.csr_matrix((0,np.size(f)))
        # Differentiation of scalar or vector
        if isinstance(f0,(np.ndarray,int)) and isinstance(J,list):
            J = dfdx_cat(J)
        return J,f0
    else:
        # dfdxx...
        f0 = diff(f,*x,n[0:-1],V)
        p  = True
        for i in range(len(J)):
            xi      = list(x)
            xmi     = xm + V[m-1][:,i]*dx
            xi[m-1] = unvec(xmi,x[m-1])
            xi      = tuple(xi)
            fi      = diff(f,*xi,n[0:-1],V)[0]
            J[i]    = dfdx(fi,f0[0],dx)[0]
            p       = p and isinstance(J[i],(np.ndarray,int))
        if p:
            J = dfdx_cat(J)
        return tuple(np.array([J]))+f0
          
def dx(dfdx,f,t):
    """
    This function, based upon spm_dx.m, computes the change in x for numerical integration

    Parameters
    ----------
    dfdx : np.ndarray 
        Jacobian.
    f : np.ndarray
        flow rate (evaluated at some point).
    t : Float
        Integration step.

    Returns
    -------
    dx : np.ndarray
        Change in x.

    """
    
    n   = np.size(f)
    if isinstance(t,list):
        tau = exp(t[0] - cmath.log(det(dfdx)).real/n)
    else:
        tau = t
    dx  = (expm(dfdx*tau) - np.eye(n))@inv(dfdx)@f
    return dx

# GRAPHICS
#==============================================================================
def figure(name):
    """
    Generate figure with standard size and string name

    Parameters
    ----------
    name : String
        Figure title.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure.

    """
    fig = plt.figure(name,figsize = (7.5,9),facecolor = [1,1,1])
    return fig

