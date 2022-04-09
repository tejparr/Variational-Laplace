from scipy import sparse
from numpy import exp
from numpy.linalg import eig, inv, det
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import auxiliaryfunctions as aux
            
# Variational Laplace
#==============================================================================

def VL_Newton(oM,U,Y):
    """
    Variational Laplace with arbitrary likelihood function using Newton's
    method (based upon spm_nlsi_Newton.m)

    Parameters
    ----------
    oM : TYPE
        Generative model.
    U : TYPE
        Inputs.
    Y : TYPE
        Data.

    Returns
    -------
    Ep : 
        Posterior expectations.
    Cp :
        Posterior covariance.
    F  :
        Free energy (log model evidence).

    """
    
    M = deepcopy(oM)
    
# Options
#------------------------------------------------------------------------------
    try: 
        M['nograph']
    except: 
        M['nograph'] = 0   
    try: 
        M['noprint']
    except:
        M['noprint'] = 0
    try: 
        M['Nmax']
    except:
        M['Nmax']    = 128
        
# Likelihood function
#------------------------------------------------------------------------------
    L = M['L']
    
# Initial parameters
#------------------------------------------------------------------------------
    try:
        M['P']
    except:
        M['P'] = M['pE']   
    
# Prior moments
#------------------------------------------------------------------------------
    pE = M['pE']        # Prior expectations
    try:
        pC = M['pC']    # Prior covariance
    except:             # If not specified, use uninformative priors
        nP = aux.length(pE)
        pC = sparse.eye(nP)*exp(16)
        pC = sparse.csr_matrix(pC)
        
# Unpack covariance
#------------------------------------------------------------------------------
    if isinstance(pC,dict):
        pC = np.diag(np.squeeze(aux.vec(pC)))
        
# Dimensionality reduction of parameter space
#------------------------------------------------------------------------------
    V = aux.svd(pC,0)
    V = V[0]
    
# Second order moments in reduced space
#------------------------------------------------------------------------------
    try:
        pC  = pC.toarray()
    except:
        pC = pC
    pC  = V.toarray().T@pC@V.toarray()
    ipC = inv(pC)
    
# Initialise conditional (posterior) density
#------------------------------------------------------------------------------

    p  = V.T@(aux.vec(M['P']) - aux.vec(M['pE']))
    Ep = aux.unvec(aux.vec(pE) + V.T@p, pE)
    
# Figure (unless disabled)
#------------------------------------------------------------------------------
    if not M['nograph']:
        Fsi = aux.figure('Variational Laplace')
        Fsi.clf()
        plt.ion()
        
# Variational Laplace
#------------------------------------------------------------------------------
    criterion = np.array([0,0,0,0])
    C = {'F':-np.inf}                                    
    v = -4;
    G = []                                     
    for k in range(M['Nmax']):
        
# Log likelihood (f), gradients (dfdp), and curvature (dfdpp)
#------------------------------------------------------------------------------
        dfdpp,dfdp,f = aux.diff(L,Ep,M,U,Y,(1,1),[V])
        dfdp  = dfdp.T
        dfdpp = dfdpp.T

# Ensure prior bounds on curvature (i.e., positive semidefinite precision)
#------------------------------------------------------------------------------
        D,E    = eig(dfdpp)
        dfdpp  = E@np.diag(D*(D<0))@E.T

# Posterior covariance
#------------------------------------------------------------------------------
        Cp =  inv(ipC - dfdpp)

# Free energy
#------------------------------------------------------------------------------

        F = float(f - p.T@ipC@p/2 + np.log(det(ipC@Cp))/2)
        G = G+[F]
        
# Record increases and reference log evidence for reporting
#------------------------------------------------------------------------------
        CdF = F - C['F']
        if not k > 0:             
            F0 = F
            CdF = 0
# If F has increased, update gradients...
#------------------------------------------------------------------------------
        if F > C['F'] or k < 8:
            # Accept current estimates
            C['p']  = p
            C['F']  = F
            C['Cp'] = Cp
            
            # Update gradients and curvatures
            dFdp  = dfdp - ipC@p
            dFdpp = dfdpp - ipC
            
            # Decrease (temporal) regularisation
            v   = min(v + 0.5,4)
            s   = "VB: (+)"
        else:
            # Reset expansion point
            p = C['p']
            
            # Decrease (temporal) regularisation
            v   = min(v - 2,-4)
            s   = "VB: (-)"

# Update parameters
#------------------------------------------------------------------------------        
        dp = aux.dx(dFdpp,dFdp,[v])
        p  = p + dp
        Ep = aux.unvec(aux.vec(pE) + V@p,pE)

# Graphics
#------------------------------------------------------------------------------
        if 'Fsi' in locals():
            if not 'ax' in locals():
                ax = Fsi.subplots(2,2)
                ax[0,0].grid(True)
                ax[0,1].grid(True)
                
            ax[0,0].plot(0,0,'r.',markersize = 32)
            col = [exp(-(k+1)/4),exp(-(k+1)),1]
            try:
                ax[0,0].plot(V[0,:]@p,V[1,:]@p,'.',color = col,markersize = 32)
                ax[0,0].set(xlabel = '1st parameter')
                ax[0,0].set(ylabel = '2nd parameter')
            except:
                ax[0,0].plot(k,V[0,:]@p,'.',color = col)
                ax[0,0].set(xlabel = 'Iteration')
                ax[0,0].set(ylabel = '1st parameter')
            ax[0,0].set(title = 'Trajectory')
            x0,x1 = ax[0,0].get_xlim()
            y0,y1 = ax[0,0].get_ylim()
            ax[0,0].set_aspect(abs(x1-x0)/abs(y1-y0))
            
            ax[0,1].cla()
            ax[0,1].grid(True)
            ax[0,1].bar(range(np.size(G)),np.array(G)-F0,color = 'c')
            ax[0,1].set(xlabel = 'Iteration')
            ax[0,1].set(ylabel = 'log evidence')
            ax[0,1].set(title = 'Free energy')
            x0,x1 = ax[0,1].get_xlim()
            y0,y1 = ax[0,1].get_ylim()
            ax[0,1].set_aspect(abs(x1-x0)/abs(y1-y0))
            
            ax[1,0].cla()
            ax[1,0].grid(True)
            ax[1,0].bar(range(np.size(V,0)),np.array((aux.vec(pE) + V@p).T)[0])
            ax[1,0].set(xlabel = 'Parameter')
            ax[1,0].set(title = 'Posterior expectation')
            x0,x1 = ax[1,0].get_xlim()
            y0,y1 = ax[1,0].get_ylim()
            ax[1,0].set_aspect(abs(x1-x0)/abs(y1-y0))
            ax[1,0].spines['bottom'].set_position('zero')
            
            N = np.size(p,1)
            n = np.size(p,0)
            ax[1,1].cla()
            ax[1,1].grid(True)
            if N == 1:
                if n > 1:
                    c = 1.64*np.sqrt(np.diag(Cp))
                    ax[1,1].bar(range(n),np.array(p).T[0],color = [0.8, 0.8, 1.0])
                    for j in range(n):
                        ax[1,1].plot([j,j],np.array([-1,1])*c[j] + np.float64(p[j]),color = [1,3/4,3/4],linewidth = 4)
                        ax[1,1].set(xlabel = 'Parameter (eigenmodes)')
                        ax[1,1].set(title = 'Posterior deviations')
                        x0,x1 = ax[1,1].get_xlim()
                        y0,y1 = ax[1,1].get_ylim()
                        ax[1,1].set_aspect(abs(x1-x0)/abs(y1-y0))
                        ax[1,1].spines['bottom'].set_position('zero')
            Fsi.tight_layout()
            plt.pause(exp(-32))
            plt.draw()
            
# Convergence
#------------------------------------------------------------------------------
            dF = dFdp.T@dp
            if not M['noprint']:
                print('%s: %d F: %d dF predicted: %d actual: %d' % (s,k,C['F'] - F0,dF,CdF))
            criterion = np.concatenate((np.array([dF[0,0]<1e-1]),criterion[0:-1]))
            if sum(criterion) == np.size(criterion):
                if not M['noprint']:
                     print('Convergence')
                break   
    return aux.unvec(aux.vec(pE) + V@C['p'],pE), V@C['Cp']@V.T, C['F']