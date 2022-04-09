import auxiliaryfunctions as aux
import scipy.special as sc
import numpy as np
import matplotlib.cm as cm
from numpy.linalg import inv, det
from copy import deepcopy

# Functions that return reduced log evidence and posteriors for different
# distributions. These are taken from Table 1 of Friston, Parr, Zeidman (2019)
# https://arxiv.org/ftp/arxiv/papers/1805/1805.07092.pdf
#==============================================================================

def gaussian(pE,pC,qE,qC,rE,rC):
    
    """
    Bayesian model reduction for Gaussian distributions. This takes the priors 
    and posteriors for a model and computes the reduced posteriors that we 
    would have found had we used the reduced priors. In addition, it returns 
    the difference in log evidence between the original (full) and the reduced 
    model.

    Parameters
    ----------
    pE,pC : numpy.ndarray
        Prior mode and covariance.
    qE,qC : numpy.ndarray
        Posterior parameters.
    rE,rC : numpy.ndarray
        Reduced priors.

    Returns
    -------
    sE,sC : numpy.ndarray
        Reduced posteriors.
    dF : float
        Difference in log evidence (reduced - full).

    """
    sP = inv(qC) + inv(rC) - inv(pC) # Reduced precision
    sC = inv(sP)                     # Reduced covariance
    sE = sC@(inv(qC)@qE + inv(rC)@pE - inv(pC)@pE)
    dF = 0.5*np.log(det(inv(rC)@inv(qC)@sC@pC)) \
        - 0.5*(qE.T@inv(qC)@qE + rE.T@inv(rC)@rE - pE.T@inv(pC)@pE - sE.T@inv(sC)@sE)
    dF = float(dF)
    return sE,sC,dF

def dirichlet(pd,qd,rd):
    """
    Bayesian model reduction for Dirichlet and Beta distributions. This takes 
    the priors and posteriors for a model and computes the reduced posteriors 
    that we would have found had we used the reduced priors. In addition, it 
    returns the difference in log evidence between the original (full) and the 
    reduced model.

    Parameters
    ----------
    pd : numpy.ndarray
        Prior parameters.
    qd : numpy.ndarray
        Posterior parameters.
    rd : numpy.ndarray
        Reduced priors.

    Returns
    -------
    sd : numpy.ndarray
        Reduced posterior.
    dF : float
        Difference in log evidence (reduced - full).

    """
    sd = qd + rd - pd
    dF = np.log(aux.beta(pd)) - np.log(aux.beta(rd)) + np.log(aux.beta(sd)) - np.log(aux.beta(qd))
    return sd,dF

def multinomial(pd,qd,rd):
    """
    Bayesian model reduction for multinomial, binomial, and categorical
    distributions. This takes the priors and posteriors for a model and
    computes the reduced posteriors that we would have found had we used 
    the reduced priors. In addition, it returns the difference in log 
    evidence between the original (full) and the reduced model.

    Parameters
    ----------
    pd : numpy.ndarray
        Prior parameters.
    qd : numpy.ndarray
        Posterior parameters.
    rd : numpy.ndarray
        Reduced priors.

    Returns
    -------
    sd : numpy.ndarray
        Reduced posterior.
    dF : float
        Difference in log evidence (reduced - full).

    """
    sd = aux.softmax(np.log(rd) + np.log(qd) - np.log(pd),0)
    dF = (np.log(rd) + np.log(qd) - np.log(pd) - np.log(sd))[0,0]
    return sd,dF

def gamma(pa,pb,qa,qb,ra,rb):
    """
    
    Bayesian model reduction for gamma distributions. This takes the priors 
    and posteriors for a model and computes the reduced posteriors that we 
    would have found had we used the reduced priors. In addition, it returns 
    the difference in log evidence between the original (full) and the reduced 
    model.

    Parameters
    ----------
    pa,pb : float
        Prior shape and rate parameters.
    qa,qb : float
        Posterior parameters.
    ra,rb : float
        Reduced priors.

    Returns
    -------
    sa,sb : float
        Reduced posterior.
    dF : float
        Difference in log evidence (reduced - full).

    """
    sa = qa + ra - pa
    sb = qb + rb - pb
    dF = qa*np.log(qb) + ra*np.log(rb) - pa*np.log(pb) - sa*np.log(sb) \
        + sc.gammaln(pa) + sc.gammaln(sa) - sc.gammaln(ra) - sc.gammaln(qa)
    return sa,sb,dF

# Model comparisons based upon the Bayesian model reduction routines above
#==============================================================================

def model_average(D):
    """

    Bayesian model averaging.
    
    This takes a list of models and their associated log evidences and returns
    the model average. Here, models are assumed to be normally distributed.
    Unlike the spm_dcm_bma.m routine, this does not use sampling. Instead, it
    takes a linear combination of expectations according to their probability
    to find the new expectation. The new covariance is calculated by taking a 
    linear combination of covariances plus their associated squared means
    and then subtracting the squared expected mean. 
    
    aEp = E[Ep]
    aCp = E[Cp + Ep@Ep.T] - aEp@aEp.T

    Parameters
    ----------
    D : list of dicts
        Models containing covariance matrices (Cp), expectations Ep, and .

    Returns
    -------
    bma : dict
        Bayesian model average.

    """
    
    # Find probabilities for each model
    #--------------------------------------------------------------------------
    n = np.size(D)
    lnp = np.zeros((1,n))
    
    for i in range(n):
        lnp[0,i] = D[i]['F']
    
    p   = aux.softmax(lnp,1)
    
    # Compute average expectations
    #--------------------------------------------------------------------------
    aEp = np.zeros(np.shape(aux.vec(D[0]['Ep'])))
    for i in range(n):
        aEp = aEp + p[0,i]*aux.vec(D[i]['Ep'])
    
    # Compute average covariances
    #--------------------------------------------------------------------------
    aCp = np.zeros(np.shape(D[0]['Cp']))
    for i in range(n):
        aCp = aCp + p[0,i]*(D[i]['Cp'] + aux.vec(D[i]['Ep'])@aux.vec(D[i]['Ep']).T)
        
    aCp = aCp - aEp@aEp.T
    
    # Compile model average structure
    #--------------------------------------------------------------------------
    bma = {'Ep':aux.unvec(aEp,D[0]['Ep']),'Cp':aCp}
    return bma

def reduce_gaussian(oD,ofield):
    """
    This routine is based on spm_dcm_bmr_all.m and implements a search through
    the different combinations of free parameters in specified fields of an 
    inverted (full) model. It compares each of the reduced models with
    combinations of parameters switched off using Bayesian model reduction.

    Parameters
    ----------
    D : Dict
        Parameters.
    field : string or list
        Fields for BMR.

    Returns
    -------
    BMR  : Dict
        Reduced model.
    BMA : Dict
        Bayesian model average.

    """
    field = deepcopy(ofield)
    D     = deepcopy(oD)
    
    # Specify reduced mode (beta) and variance (gamma)
    #--------------------------------------------------------------------------
    try:    rbeta  = D['beta']
    except: rbeta  = 0
    try:    rgamma = D['gamma']
    except: rgamma = np.exp(-16) # In the corresponding spm function, this is
                                 # set to be zero. However, the exp(-16) is
                                 # introduced in the spm_log_evidence.m
                                 # routine (absent above) - so overall this is 
                                 # the same.
    
    # Check fields of parameter dictionary
    #--------------------------------------------------------------------------
    if isinstance(field,str):
        if field.lower() == 'all' and isinstance(D['Ep'],dict):
            field = list(D['Ep'].keys())
        else:
            field = [field]
    
    # Get prior covariances
    #--------------------------------------------------------------------------
    if isinstance(D['M']['pC'],dict):
        D['M']['pC'] = np.diag(np.squeeze(aux.vec(D['M']['pC'])))
    if not aux.length(D['M']['pE']) == np.size(D['M']['pC'],0):
        D['M']['pC'] = np.diag(np.squeeze(aux.vec(D['M']['pC'])))
        
    # Get priors and posteriors
    #--------------------------------------------------------------------------
    qE = D['Ep']
    qC = D['Cp']
    pE = D['M']['pE']
    pC = D['M']['pC']
    
    # Remove nullspace
    #--------------------------------------------------------------------------
    U,S,V = aux.svd(pC,0)
    del S,V
    qE = U.T@aux.vec(qE)
    pE = U.T@aux.vec(pE)
    qC = U.T@qC@U
    pC = U.T@pC@U
    
    # Greedy search
    #==========================================================================
    
    # Accumulated reduction vector (C)
    #--------------------------------------------------------------------------
    q = np.diag(D['M']['pC'])
    if np.sum(q < 1024):
        C = np.array(q > np.mean(q[q < 1024])/1024).astype(int)
    else:
        C = np.array(q > 0).astype(int)
        
    GS = 1
    while GS:
        # Find free coupling parameters
        #----------------------------------------------------------------------
        if isinstance(D['Ep'],dict):
            k = []
            m = 0
            for i in D['Ep']:
                if i in field:
                    if isinstance(D['Ep'][i],(int,float)):
                        k.append(m)
                        m = m + 1
                    else:
                        k = k + list(range(m,m+np.size(D['Ep'][i])))
                        m = m + np.size(D['Ep'][i])
            del m
            k = np.array(k)
        else:
            k = np.r_[0:aux.length(D['Ep'])]
        k = k[np.nonzero(C[k])]
        
        # If too many parameters, find those with the lowest evidence
        #----------------------------------------------------------------------
        nparam = np.size(k)
        nmax   = int(max(nparam/4,8))
        if nparam > nmax:
            # Model search over new prior without ith parameter
            #------------------------------------------------------------------
            Z = np.zeros((1,nparam))
            for i in range(nparam):
                # Identify parameters to retain (r) and to remove (s)
                #--------------------------------------------------------------
                r = deepcopy(C)
                r[k[i]] = 0
                s = 1 - r
                # Create reduced prior covariance matrix
                #--------------------------------------------------------------
                R  = U.T@np.diag(r + s*rgamma)@U
                rC = R@pC@R
                
                # Create reduced prior modes
                #--------------------------------------------------------------
                if isinstance(rbeta,(float,int)):
                    S  = U.T@np.diag(r)@U
                    rE = S@pE + U.T@np.array([s]).T*rbeta
                else:
                    rE = deepcopy(pE)
                Z1,Z2,Z3 = gaussian(pE,pC,qE,qC,rE,rC)
                Z[0,i] = Z3
                del Z1,Z2,Z3
            
            # Find parameters with least evidence
            #------------------------------------------------------------------
            i = np.argsort(-Z)
            k = k[i[0,np.r_[0:nmax]]]
            GS = 1
            
        elif np.size(k) == 0:
            print("There are no free parameters in this model")
            return
        else:
            GS = 0
            
        # Compare models
        #======================================================================
        for j in range(2):
            if j == 0:
                # Compare models with and without nmax first
                #--------------------------------------------------------------
                K = np.tile(np.array([[True],[False]]),(1,np.size(k)))
            else:
                # Compare all combinations
                #--------------------------------------------------------------
                k = k[np.r_[0:min(8,np.size(k))]]
                K = aux.perm_mtx(np.size(k))
                
            # Model search over new prior
            #------------------------------------------------------------------
            nK = np.size(K,0)
            G  = np.zeros((1,nK))
            for i in range(nK):
                # Identify parameters to retain (r) and to remove (s)
                #--------------------------------------------------------------
                r = deepcopy(C)
                r[k[K[i,:]]] = 0
                s = 1 - r
                # Create reduced prior covariance matrix
                #--------------------------------------------------------------
                R  = U.T@np.diag(r + s*rgamma)@U
                rC = R@pC@R
                # Create reduced prior modes
                #--------------------------------------------------------------
                if isinstance(rbeta,(float,int)):
                    S  = U.T@np.diag(r)@U
                    rE = S@pE + U.T@np.array([s]).T*rbeta
                else:
                    rE = deepcopy(pE)
                G1,G2,G3 = gaussian(pE,pC,qE,qC,rE,rC) 
                G[0,i] = G3
                del G1,G2,G3
                
                # If sufficient complexity reduction, omit combinations
                #--------------------------------------------------------------
                if G[0,0] - G[0,-1] > nmax and nparam > nmax:
                    break
                else:
                    nmax = 8
            
        # Posterior probability
        #----------------------------------------------------------------------
        p = aux.softmax(G,1)
        
        # Get selected model and prune redundant parameters
        #----------------------------------------------------------------------
        i = np.argmax(p)
        C[k[K[i,:]]] = 0
        
        # Continue greedy search if parameters were eliminated
        #======================================================================
        nelim = np.sum(K[i,:])
        GS = GS and nelim
        
        # Show results
        #----------------------------------------------------------------------        
        print('%s out of %s free parameters removed' %(nelim,nparam))
        if nmax < 9:
            Fbmr = aux.figure('BMR - all')
            Fbmr.clf()
            ax = Fbmr.subplots(3,2)
            if np.size(G) > 32:
                ax[0,0].plot(range(np.size(G)),G[0],color = 'k')
            else:
                ax[0,0].bar(range(np.size(G)),G[0],color = 'c')                    
            ax[0,0].set(title = 'log posterior',xlabel = 'model',ylabel = 'log probability')
            x0,x1 = ax[0,0].get_xlim()
            y0,y1 = ax[0,0].get_ylim()
            ax[0,0].set_aspect(abs(x1-x0)/abs(y1-y0))
            
            if np.size(G) > 32:
                ax[0,1].plot(range(np.size(p)),p[0],color = 'k')
            else:
                ax[0,1].bar(range(np.size(p)),p[0],color = 'r')
            ax[0,1].set(title = 'model posterior',xlabel = 'model',ylabel = 'probability')
            x0,x1 = ax[0,1].get_xlim()
            y0,y1 = ax[0,1].get_ylim()
            ax[0,1].set_aspect(abs(x1-x0)/abs(y1-y0))
            
    # Inference over families (one family per parameter)
    #--------------------------------------------------------------------------
    Pk = np.zeros((2,np.size(k)))
    for i in range(np.size(k)):
        Pk[0,i] = np.mean(p[0,np.logical_not(K[:,i])])
        Pk[1,i] = np.mean(p[0,K[:,i]])
    Pk = Pk[0,:]/np.sum(Pk,axis = 0)
    Pp = np.array(C,dtype = float)
    Pp[k] = Pk     

    # Bayesian model average
    #--------------------------------------------------------------------------
    Gmax = max(max(G))
    BMA  = [] 
    for i in range(np.size(K,0)):
        # Include in model average if within Occam's window
        #----------------------------------------------------------------------
        if G[0,i] > (Gmax - 8):
            # Reduced model
            #------------------------------------------------------------------
            r            = deepcopy(C)
            r[k[K[i,:]]] = 0
            s            = 1 - r
            R            = np.diag(r + s*rgamma)
            rC           = R@pC@R
            S            = np.diag(r)
            if isinstance(rbeta,(float,int)):
                rE = S@aux.vec(pE) + np.array([s]).T*rbeta
            else:
                rE = deepcopy(pE)
                
            # BMR
            #------------------------------------------------------------------
            
            Ep,Cp,F = gaussian(pE,pC,qE,qC,rE,rC)
            BMA.append({'Ep':aux.unvec(Ep,D['Ep']),'Cp':Cp,'F':F})
            
    # Bayesian model averaging
    #------------------------------------------------------------------
    BMA = model_average(BMA)
    Ep  = BMA['Ep']
    Cp  = BMA['Cp']
    
    # Show full and reduced posterior estimates
    #------------------------------------------------------------------
    if isinstance(D['Ep'],dict):
        q = np.diag(Cp)
        i = np.nonzero(q > np.mean(q[q < 1024])/1024)[0]
    else:
         i = np.r_[0:aux.length(D['Ep'])]
                
    qE = aux.vec(qE)
    Ep = aux.vec(Ep)

    j  = i[np.isin(i,np.r_[0:aux.length(D['Ep'])])]
          
    # BMR summary and plotting
    #==================================================================
         
    # Get names of fields indexed by k
    #------------------------------------------------------------------
    try:
        Pnames = []
        m = 0
        mm = 0
        for n in D['Ep']:
            ne = np.sum(np.isin(np.r_[0:aux.length(D['Ep'][n])]+m,k))
            Pnames.append([list(D['Ep'].keys())[mm]]*ne)
            m  = m + aux.length(D['Ep'][n])
            mm  = mm + 1
    except:
        Pnames = "all parameters"
    BMR = {'name':Pnames,'F':G,'P':p,'K':K,'k':k}
            
    # Plotting bars with credible intervals for full model
    #------------------------------------------------------------------
    ax[1,0].cla()
    c = 1.64*np.sqrt(np.diag(qC))[i]
    ax[1,0].bar(range(np.size(i)),np.array(qE[i]).T[0],color = [0.8, 0.8, 1.0])
    for n in range(np.size(i)):
        ax[1,0].plot([n,n],np.array([-1,1])*c[n] + np.float64(qE[i][n]),color = [1,3/4,3/4],linewidth = 4)
        ax[1,0].set(title = 'MAP (full)')
        x0,x1 = ax[1,0].get_xlim()
        y0,y1 = ax[1,0].get_ylim()
        ax[1,0].set_aspect(abs(x1-x0)/abs(y1-y0))
        ax[1,0].spines['bottom'].set_position('zero')

    # Plotting bars with credible intervals for reduced model
    #------------------------------------------------------------------
    ax[1,1].cla()
    c = 1.64*np.sqrt(np.diag(Cp))[j]
    ax[1,1].bar(range(np.size(i)),np.array(Ep[j]).T[0],color = [0.8, 0.8, 1.0])
    for n in range(np.size(j)):
        ax[1,1].plot([n,n],np.array([-1,1])*c[n] + np.float64(Ep[j][n]),color = [1,3/4,3/4],linewidth = 4)
        ax[1,1].set(title = 'MAP (reduced)')
        x0,x1 = ax[1,1].get_xlim()
        y0,y1 = ax[1,1].get_ylim()
        ax[1,1].set_aspect(abs(x1-x0)/abs(y1-y0))
        ax[1,1].spines['bottom'].set_position('zero')
         
    # Show alternative models as image
    #------------------------------------------------------------------
    ax[2,0].imshow(1 - K.T,cmap = cm.gray)
    ax[2,0].set(title = 'Model space',xlabel = 'Model', ylabel = 'Parameter')
    x0,x1 = ax[2,0].get_xlim()
    y0,y1 = ax[2,0].get_ylim()
    ax[2,0].set_aspect(abs(x1-x0)/abs(y1-y0))
    
    # Posteriors
    #------------------------------------------------------------------
    ax[2,1].bar(range(np.size(Pp[i])),Pp[i])
    ax[2,1].set(title = 'Posterior',xlabel = 'Parameter')
    x0,x1 = ax[2,1].get_xlim()
    y0,y1 = ax[2,1].get_ylim()
    ax[2,1].set_aspect(abs(x1-x0)/abs(y1-y0))
        
    Fbmr.tight_layout()
            
    # Save reduced models
    #------------------------------------------------------------------
    if isinstance(D['Ep'],dict):
        Pp = aux.unvec(Pp,D['Ep'])
        Ep = aux.unvec(Ep,D['Ep'])
    M = D['M']
    M['pE'] = rE
    M['pC'] = rC
    BMR['D'] = {'M':M,'Ep':Ep,'Cp':Cp,'Pp':Pp}

    return BMR,BMA
