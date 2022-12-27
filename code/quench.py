import numpy as np
from scipy.linalg import expm
from tqix import *
import stoc.bases


def quench(p, opers, params, t=1, mu=np.pi/20, y=0.0):
    """ compute quench tomography
    Args:
        - p: number of randoms
        - opers: Hamiltonian
        - params: unknown params
        - t: time
        - mu: shift
        - y: noise
    Return: dX2 (the variance of the tomography)    
    """

           
    d = len(params)
    dim = len(opers[-1])

    Xmat = []
    dXmat = [] #an array of derivatives (gradient)


    for k in range(p):
        #create rho_0^(k) and rho^(k)
        rho0 = operx(random(dim)) # dim = 4
        rhox = dotx(stoc.bases.Uni(opers,params,t),rho0,daggx(stoc.bases.Uni(opers,params,t)))
        drhox = stoc.stocf.grad_stoc(rho0,opers,params,t,mu,y) #grad_inf

        for l in range(d): #always d
            Xmat.append(tracex(dotx(rho0,opers[l])) - tracex(dotx(rhox,opers[l])))
            for j in range(d):
                dXmat.append(-tracex(dotx(drhox[j],opers[l])))


    #compute F_{ij} 
    Fmat = np.zeros((d,d), dtype = complex)       
    dXmats = np.transpose(np.reshape(dXmat,(p*d,d)))

    for i in range(d):
        for j in range(d):
            Fmat[i,j] = np.sum(dXmats[i]*dXmats[j]/np.abs(Xmat))

    # compute F^(-1)
    dX2 = tracex(np.linalg.inv(Fmat))
    return dX2
