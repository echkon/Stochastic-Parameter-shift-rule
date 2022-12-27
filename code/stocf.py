import numpy as np
from scipy.linalg import expm
from tqix import *
import stoc.bases


def grad_stoc(psi0,opers,phases,t,mu,y):
    """ get gradient of rho by stoc param
    
    Args:
        - psi0: initial pure state
        - phases: unkown estimated phases
        - t: upper bound time in the intergral
        - mu: parameter mu
        - y: noise
        
    Returns: gradient of rho
    """

    num_s = 1000 # number of ensemble s
    drhos = [0.0, 0.0, 0.0]
    drhos_norm = []
    
    for _ in range(num_s):
        
        s = np.random.uniform(0, t)
        drho = []

        us = stoc.bases.Uni(opers,phases,s)
        uts = stoc.bases.Uni(opers,phases,t-s)
        ut = stoc.bases.Uni(opers,phases,t)

        for j in range(len(phases)):
            
            phasej = np.zeros(len(phases))
            phasej[j] = 1.0 #no multiply phases here

            ujp = stoc.bases.Uni(opers,phasej,t*mu)
            ujm = stoc.bases.Uni(opers,phasej,-t*mu)

            uplus = dotx(us,ujp,uts)
            uminus = dotx(us,ujm,uts)

            #get para shift
            psip = dotx(uplus,psi0)
            psim = dotx(uminus,psi0)
            
            #make rho
            rhop = dotx(psip,daggx(psi0),daggx(ut)) + dotx(ut,psi0,daggx(psip))
            rhom = dotx(psim,daggx(psi0),daggx(ut)) + dotx(ut,psi0,daggx(psim))

            #derivative
            drho.append(rhop - rhom)
            #drho.append(psip-psim)

        #sum w.r.t random s (t is fit)
        k = 1/(np.sin(2*t*mu))
        for j in range(len(phases)):
            drhos[j] += k*drho[j]

    #normailized
    for j in range(len(phases)):
        #add noises
        drhos[j] = stoc.bases.dephasing(drhos[j],t,y)
        drhos_norm.append(t*drhos[j]/num_s)

    return drhos_norm


def grad_inf(rho0,opers,phases,t,mu,y):
    """ get gradient of rho by finite approximation
    Args:
        - rho0: initial state
        - phases: unkown estimated phases
        - t: time
        - mu: parameter mu
        - y: noise
    Returns: gradient of rho
    """
    #unif = stoc.bases.Uni(opers,phases,t)
    dr = []
    s = 0.00001

    def rho_f(phases):
        ut = stoc.bases.Uni(opers,phases,t)
        rhot = dotx(ut,rho0,daggx(ut))
        rho = stoc.bases.dephasing(rhot,t,y)
        return rho
    
    for i in range(0, len(phases)):
        phases1, phases2 = phases.copy(),phases.copy()
        phases1[i] += s
        phases2[i] -= s
        
        delr = rho_f(phases1) - rho_f(phases2)
        dr.append(delr/(2*s))
        
    return dr
