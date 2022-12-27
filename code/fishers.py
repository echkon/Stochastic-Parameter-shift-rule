from tqix import *
import numpy as np
from scipy.linalg import expm
from autograd.numpy.linalg import inv, multi_dot

import stoc.bases


def qfim(grad_f,rho0,opers,phases,t,mu,y):
    """get quantum fisher information matrix
    
    Args:
        - grad_f: gradient fuction (such as stoc)
        - rho0: initial state
        - phases: unkown estimated phases
        - t: time
        - mu: parameter mu
        - y: noise
        
    Return: qfim
    """
    #get rho
    if not(isoperx(rho0)):
        rho0 = operx(rho0)
        
    ut = stoc.bases.Uni(opers,phases,t)
    rhot = dotx(ut,rho0,daggx(ut))
    rho = stoc.bases.dephasing(rhot,t,y)

    grho = grad_f(rho0,opers,phases,t,mu,y)
    
    #print("grho",grho)
        
    d = len(grho) # number of paramaters
    H = np.zeros((d,d), dtype = complex)
    invM = _inv_M(rho)
    
    vec_grho = []  
    for i in range(d):
        vec_grho.append(_vectorize(grho[i]))

    for i in range(d):
        for j in range(d): 
            H[i,j] = 2*multi_dot([daggx(vec_grho[i]), invM, vec_grho[j]]) 
    return np.real(H) 


def qfim_bound(grad_f,rho0,opers,phases,t,mu,y):
    """get quantum fisher information matrix
    
    Args:
        - grad_f: gradient fuction (such as stoc)
        - rho0: initial state
        - phases: unkown estimated phases
        - t: time
        - mu: parameter mu
        - y: noise
        
    Return: qfim bound
    """
    
    d = len(phases)
    W = np.identity(d)
    
    bound = qfim(grad_f,rho0,opers,phases,t,mu,y)
    return np.real(np.trace(W @ inv(bound + np.eye(len(bound)) * 10e-10)))


def qfim_pure(grad_f,psi0,opers,phases,t,mu,y):
    """get quantum fisher information matrix
    
    Args:
        - grad_f: gradient fuction (such as stoc)
        - rho0: initial state
        - phases: unkown estimated phases
        - t: time
        - mu: parameter mu
        - y: noise
        
    Return: qfim
    """
    #get psi
    ut = stoc.bases.Uni(opers,phases,t)
    psi = dotx(ut,psi0)
    #rho = stoc.bases.dephasing(rhot,t,y)

    grho = grad_f(psi0,opers,phases,t,mu,y)
    
    #print("grho",grho)
        
    d = len(grho) # number of paramaters
    H = np.zeros((d,d), dtype = complex)
    
    for i in range(d):
        for j in range(d): 
            H[i,j] = 4*(dotx(daggx(grho[i]),grho[j]) - dotx(daggx(grho[i]),psi)*dotx(daggx(psi),grho[j]))
    return np.real(H) 


def qfim_bound_pure(grad_f,psi0,opers,phases,t,mu,y):
    """get quantum fisher information matrix
    
    Args:
        - grad_f: gradient fuction (such as stoc)
        - rho0: initial state
        - phases: unkown estimated phases
        - t: time
        - mu: parameter mu
        - y: noise
        
    Return: qfim bound
    """
    
    d = len(phases)
    W = np.identity(d)
    
    bound = qfim_pure(grad_f,psi0,opers,phases,t,mu,y)
    return np.real(np.trace(W @ inv(bound + np.eye(len(bound)) * 10e-10)))


def _vectorize(rho):
    # return a vectorized of rho
    # rho: a matrices (data)
    vec_rho = np.reshape(rho, (len(rho)**2,1), order='F')
    return vec_rho


def _inv_M(rho, epsilon = 10e-10): 
    """ return inverse matrix M 
        M = rho.conj()*I + I*rho.conj()
    
    Args:
        - quantum state rho (data)

    Returns:
       - inverse matrix M 
    """
    
    d = len(rho)
    M = np.kron(np.conj(rho), np.identity(d)) + np.kron(np.identity(d), rho)
    
    return inv(M + np.eye(len(M)) * epsilon)
