from tqix import *
import numpy as np
from scipy.linalg import expm


def Hami(opers,thetas):
    """get Hamiltonian
    Args:
        - opers: operators
        - thetas: phases
    Return: Hamiltonian
    """
    d = len(thetas)
    H = 0.0 + 0.0j
    for i in range(d):
        H += thetas[i]*opers[i]
    return H
    
    
def Uni(opers,thetas,t):
    """ get Unitary Operator
    Args:
        - opers: operators
        - thetas: phases
        - t: time
    """
    return expm(-1j*t*Hami(opers,thetas))


def dephasing(rho,t,y):
    """ add dephasing noise to rho
    Args:
        - rho: quantum state
        - t: time
        - y: error rate [0,1]
    Return: rho
    """
    # get number of qubits
    d = len(rho)
    numq = int(np.log2(d))
    
    #Kraus oprators
    k1 = np.array([[np.exp(-y*t),0],[0,1]])
    k2 = np.array([[np.sqrt(1-np.exp(-y*t)**2),0],[0,0]])
    
    #tensor from for Kraus operstor
    ks1 = _array_kraus(numq,k1)
    ks2 = _array_kraus(numq,k2)
    
    #state after noise
    rho1 = dotx(ks1,rho,daggx(ks1))
    rho2 = dotx(ks2,rho,daggx(ks2))

    return np.array(rho1 + rho2)


def _array_kraus(N,k):
    """ get an array of Kraus opreators
    Args:
        - N: number of qubits
        - k: kraus operator
    """
    ks = []
    for i in range(N):
        ks.append(k)
    return tensorx(ks)


def Haar_random(N):
    """create Haar random matrix
    Args:
        - N: number of qubits
    Return:
        - Haar matrix
    """
    
    
