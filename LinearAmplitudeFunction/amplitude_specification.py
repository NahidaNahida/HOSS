import math, cmath
import numpy as np

def program_specification_angles(qubitIndex, n, number, slop, offset, domain, image):
    '''
        return theta for Ry gate
    '''
    strNumber = bin(number)[2:]
    strNumber = strNumber.zfill(n)
    initial_state = [int(bit) for bit in strNumber]
    revInitialState = initial_state[::-1]
    
    if qubitIndex < n:      # |x>
        qubitVal = revInitialState[qubitIndex]
        if qubitVal == 0:
            theta = 0
        elif qubitVal == 1:
            theta = math.pi
    else:                   # F|0>           
        a, b = domain[0], domain[1] 
        c, d = image[0], image[1] 
        slopm = slop * (b - a) / (2 ** n - 1) / (d - c)
        offsetm = (offset - c) / (d - c)
        theta = np.pi * (slopm * number + offsetm)
    
    beta = 0
    return beta, theta

def program_specification_state(n, number, slop, offset, domain, image):
    '''
        return the state vector
    '''
    inVec = [0] * (2 ** n)
    inVec[number] = 1
    
    a, b = domain[0], domain[1] 
    c, d = image[0], image[1] 
    slopm = slop * (b - a) / (2 ** n - 1) / (d - c)
    offsetm = (offset - c) / (d - c)
    theta = np.pi * (slopm * number + offsetm)
    
    resVec = [math.cos(theta / 2), math.sin(theta / 2)] 
    
    final_state = np.kron(np.array(resVec), np.array(inVec))
    
    return final_state