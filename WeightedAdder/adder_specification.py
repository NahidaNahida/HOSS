import math, cmath
import numpy as np

def program_specification_angles(qubitIndex, n, num_qubits, number, weight):
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
    else:                           # [cos(aq+b)/2, sin(aq+b)/2]
        expRes = 0
        s = num_qubits - n
        for i in range(n):
            expRes += revInitialState[i] * weight[i]
        strExpRes = bin(int(expRes))[2:]
        strExpRes = strExpRes.zfill(s)
        outState = [int(bit) for bit in strExpRes]
        revOutState = outState[::-1]
        
        outIndex = qubitIndex - n
        theta = 0 if revOutState[outIndex] == 0 else math.pi

    beta = 0    
    return beta, theta

def program_specification_state(n, num_qubits, number, weight):
    '''
        return the state vector
    '''
    inVec = [0] * (2 ** n)
    inVec[number] = 1
    
    strNumber = bin(number)[2:]
    strNumber = strNumber.zfill(n)
    initial_state = [int(bit) for bit in strNumber]
    revInitialState = initial_state[::-1]
    
    s = num_qubits - n
    outVec = [0] * (2 ** s)
    expRes = 0
    for i in range(n):
        expRes += revInitialState[i] * weight[i]      
    outVec[expRes] = 1
    
    final_state = np.kron(np.array(outVec), np.array(inVec))
    return final_state


def program_specification_value(n, num_qubits, number, weight):
    '''
        return theta for Ry gate
    '''
    # initial state = [1, 0] means (01)b = 1
    s = num_qubits - n
    strNumber = bin(number)[2:]
    strNumber = strNumber.zfill(n)
    initial_state = [int(bit) for bit in strNumber]
    revInitialState = initial_state[::-1]
    expRes = 0
    for i in range(n):
        expRes += revInitialState[i] * weight[i]
    strExpRes = bin(int(expRes))[2:]
    strExpRes = strExpRes.zfill(s)
    strExpRes = strExpRes[::-1]
    value = number
    for ind, bit in enumerate(strExpRes):
        value += int(bit) * (2 ** (n + ind))
    return value

