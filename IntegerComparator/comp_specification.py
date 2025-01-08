import math
import numpy as np

def program_specification_angles(qubitIndex, n, number, L, sign):
    '''
        return theta for Ry gate
    '''
    strNumber = bin(number)[2:]
    strNumber = strNumber.zfill(n)
    initial_state = [int(bit) for bit in strNumber]
    revInitialState = initial_state[::-1]
    
    if qubitIndex < n:      # |x>
        qubitVal = revInitialState[qubitIndex]
        theta = 0 if qubitVal == 0 else math.pi
    else:                         # |q>n>
        if sign == True:
            zeroVal = int(number >= L)
        else:
            zeroVal = int(number < L)
        if qubitIndex == n:
            theta = 0 if zeroVal == 0 else math.pi
        else:
            theta = 0
    beta = 0
    return beta, theta


def program_specification_value(n, number, L, sign):
    '''
        return value
    '''
    if sign == True:
        high = int(number >= L)
    else:
        high = int(number < L)
    value = number + high * (2 ** n)
    
    return value