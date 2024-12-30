import numpy as np
from collections import Counter

def count2probs(counts, n):
    '''
        Transform the counts of measurement results into the probability distribution

        Input variables:
            + counts    [dict]  the dictionary of values along with their amounts
            + n         [int]   the qubit number
        
        Output variable:
            + probs     [list]  the probability distribution
    '''
    output_dic = counts.int_outcomes()
    probs = []
    for i in range(2**n):
        if i not in output_dic:
            probs.append(0)
        else:
            probs.append(output_dic[i]/counts.shots())
    probs = np.asarray(probs) / np.sum(probs)
    return probs

def count2frequencies(data, A):
    '''
        Transform the counts of measurement results into the frequencies
    '''
    count = Counter(data)
    frequencies = [count[i] for i in range(A)]
    return frequencies
    
def samp2prob(n, samp):
    '''
        Transform samples into corresponding probability distributions 
    '''
    shots = len(samp)
    freq = dict(Counter(samp))
    samp_probs = [0] * (2 ** n)
    keyList = list(freq.keys())
    for key in keyList:
        val = freq[key]
        samp_probs[key] = val
    samp_probs = np.array(samp_probs) / shots    
    return samp_probs
    
def number_list_generation(n, m):
    '''
        Generation all the number with n digits in base m
    
        Input variables:
            + n             [int]
            + m             [int]

        Output variable:
            + numbers       [list]  the list of numbers
    '''
    if n <= 0:
        return [[]]                     # return the empty list
    if n == 1:
        return [[i] for i in range(m)]  # generate numbers with n = 1

    # recursive procedure
    smaller_numbers = number_list_generation(n - 1, m)
    numbers = []
    for digit in range(m):
        for smaller_number in smaller_numbers:
            numbers.append([digit] + smaller_number)

    return numbers