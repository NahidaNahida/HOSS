from scipy.spatial.distance import jensenshannon
from scipy.stats import chisquare
from scipy.stats import mannwhitneyu
from scipy.stats import ks_2samp
from scipy.stats import entropy

import numpy as np 
import warnings
from preprossing import samp2prob, count2frequencies

def statistical_method_selection(n, exp_outputs, test_samps, test_oracle, toler_err=0.05):
    '''
        implement the required statistical methods and yield the test result

        Input variables:
            + n             [int]   the number of output qubits
            + exp_outputs   [list]  the expected samples for NHTs and expected probabilities for SDMs
            + test_samps    [list]  the list of measurement results from the tested program
            + test_oracle   [str]   the OPO type using statistical methods
            + toler_err     [float] the threshold for NHTs and SDMs
        
        Output variable:
            + test result   [str]   either 'pass' of 'fail'
    '''
    test_samp_probs = samp2prob(n, test_samps)

    if test_oracle == 'ChiTest':
        test_result = OPO_ChiSquared(exp_outputs, test_samps, n, toler_err)
    elif test_oracle == 'KSTest':
        test_result = OPO_KSTest(exp_outputs, test_samps, toler_err)
    elif test_oracle == 'MWTest':
        test_result = OPO_MWUTest(exp_outputs, test_samps, toler_err)
    elif test_oracle == 'JSDiv':
        test_result = OPO_JSDivergence(exp_outputs, test_samp_probs, toler_err)
    elif test_oracle == 'CrsEnt':
        test_result = OPO_CrossEntropy(exp_outputs, test_samp_probs, toler_err)
    return test_result

def OPO_ChiSquared(exp_samples, test_samples, n, toler_err=0.05):
    exp_frequencies = count2frequencies(exp_samples, 2**n)
    test_frequencies = count2frequencies(test_samples, 2**n)
    
    popList = []
    
    # To avoid 0 / 0, remove the exp_frequencies[i] and test_frequencies[i] if
    # exp_frequencies[i] / test_frequencies[i] approximates 0 / 0.
    for i in range(len(exp_frequencies)):
        if exp_frequencies[i] <= 1e-6 and test_frequencies[i] <= 1e-6:
            popList.append(i)
    if len(popList) > 0:
        for pop_index in sorted(popList, reverse=True):
            exp_frequencies.pop(pop_index)
            test_frequencies.pop(pop_index)
    
    if len(exp_frequencies) == 1 and len(test_frequencies) == 1:
        return 'pass'
    else:        
        with warnings.catch_warnings():                               # ignore the warning of dividing 0
            warnings.simplefilter("ignore", category=RuntimeWarning)
            _, p_value = chisquare(exp_frequencies, test_frequencies) # Pearson chi-squares
            if p_value > toler_err:
                return 'pass'
            else:
                return 'fail'

def OPO_JSDivergence(exp_probs, res_probs, toler_err=0.05):
    exp_probs = list(exp_probs)
    res_probs = list(res_probs)
    dist = jensenshannon(exp_probs, res_probs)
    if dist <= toler_err:
        return 'pass'
    else:
        return 'fail'

def OPO_CrossEntropy(exp_probs, res_probs, toler_err=0.05):
    exp_probs = np.array(exp_probs)
    res_probs = np.array(res_probs)
    dist = entropy(exp_probs, res_probs)
    if dist <= toler_err:
        return 'pass'
    else:
        return 'fail'

def OPO_MWUTest(expSamps, resSamps, toler_err=0.05):
    expSamps = list(expSamps)
    resSamps = list(resSamps)
    _, p_value = mannwhitneyu(expSamps, resSamps)
    if p_value > toler_err:
        return 'pass'
    else:
        return 'fail'
    
def OPO_KSTest(expSamps, resSamps, toler_err=0.05):
    expSamps = list(expSamps)
    resSamps = list(resSamps)
    _, p_value = ks_2samp(expSamps, resSamps, method='asymp')
    if p_value > toler_err:
        return 'pass'
    else:
        return 'fail'