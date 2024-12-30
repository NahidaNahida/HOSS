from qiskit import QuantumCircuit 
import numpy as np
import csv
from tqdm import tqdm
import random

from diagonal import Diagonal
from diagonal_defect1 import Diagonal_defect1
from diagonal_defect2 import Diagonal_defect2
from diagonal_defect3 import Diagonal_defect3
from diagonal_defect4 import Diagonal_defect4
from diagonal_defect5 import Diagonal_defect5

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)  

from copy import deepcopy
from preprossing import number_list_generation

from circuit_execution import *

def version_selection(program_name, program_version):
    '''
        select the quantum program to be tested

        Input variable:
            + program_name       [str] e.g. "IntegerComparator"
            + program_version    [str] including "v1", "v2", "v3", "v4", "v5"
    '''
    if program_version[0] == "v":
        function_name = program_name + '_defect' + program_version[1:]
    else:
        return f"Invalid program program_."

    if function_name in globals():
        func = globals()[function_name]
        return func
    else:
        return f"Function '{function_name}' not found."
 
def random_input(length, elemList, maxSize=1000):
    numInput = 0
    inputList = []
    while numInput <= min(maxSize, len(elemList) ** length):
        templist = []
        for _ in range(length):
            randElem = random.choice(elemList)
            templist.append(randElem)
        inputList.append(templist)
        uniqueList = list(set(tuple(x) for x in inputList))
        inputList = [list(x) for x in uniqueList]
        numInput = len(inputList)
    return inputList

 
def test_input_generation(n_list, maxTests=None):
    '''
        Generation the test inputs
    
        Input variables:
            + n_list             [list]  the list of qubit numbers
                                         note that n only corresponds to the input domain
            + maxTests           [int]   the maximum size of the test suite

        Output variable:
            + testOrderListRaw   [list]  the randomized order for executing test cases
            + testInput          [list]  the list of test inputs
    '''
    # the maximum number of test cases
    initial_gates = ['|0>', '|1>', '|+>', '|->']              # four possible states
    elemlist = [1, -1, 1j, -1j]                               # diagonal elements

    tempNum = 0
    testInput = []
    for n in n_list:  
        initialStatList = random_input(n, initial_gates)
        diagPairIndexList = number_list_generation(2, len(elemlist))
        diagPairList = []
        for indexList in diagPairIndexList:
            tempList = [elemlist[index] for index in indexList]
            diagPairList.append(tempList)
                
        diagMatrixIndexList = random_input(n, list(range(len(diagPairList))))
        for diagMatrixIndex in diagMatrixIndexList:
            for initialState in initialStatList:
                tempNum += 1
                diagMatrix = [diagPairList[indexList] for indexList in diagMatrixIndex]
                testInput.append([n, initialState, diagMatrix])
                if maxTests != None and tempNum >= maxTests:
                    break

    numTests = len(testInput)
    maxTests = numTests if maxTests == None else maxTests
    testOrderListRaw = np.random.choice(range(numTests), size=min(numTests, maxTests), replace=False)
    return testOrderListRaw, testInput

def program_running(testOrderListRaw, testInput, version, maxTests=None, faultRatio=0.5, eps_0=1e-10):
    '''
        determine the ground truth, select test cases with a required distribution of faults, and output the
        test suites

        Variables:
            + testOrderListRaw   [list]  the randomized order for executing test cases
            + testInput          [list]  the list of test inputs
            + version            [str]   the program version, e.g., 'v1', 'v2', etc.
            + maxTests           [int]   the required size of the generated test suite
            + faultRation        [float] the proportion of failed test cases determined by the ground truth
            + eps_0              [float] the threshold to identify the failure of test cases
    '''

    faultTestSuite = []
    correTestSuite = []
    
    numTests = len(testOrderListRaw)
    maxTests = numTests if maxTests == None else min(maxTests, numTests)
     
    testOrderList = np.random.choice(testOrderListRaw, size=numTests, replace=False)
    for testOrder in tqdm(testOrderList):                    
        initialState, diagPairList = testInput[testOrder][1], testInput[testOrder][2]

        # separate real and imag
        realDiagPair, imagDiagPair = diagPairList.copy(), diagPairList.copy()
        for index, tempPair in enumerate(diagPairList):
            realDiagPair[index] = [tempPair[0].real, tempPair[1].real]
            imagDiagPair[index] = [tempPair[0].imag, tempPair[1].imag]

        diagMatrixArray = [np.array(elem) for elem in diagPairList]
        for index, diag in enumerate(diagMatrixArray[::-1]):
            if index == 0:
                tempMatrix = np.array(diag)
            else:
                tempMatrix = np.kron(diag, tempMatrix)
        diagMatrix = list(tempMatrix)

        n = len(initialState)
        qc = QuantumCircuit(n, n)
        for index, val in enumerate(initialState[::-1]):
            if val == '|1>':
                qc.x(index)
            elif val == '|+>':
                qc.h(index)
            elif val == '|->':
                qc.x(index)
                qc.h(index)
                    
        # running the quantum programs
        qc_res = deepcopy(qc)
        qc_exp = deepcopy(qc)
 
        func = version_selection("Diagonal", version)
        qc_test = func(diagMatrix)
        qc_raw = Diagonal(diagMatrix)
        qc_res.append(qc_test, qc_res.qubits)
        qc_exp.append(qc_raw, qc_exp.qubits)
        
        # derive the state vector to analyze whether the test really fails or not
        test_vec = circuit_execution_fake(qc_res)
        exp_vec = circuit_execution_fake(qc_exp)
        fidelity = (np.abs(np.vdot(np.array(test_vec), np.array(exp_vec)))) ** 2
        truth =  bool(fidelity >= 1 - eps_0)

        test_case = [n, qc_exp.num_qubits, initialState, realDiagPair, imagDiagPair, truth]

        if truth:  
            correTestSuite.append(test_case)
        else:
            faultTestSuite.append(test_case)
            
    num_faults = int(round(maxTests * faultRatio))
    num_faults_raw = len(faultTestSuite)
    num_correct_raw = len(correTestSuite)
    if num_faults_raw >= num_faults:
        faultTestSuiteIndex = np.random.choice(range(num_faults_raw), size=num_faults, replace=False)
    else:
        faultTestSuiteIndex = np.random.choice(range(num_faults_raw), size=num_faults_raw, replace=False)
    correTestSuiteIndex = np.random.choice(range(num_correct_raw), size=maxTests-num_faults,replace=False)
    
    faultTestSuiteOpt = [faultTestSuite[index] for index in faultTestSuiteIndex]
    correTestSuiteOpt = [correTestSuite[index] for index in correTestSuiteIndex]
    print("# faults = {}".format(len(faultTestSuiteOpt)))
    print("# correc = {}".format(len(correTestSuiteOpt)))
    finalTestSuite = correTestSuiteOpt + faultTestSuiteOpt
    count_tests = len(faultTestSuiteOpt)+len(correTestSuiteOpt)

    file_name = "DO_{}_testSuites_(qubit={},fr={},#t={}).csv".format(str(version), 
                                                                     str(qc_exp.num_qubits),
                                                                     str(faultRatio),
                                                                     str(count_tests))

    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ['n', '# qubits','initialState', 'realDiagonalPair', 'imagDiagonalPair', 'if_pass']
        writer.writerow(header)

        for tempTest in finalTestSuite:
            writer.writerow(tempTest)
    print('done!')

if __name__ == '__main__':
    qubitList= [10]
    for temp in qubitList:
        n_list = [temp]
        for version in ['v1', 'v2', 'v3', 'v4', 'v5']:
            if version == 'v3':
                maxSize = 5000
            elif version == 'v5':
                maxSize = 10000
            else:
                maxSize = 1000
            testOrderListRaw, testInput = test_input_generation(n_list, maxSize) 
            program_running(testOrderListRaw, testInput, version, maxTests=50, faultRatio=0.5)