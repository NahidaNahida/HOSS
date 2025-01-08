from qiskit import QuantumCircuit 
import csv
from scipy.stats import mannwhitneyu
import numpy as np
import time
from tqdm import tqdm
from circuit_execution import *
import math

# The raw program
def uniform_generator(n):
    qc = QuantumCircuit(n)
    for i in range(n):              # defect about the wrong range
        qc.h(i)
    return qc

# The buggy version
def uniform_generator_bug(n):
    qc = QuantumCircuit(n)
    for i in range(1, n):           # defect about the wrong range
        qc.h(i)
    return qc

def MW_U_test(qubit_list, shots, repeats=1000):
    fault_list, time_list = [], []
    for n in tqdm(qubit_list):
        failure = 0
        startTime = time.time()
        for _ in range(repeats):
            qc = uniform_generator_bug(n)
            qc.measure_all()
            dictCounts = circuit_execution(qc, shots=shots)
            # generate expected sample according to the expected probabilities
            exp_probs = [1 / (2 ** n)] * (2 ** n)
            exp_samps = np.random.choice(range(2**n), size=shots, p=exp_probs)    
            # transform a dict into a list
            test_samps = []
            for (key, value) in dictCounts.items():
                test_samps += [key] * value
            _, p_value = mannwhitneyu(exp_samps, test_samps)
            if p_value <= 0.05:
                failure += 1    
        dura_time = time.time() - startTime
        fault_list.append(failure / repeats)
        time_list.append(dura_time / repeats)

    file_name = "shots={}_MWTest.csv".format(shots)
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ['# qubits', 'ave_fault', 'ave_time']
        writer.writerow(header)
        for index, n in enumerate(qubit_list):
            data = [n, fault_list[index], time_list[index]]
            writer.writerow(data)

def HOSS(qubit_list, shots, repeats=1000):
    fault_list, time_list = [], []
    for n in tqdm(qubit_list):
        failure = 0
        startTime = time.time()
        qubit_test_list = np.random.choice(range(n), size=n, replace=False)
        for _ in range(repeats):
            for sel_qubit in qubit_test_list:
                qc = QuantumCircuit(n + 2, 1)
                qc_test = uniform_generator_bug(n)
                qc.append(qc_test, qc.qubits[: n])
                qc.ry(math.pi / 2, n)
                qc.rz(0, n)
                qc.h(-1)
                qc.cswap(-1, sel_qubit, n)
                qc.h(-1)
                qc.measure(qc.qubits[-1],qc.clbits)

                dict_counts = circuit_execution(qc, shots=shots)
                resList = list(dict_counts.keys())
                if len(resList) == 1 and resList[0] == 0:   # this qubit passes                     
                    continue
                else:                                       # this qubit fails
                    failure += 1 
                    break

        dura_time = time.time() - startTime
        fault_list.append(failure / repeats)
        time_list.append(dura_time / repeats)

    file_name = "shots={}_HOSS.csv".format(shots)
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ['# qubits', 'ave_fault', 'ave_time']
        writer.writerow(header)
        for index, n in enumerate(qubit_list):
            data = [n, fault_list[index], time_list[index]]
            writer.writerow(data)

if __name__ == '__main__':
    qubit_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    shots_list_MWU = [1000, 10000, 100000]
    shots_list_HOSS = [20]
    for shots in shots_list_MWU:
        MW_U_test(qubit_list, shots)
    for shots in shots_list_HOSS:
        HOSS(qubit_list, shots)
            