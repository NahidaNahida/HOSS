import os
import pandas as pd
import csv
from comp_test_process_RQs1and2 import *

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)    
OPOs = ['ChiTest', 'KSTest', 'MWTest', 'JSDiv', 'CrsEnt']
WOOs = ['STFQ', 'STSQ', 'HOSS']

def RQ1_running(test_oracles, num_qubits, total_repeats=20, toler_err=0.05):
    version_list = ['v1', 'v2', 'v3', 'v4', 'v5']
    shot_list = range(2, 51, 2)
    for num_qubits in num_qubits:
        for program_version in version_list:
            print(program_version)
            filename = '/IC_' + str(program_version) + '_testSuites_(qubit=' + str(num_qubits) + ',fr=0.5,#t=50).csv'
            df = pd.read_csv(parent_dir + filename, skiprows=0)
            for test_oracle in test_oracles:
                if test_oracle in OPOs:
                    time_records, fault_records = testing_with_statistical_methods(df, 
                                                                                   program_version, 
                                                                                   total_repeats, 
                                                                                   shot_list, 
                                                                                   test_oracle,
                                                                                   toler_err=toler_err)
                elif test_oracle == 'Hoss':
                    time_records, fault_records = testing_with_HOSS(df, 
                                                                    program_version, 
                                                                    total_repeats, 
                                                                    shot_list)
                
                if toler_err == 0.05:
                    file_name = "IC_{}_RQ1_{}.csv".format(program_version, test_oracle)
                else:
                    file_name = "IC_{}_RQ1_{}_err={}.csv".format(program_version, test_oracle, toler_err)

                with open(file_name, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    header = ['shot', 'time_list', 'fault_list', 'ave_time', 'ave_fault', 'std_time', 'std_fault']
                    writer.writerow(header)
                    for index, shots in enumerate(list(shot_list)):
                        time_list, fault_list = time_records[index], fault_records[index]
                        data = [shots, list(time_list), list(fault_list),
                                np.mean(time_list), np.mean(fault_list),
                                np.std(time_list, ddof=1), np.std(fault_list, ddof=1)]
                        writer.writerow(data)
        
def RQ2_running(test_oracles, num_qubits, total_repeats=20):
    version_list = ['v1', 'v2', 'v3', 'v4', 'v5']
    shot_list = [5, 10, 15]          
    for num_qubits in num_qubits:
        for program_version in version_list:
            print(program_version)
            filename = '/IC_' + str(program_version) + '_testSuites_(qubit=' + str(num_qubits) + ',fr=0.5,#t=50).csv'
            df = pd.read_csv(parent_dir + filename, skiprows=0)

            for test_oracle in test_oracles:
                if test_oracle == 'STFQ':
                    time_records, fault_records = testing_with_STFQ(df, 
                                                                    program_version, 
                                                                    total_repeats, 
                                                                    shot_list)
                elif test_oracle == 'STSQ':
                    time_records, fault_records = testing_with_STSQ(df, 
                                                                    program_version, 
                                                                    total_repeats, 
                                                                    shot_list)

                elif test_oracle == 'HOSS':
                    time_records, fault_records = testing_with_HOSS(df, 
                                                                    program_version, 
                                                                    total_repeats, 
                                                                    shot_list)
                
                file_name = "IC_" + str(program_version) + "_RQ2_" + str(test_oracle) + ".csv"
                with open(file_name, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    header = ['shot', 'time_list', 'fault_list', 'ave_time', 'ave_fault', 'std_time', 'std_fault']
                    writer.writerow(header)
                    for index, shots in enumerate(list(shot_list)):
                        time_list, fault_list = time_records[index], fault_records[index]
                        data = [shots, list(time_list), list(fault_list),
                                np.mean(time_list), np.mean(fault_list),
                                np.std(time_list, ddof=1), np.std(fault_list, ddof=1)]
                        writer.writerow(data)

if __name__ == '__main__':
    test_oracles_RQ1 = ['ChiTest', 'KSTest', 'MWTest', 'JSDiv', 'CrsEnt', 'HOSS']
    test_oracles_RQ2 = ['STFQ', 'STSQ', 'HOSS']
    num_qubits = [10]
    RQ1_running(test_oracles_RQ1, num_qubits)
    RQ2_running(test_oracles_RQ2, num_qubits)