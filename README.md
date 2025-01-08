# HOSS

## Description

This repository incorporates the artifact involved in the manuscript '*A Hybrid Test Oracle for Quantum Programs with Separable Output States*'. 

More details will be updated if the manuscript is possibly accepted for publication. 

## Environment

First create a conda environment, such as named hoss.

```
conda create -n hoss python=3.11.0
```

Then, activate the environment and install the packages in `requirements.txt`.

```
conda activate hoss
pip install -r requirements.txt
```

Here are the requirements:

```
numpy==2.2.0
pandas==2.2.3
qiskit==0.46.2
qiskit_aer==0.13.3
qiskit_terra==0.46.2
scipy==1.14.1
tqdm==4.66.1
```

## Data

### Test Suites

All the involved test suites are zipped into `test_suites.zip`. In this zip, the test suites for each quantum program are presented in a sperate file entitled with the corresponding program name. More specifically, `RQs1and2` and `RQ3` respectively refer to test suites used for RQs1~2 and RQ3.

The concrete data is stored in `aaa_vbbb_testSuites_(qubit=ccc,fr=50,#=50).csv`, where

+ `aaa` $\in$ {`LPR`, `LAF`, `DO`, `QFT`, `IC`, `WA`};
+ `bbb` $\in$ {`1`, `2`, `3`, `4`, `5`} for RQs1~2 and `bbb` $\in$ {`1`} for RQ3;
+ `ccc` $\in$ {`10`} for any program in RQs1~2. In the case of RQ3, `ccc` $\in$ {`6`, `7`, `8`, `9`, `10`, `11`, `12`} for programs excluding `IC` and `ccc` $\in$ {`6`, `8`, `10`, `12`} for `IC`.

Moreover, the rightmost column `if_pass` indicates the ground truth and the other columns refer to the test inputs.

### Experimental Results

As is stated in the manuscript, only partial representative results and preprocessed data are exhibited, due to limited space. Instead, this repository provides full results and raw data with the zip named `experimental_data.zip`. There are descriptions for the involved files, where all the files named `figures` include the experimental plots in a PDF form and the files `raw_data` offer tables of detailed data.

+ `case_study`: This file contain the experimental results of testing the artificial program, (i.e., data related to both **Section IV** and **Section VI. A**). The tables are named `shots=aaa_bbb.csv`, where `aaa` is the employed shots and `bbb` $\in$ \{`HOSS`, `MWTest`} is the test oracle. For each table, the qubit number (`# qubits`), average detected fault (`ave_fault`) and average run time (`ave_time`) are included.
+ `RQ1`: All the experimental results of RQ1 are in this file. The file `variable_shots` of `figures` complements the plots about variation of shots for all the 5 versions of each program. In `raw_data`, the data (i.e., shots, run time of $r$ repeats, detected faults of $r$ repeats, the mean values and standard derivation for time and faults) corresponding to each plot is presented. The tables of data are named in a form of `aaa_vbbb_RQ1_ccc`, where `aaa` $\in$ {`LPR`, `LAF`, `DO`, `QFT`, `IC`, `WA`}, `bbb` $\in$ {`1`, `2`, `3`, `4`, `5`} and `ccc` $\in$ {`ChiTest`, `Crs Ent`, `HOSS`, `JSDiv`, `KSTest`, `MWTest`}. In addition, `RQ1_processed_data.xlsx` is the data corresponding to **TABLE II** in the manuscript.
+ `RQ2`: This file refers to the data for RQ2. In `figures`, the boxplots for each program with 5, 10, 15 shots are included, which serve as supplementary materials of **Fig. 10**. `RQ2_mean_values.xlsx` shows the data in **TABLE III**, derived from the tables in `raw_data`. `RQ2_fault_statistics.xlsx` demonstrates the concrete statistics for the comparison between 2 WOOs, where $p$-values (p-value), effect sizes (es) and the magnitudes of effect sizes (mag) are included. `RQ2_counts.xlsx` is the processed from `RQ2_fault_statistics.xlsx` and corresponds to data in **TABLE IV**.
+ `RQ3`: It contains the data for RQ3. The 6 subplots of **Fig. 11** are in `figures`. `raw_data` presents the detailed data corresponding to the subplots.

## Codes

### Object Programs

The files `LinearPauliRotations`, `LinearAmplitudeFunction`, `Diagonal`, `QuantumFourierTransform`, `IntegerComparator` and `WeightedAdder` correspond to 6 employed quantum programs. In each file, 

+ `aaa.py`: The bug-free version downloaded from `qiskit.circuit.library` (https://github.com/Qiskit/qiskit/tree/stable/1.2/qiskit/circuit/library), where `aaa` refers to the name of the object program (i.e., `pauli`, `amplitude`, `diagonal`, `qft`, `comp` and `adder`).

+ `aaa_circuits_info.py`:

  This file is used to collect basic information (# Qubits, # Gates and Depths) of the quantum circuit corresponding to each program version. The relevant results are displayed in **Table I** of the manuscript.

+ `aaa_specification.py`:

  This file provides the formula-based program specification provided by Qiskit. `program_specification_angles` offers the two angles $\beta$ and $\theta$ for the swap test mode of HOSS. `program_specification_state` directly yields the expected output state $\ket{\varphi^{\text{ps}}_x}$ of the partition $Q_x$.

+ `aaa_versions_info.md`:

  This file details the manually implanted bugs mentioned in **Table I** of the manuscript.

+ `aaa_defectbbb.py`:

  The buggy version mutating from the raw one. `bbb` indicates the name of the buggy version. For example, `adder_defect2.py` is the `v2` of the object `WA`.

+ `aaa_test_suites_generation.py`:

  The source codes to generate test inputs and ground truths of each test suite.

+ `aaa_test_process_ccc.py`:

  The implementation of test processes using required test oracles. `ccc` can be either `RQs1and2` or `RQ3`, indicating the experiment of each RQ.

+ `aaa_running_ccc.py`:

  The codes to run the test process, where the some arguments can be configured, e.g., test oracles, qubit number, shots and thresholds (only for SDMs in RQ1).

### Implementation

​	At first, use `cd` to locate at the path of this repository.

+ **Test suite generation:** Implement `aaa_test_process_ccc.py` to generate test suites and then save the data table. Taking an example of generating test suites of $\texttt{DO}$, run the following code

  ```
  python Diagonal/diagonal_test_suite_generation.py
  ```

  In addition, some parameters of the main routine can be manually changed according to the specific requirements, such as the qubit numbers, versions, maximum sizes of test suites and ratio of failure.

+ **Case study execution**: The case study refers to testing the artificial quantum program  for producing a uniform distribution. Run the following code to execute this experiment and produce corresponding data tables,

  ```
  python artificial_program.py
  ```

+ **Test process implementation**: Execute `aaa_running_ccc.py` to run `aaa_test_process_ccc.py` and then save the data table. Before running, it is necessary to copy the `.csv` files of test suites to the path where `aaa_running_ccc.py` locates. The required files are listed as follows,

  + **RQ1 and RQ2**: `aaa_vbbb_testSuites_(qubit=10,fr=50,#=50).csv`, where for a tested program `aaa`, `bbb` should traverse from `1` to `5`.
  + **RQ3**: `aaa_v1_testSuites_(qubit=ccc,fr=50,#=50).csv`, where for a tested program `aaa`, `ccc` should traverse each integer from 6 to 12 for programs except `IC` while only even number from 6 to 12 for `IC`.

  Then, taking an example of performing the RQ2 experiment of `LPR`, run the following code

  ```
  python LinearPauliRotations/pauli_runinng_RQs1and2.py
  ```

  Also, some parameters of the main routine can be manually changed, such as the qubit numbers, test oracles and shots.

### Data Processing

A notebook `data_processing.ipynb` is given to reproduce all the figures and tables in **Section IV**, **Section VI. A** and **Section VI. F**.

### Functions

+ `circuit_execution.py`

  This file provides two functions to yield test outputs from the given quantum circuit. `circuit_execution` is used for the actual test process with the backend `'qasm_simulator'`, while `circuit_execution_fake` is not allowed for the generation of ground truths via the backend `statevector_simulator`.

+ `preprocessing.py`

  Several simple functions for conversion among output forms are included.

+ `statistical_tests.py`

  This is a collection of the involved 5 output probability oracles using statistical methods. It returns the test result ('pass' or 'fail') based on the given two groups of samples.

