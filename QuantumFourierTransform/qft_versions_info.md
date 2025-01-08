# qft - v1
    bug:
        line 285: replace the \texttt{swap} gate with a \texttt{ch} gate
        - circuit.swap(i, num_qubits - i - 1)
        + circuit.ch(i, num_qubits - i - 1) # def

    complexity:
        # qubits = 6 :  # gates = [97, 102],    depth = [43, 43]
        # qubits = 7 :  # gates = [128, 133],   depth = [51, 51]
        # qubits = 8 :  # gates = [171, 176],   depth = [59, 59]
        # qubits = 9 :  # gates = [212, 217],   depth = [67, 67]
        # qubits = 10:  # gates = [265, 270],   depth = [75, 75]
        # qubits = 11:  # gates = [316, 321],   depth = [83, 83]
        # qubits = 12:  # gates = [379, 384],   depth = [91, 91]

# qft - v2
    bug:
        line 285: replace the \texttt{swap} gate with a \texttt{cx} gate
        - circuit.swap(i, num_qubits - i - 1)
        + circuit.cx(i, num_qubits - i - 1)   # def

    complexity:
        # qubits = 10:  # gates = [235, 240],   depth = [72, 72]

# qft - v3
    bug:
        line 286: add a \texttt{cx} gate
        + circuit.cx(i, num_qubits - i - 1)   # def

    complexity:
        # qubits = 10:  # gates = [250, 255],   depth = [75, 75]

# qft - v4
    bug:
        line 285: replace the \texttt{swap} gate with a \texttt{ch} gate,
                switch qubits of the \texttt{ch} gate
        - circuit.swap(i, num_qubits - i - 1)
        + circuit.ch(num_qubits - i - 1, i) # def

    complexity:
        # qubits = 10:  # gates = [265, 270],   depth = [78, 78]

# qft - v5
    bug:
        line 286: add an \texttt{h} gate
        + circuit.h(i)   # def

    complexity:
        # qubits = 10:  # gates = [250, 255],   depth = [75, 75]

