# LPR - v1
    bug:
        line 177: replace the \texttt{ry} gate with a \texttt{rx} gate
        - circuit.ry(self.offset, qr_target)
        + circuit.rx(self.offset, qr_target)

    complexity:
        # qubits = 6 :  # gates = [21, 21],  depth = [21, 21]
        # qubits = 7 :  # gates = [25, 25],  depth = [25, 25]
        # qubits = 8 :  # gates = [29, 29],  depth = [29, 29]
        # qubits = 9 :  # gates = [33, 33],  depth = [33, 33]
        # qubits = 10:  # gates = [37, 37],  depth = [37, 37]
        # qubits = 11:  # gates = [41, 41],  depth = [41, 41]
        # qubits = 12:  # gates = [45, 45],  depth = [45, 45]

# LPR - v2
    bug:
        line 185: replace the \texttt{cry} gate with a \texttt{crz} gate
        - circuit.cry(self.slope * pow(2, i), q_i, qr_target)
        + circuit.crz(self.slope * pow(2, i), q_i, qr_target)      # defect

    complexity:
        # qubits = 10:  # gates = [37, 37],  depth = [37, 37]

# LPR - v3
    bug:
        line 177: replace the \texttt{ry} gate with a \texttt{rx} gate
        - circuit.ry(self.offset, qr_target)
        + circuit.rx(self.offset, qr_target)      # defect 1

        line 185: switch qubits of the \texttt{cry} gate
        - circuit.cry(self.slope * pow(2, i), q_i, qr_target)
        + circuit.cry(self.slope * pow(2, i), qr_target, q_i)   # defect 2

    complexity:
        # qubits = 10:  # gates = [37, 37],  depth = [28, 28]

# LPR - v4
    bug:
        line 177: replace the \texttt{ry} gate with a \texttt{rx} gate
        - circuit.ry(self.offset, qr_target)
        + circuit.rx(self.offset, qr_target)      # defect 1

        line 185: modify the parameter of the \texttt{cry} gate
        - circuit.cry(self.slope * pow(2, i), q_i, qr_target)
        + circuit.cry(self.slope * pow(2, i-1), q_i, qr_target)   # defect 2

    complexity:
        # qubits = 10:  # gates = [37, 37],  depth = [37, 37]
 
# LPR - v5
    bug:
        line 185: modify the parameter of the \texttt{cry} gate
        - circuit.cry(self.slope * pow(2, i), q_i, qr_target)
        + circuit.cry(self.slope * pow(2, i-1), q_i, qr_target)   # defect

    complexity:
        # qubits = 10:  # gates = [37, 37],  depth = [37, 37]