# LAF - v1
    bug:
        line 144: modify an arithmetical operator
        - offset_angles[i] += np.pi * rescaling_factor * (mapped_offset[i] - c) / 2 / (d - c)
        + offset_angles[i] -= np.pi * rescaling_factor * (mapped_offset[i] - c) / 2 / (d - c)
  
    complexity:
        # qubits = 6 :  # gates = [21, 21],     depth = [21, 21]
        # qubits = 7 :  # gates = [25, 25],     depth = [25, 25]
        # qubits = 8 :  # gates = [29, 29],     depth = [29, 29]
        # qubits = 9 :  # gates = [33, 33],     depth = [33, 33]
        # qubits = 10:  # gates = [37, 37],     depth = [37, 37]
        # qubits = 11:  # gates = [41, 41],     depth = [41, 41]
        # qubits = 12:  # gates = [45, 45],     depth = [45, 45]

# LAF - v2
    bug:
        line 141: modify a number
        - offset_angles = np.pi / 4 * (1 - rescaling_factor) * np.ones(len(breakpoints))
        + offset_angles = np.pi / 2 * (1 - rescaling_factor) * np.ones(len(breakpoints)) 
    
        line 144: modify an arithmetical operator
        - offset_angles[i] += np.pi * rescaling_factor * (mapped_offset[i] - c) / 2 / (d - c)
        + offset_angles[i] -= np.pi * rescaling_factor * (mapped_offset[i] - c) / 2 / (d - c)         # defect + -> -
    
    complexity:
        # qubits = 10:  # gates = [37, 37],     depth = [37, 37]

# LAF - v3
    bug:
        line 143: replace an arithmetical operation
        - slope_angles[i] = np.pi * rescaling_factor * mapped_slope[i] / 2 / (d - c)
        + slope_angles[i] = np.pi * rescaling_factor * mapped_slope[i] / 2 / (c - d) # d-c -> c-d
    
    complexity:
        # qubits = 10:  # gates = [37, 37],     depth = [37, 37]

# LAF - v4
    bug:
        line 144: delete an operational component, replace an arithmetical operation
        - offset_angles[i] += np.pi * rescaling_factor * (mapped_offset[i] - c) / 2 / (d - c)
        + offset_angles[i] += np.pi * rescaling_factor * (mapped_offset[i]) / 2 / (c - d)   # delete -c , change d-c -> c-d

    complexity:
        # qubits = 10:  # gates = [37, 37],     depth = [37, 37]

# LAF - v5
    bug:
        line 136: replace an arithmetical operation
        - mapped_slope += [slope[i] * (b - a) / (2**num_state_qubits - 1)]
        + mapped_slope += [slope[i] * (b - a) / (2**(num_state_qubits - 1))]        # def 

    complexity:
        # qubits = 10:  # gates = [37, 37],     depth = [37, 37]