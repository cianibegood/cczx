import numpy as np
from typing import List, Union, Callable

def normalized_pauli(xi: Union[np.ndarray, List]) -> np.ndarray:
    """ 
    Description
    --------------------------------------------------------------------------
    Returns the normalized Pauli operator on n qubits associated with the 
    binary vector xi. Note that the ordering is chosen in such a way that
    the tensor product structure is preserved. For instance
    [1, 0, 1, 0] -> X tensor X
    [1, 1, 1,  0] -> Y tensor X
    [1, 0, 0, 1] -> X tensor Z
    """

    n = int(len(xi)/2)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    z = np.array([[1, 0], [0, -1]], dtype=complex)
    op = np.linalg.matrix_power(x, xi[0])@np.linalg.matrix_power(z, xi[1])
    f = xi[0]*xi[1]
    for k in range(1, n):
        f += xi[2*k]*xi[2*k + 1]
        op = np.kron(op, np.linalg.matrix_power(x, xi[2*k])@np.linalg.matrix_power(z, xi[2*k + 1]))
    pauli_op = (1j)**f*op
    return pauli_op/np.sqrt(2**n)

def decimal_to_binary(
    k: int, 
    nbit: int
    ) -> np.ndarray:
    """
    Description
    --------------------------------------------------------------------------
    Returns the integer k as a binary vector with nbit
    """

    y = np.zeros(nbit, dtype=int)
    iterate = True
    x = np.mod(k, 2)
    y[nbit - 1] = int(x)
    if nbit > 1:
        k = (k - x)/2
        l = 1
        while iterate == True:
            l += 1
            x = np.mod(k, 2)
            y[nbit - l] = int(x)
            k = (k - x)/2
            if k <= 0:
                iterate = False
    return y

def binary_to_decimal(k_bin: np.ndarray) -> int:
    """
    Description
    --------------------------------------------------------------------------
    Returns the integer associated with a binary vector
    """

    n = len(k_bin)
    y = k_bin[n-1]
    for l in range(1, n):
        y += 2**l*k_bin[n -l -1]
    return y

def normalized_pauli_by_index(
    i: int,
    d: int
) -> np.ndarray:

    """ 
    Description
    --------------------------------------------------------------------------
    Returns the normalized Pauli operator on n = log_2(d) qubits associated 
    with the integer i 
    """
    
    if np.mod(np.log2(d), 1) != 0.0 or d <= 0:
        raise ValueError("Dimension error: d must be a positive power of 2")
    
    n = int(np.log2(d))
    xi = decimal_to_binary(i, 2*n)
    
    return normalized_pauli(xi)

def return_ptm(map: Callable, num_qubits: int, params: dict):
    d = 2**num_qubits
    ptm = np.zeros([d**2, d**2], dtype=complex)
    for k in range(d**2):
        for l in range(d**2):
            pauli_k = normalized_pauli_by_index(k, d)
            pauli_l = normalized_pauli_by_index(l, d)
            ptm[k, l] = np.trace(pauli_k@map(pauli_l, params))
    return ptm