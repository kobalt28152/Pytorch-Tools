import numpy as np

def cosineAnnealing_getEpochs(T_0, T_mult, cycles=1):
    """ Compute the epochs properties for cosine annealing

    Compute when a warm restart occurs in cosine annealing and the total
    number of epochs given T_0 and T_mult. Recall that:

        T_{i+1} = T_mult * T_i 

    Parameters
    ----------
    T_0 : int
        Number of iterations for the first restart
    T_mult : int
        Factor increasing T_i after each restart
    cycles : int
        number of full cycles to complete

    Returns
    -------
    np.ndarray
        array containing the epochs where a warm restart occurs
    int
        total number of epochs
    """
    ret = np.zeros(cycles, dtype=int)
    t = T_0
    for i in range(cycles):
        ret[i] = t
        t *= T_mult
    return np.cumsum(ret), np.sum(ret)
