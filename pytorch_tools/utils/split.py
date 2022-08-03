import numpy as np

def split_DataFrame(df, test):
    """ Split DataFrame

    Split the dataframe 'df' based on the test function 'test'. Split 'df'
    based on the result of 'test' (returns boolean).

    Parameters
    ----------
    df : DataFrame
        Input dataframe
    test : function
        function applied to the rows of 'df'; must return boolean
            f: row -> True/False

    Returns
    -------
    pd.DataFrame
        part of 'df' where 'test' returns True
    pd.DataFrame
        part of 'df' split into two non overlapping parts.
    """
    tf = df.apply(test, axis=1).values
    idx_1 = np.where(tf)
    idx_2 = np.where(~tf)

    return df.iloc[idx_1], df.iloc[idx_2]


def split_ndarrays(*arrays, cond):
    """ Split ndarrays

    Split arrays based on the condition 'cond' (boolean array). All arrays in
    'arrays' and 'cond' should be of the same length. Split each array into two
    non-overlapping parts:

        array_left :  [arr[i], cond[i] == True]
        array_right : [arr[i] : cond[i] == False]

    Parameters
    ----------
    *arrays : np.ndarray (variadic)
        N np.arrays with the same length.
    cond : list or np.ndarray
        boolean array with the same length as the arrays.
                        (True -> left, False -> right)
    Returns
    -------
    np.ndarray
        left part (where cond[i] True)
            [array_1_left,  array_2_left, ...,  array_N_left]
    np.ndarray
        right part (where cond[i] False)
            [array_1_right, array_2_right, ..., array_N_right]
    """
    cond = np.array(cond)    # transform to np if necessary
    left = []
    right = []
    for arr in arrays:
        left.append(arr[cond])
        right.append(arr[~cond])
        
    return (*left, *right)
