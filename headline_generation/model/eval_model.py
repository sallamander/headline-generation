"""A module including functions to evaluate the predictions out of an LSTM."""

import numpy as np

def generate_sequence(lstm_model, x): 
    """Generate a sequence of words from predicting once on x and building on that. 

    Args: 
    ----
        lstm_model: keras.model.Model object
        x: 1d np.ndarray
        length: int

    Return:
    ------
        y_pred: 1d np.ndarray 
    """

    input_lst = x.tolist()
    y_pred = []
    
    # Keep going while getting non-zero predictions (zero is "EOF"), or until 
    # hitting a length of 20 (5 longer than longest headline in the dataset). 
    pred = 10
    while pred and len(y_pred) < 20: 
        # This is a little messy, but probably better than doing so later. 
        x_arr = np.array(input_lst)[np.newaxis]
        pred_vector = lstm_model.predict(x_arr)
        pred = np.argmax(pred_vector)

        y_pred.append(pred)
        input_lst = input_lst[1:]
        input_lst.append(pred)
    
    y_pred = np.array(y_pred)

    return y_pred

def return_xy_subset(X, y, headlines_arr, nobs=10, train=True): 
    """Return a train or test subset of the data to predict on during training. 

    If training, keep the observations in the original X and y matrices. If testing,
    then remove the observations in the original X and y matrices. 

    Args: 
    ----
        X: 2d np.ndarray
        y: 2d np.ndarray
        headlines_arr: 1d np.ndarray
        nobs (optional): int
        train (optional): boolean
            Determines whether to keep observations in the original X, y matrices.

    Returns: 
    -------
        X_subset: 2d np.ndarray
        y_subset: 2d np.ndarray
        X: 2d np.ndarray
        y: 2d np.ndarray
        headlines_arr: 1d np.ndarray
    """
    
    X_subset = np.zeros((0, X.shape[1]))
    y_subset = np.zeros((0, y.shape[1]))
    headlines = headlines_arr[:nobs]

    row_idx = 0
    for headline in headlines: 
        len_hline = len(headline)
        X_ob, y_ob = X[row_idx:(row_idx + 1)], y[row_idx:(row_idx + 1)]
        X_subset = np.concatenate([X_subset, X_ob])
        y_subset = np.concatenate([y_subset, y_ob])
        row_idx += len_hline

    if not train: 
        X = X[row_idx:]
        y = y[row_idx:]
        headlines_arr = headlines_arr[nobs:]

    return X_subset, y_subset, X, y, headlines_arr
