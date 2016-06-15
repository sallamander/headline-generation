"""A module including functions to evaluate the predictions out of an LSTM."""

import numpy as np

def generate_sequence(lstm_model, x, length): 
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

    for _ in range(length): 
        # This is a little messy, but probably better than doing so later. 
        x_arr = np.array(input_lst)[np.newaxis]
        pred_vector = lstm_model.predict(x_arr)
        pred = np.argmax(pred_vector)

        y_pred.append(pred)
        input_lst = input_lst[1:]
        input_lst.append(pred)

    y_pred = np.array(y_pred)

    return y_pred
