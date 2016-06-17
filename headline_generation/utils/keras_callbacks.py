"""A script with Keras callback objects."""

import numpy as np
from keras.callbacks import Callback
from headline_generation.model.eval_model import generate_sequence
from headline_generation.utils.mappings import map_idxs_to_str

class PredictForEpoch(Callback): 
    """A utility class to save predictions on inputted data after each epoch.

    Args: 
    ----
        X_train: 2d np.ndarray
        y_train: 2d np.ndarray
        X_test: 2d np.ndarray
        y_test: 2d np.ndarray
        idx_word_dct: dct
        save_filepath: str
    """

    def __init__(self, X_train, y_train, X_test, y_test, idx_word_dct, 
                 save_filepath): 
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.idx_word_dct = idx_word_dct
        self.save_filepath = save_filepath

    def on_epoch_end(self, epoch, epoch_logs): 
        """Generate a headline for every article body in self.X_train and self.X_test 

        Args: 
        ----
            epoch: int
            epoch_logs: 
        """

        train_preds = []
        test_preds = []

        train_filepath = '{}_train.txt'.format(self.save_filepath)
        test_filepath = '{}_test.txt'.format(self.save_filepath)

        train = (train_filepath, self.X_train, self.y_train)
        test = (test_filepath, self.X_test, self.y_test)

        for fp, X, y in (train, test): 
            with open(fp, 'a+') as f: 
                f.write('Epoch {}: \n'.format(epoch))
                f.write('-' * 50 + '\n')
                for x, y_true in zip(X, y): 
                    y_pred = generate_sequence(self.model, x)
                    y_pred_str = map_idxs_to_str(y_pred, self.idx_word_dct)
                    y_true_str = map_idxs_to_str(y_true, self.idx_word_dct)
                    f.write('Actual: ' + repr(y_true_str) + '\n')
                    f.write('Predicted: ' + repr(y_pred_str))
                    f.write('\n')

                f.write('\n' * 2)


