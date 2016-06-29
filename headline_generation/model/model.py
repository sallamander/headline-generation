"""A script for fitting a recurrent net to generate headlines for article text."""

import numpy as np
np.random.seed(427)  # For reproducibility

import sys
sys.setrecursionlimit(10000) # To avoid dropout errors
import datetime
from keras.layers import Input
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.layers.core import Dense
from keras.models import Model
from keras.callbacks import EarlyStopping
from headline_generation.model.eval_model import return_xy_subset
from headline_generation.utils.preprocessing import vectorize_texts, format_inputs
from headline_generation.utils.data_io import return_data
from headline_generation.utils.mappings import create_mapping_dicts, \
        map_idxs_to_str, gen_embedding_weights
from headline_generation.utils.keras_callbacks import PredictForEpoch
from headline_generation.model.eval_model import generate_sequence, return_xy_subset

def make_model(embedding_weights, input_length=50):
    """Build an recurrent net based off the input parameters and return it compiled.

    Args: 
    ----
        embedding_weights: 2d np.ndarray
        input_length (optional): int
            Holds how many words each article body will hold

    Return: 
    ------
        model: keras.model.Sequential compiled model
    """

    dict_size = embedding_weights.shape[0] # Num words in corpus
    embedding_dim = embedding_weights.shape[1] # Num dims in vec representation

    bodies = Input(shape=(input_length,), dtype='int32') 
    embeddings = Embedding(input_dim=dict_size, output_dim=embedding_dim,
                           weights=[embedding_weights], dropout=0.5)(bodies)
    layer = GRU(1024, return_sequences=True, dropout_W=0.5, dropout_U=0.5)(embeddings)
    layer = GRU(1024, return_sequences=False, dropout_W=0.5, dropout_U=0.5)(layer)
    layer = Dense(dict_size, activation='softmax')(layer)

    model = Model(input=bodies, output=layer)

    model.compile(loss='categorical_crossentropy', optimizer='adagrad')

    return model

def fit_model(model, X_fit, y_fit, X_train, y_train, X_test, y_test,
              batch_size=32, nb_epoch=10, early_stopping_tol=0, validation_split=0.0, 
              save_filepath=None, on_epoch_end=False, idx_word_dct=None): 
    """Fit the inputted model according to the inputted specifications. 

    (X_fit, y_fit) are used to actually fit the model, and both (X_train, y_train)
    and (X_test, y_test) are used to evaluate the model at the end of each epoch.
    The idea is that we should see better and better predictions with both 
    (X_train, y_train) and (X_test, y_test), but also be able to see how the model 
    is doing on data it's not training on ((X_test, y_test)). 
    
    Args: 
    ----
        model: compiled keras.model.Model object
        X_fit: 2d np.ndarray
        y_fit: 2d np.ndarray
        X_train: 2d np.ndarray
        y_train: 2d np.ndarray
        X_test: 2d np.ndarray
        y_test: 2d np.ndarray
        batch_size (optional): int
        nb_epoch (optional): int 
        early_stopping_tol (optional): int 
            Holds the `patience` to pass into a keras.callbacks.EarlyStopping object
        validation_split (optional): float 
            Holds the `validation_split` to pass into the `fit` method on the model
        save_filepath (optional): str
            Holds where to save the predictions after each epoch
        on_epoch_end (optional): boolean
            Whether to log predictions on each epoch end. 
        idx_word_dct (optional): dict

    Returns: 
    -------
        model: fitted keras.model.Model object
    """
    callbacks = []
    if early_stopping_tol: 
        monitor = 'loss' if not validation_split else 'val_loss'
        early_stopping = EarlyStopping(monitor=monitor, patience=early_stopping_tol)
        callbacks.append(early_stopping) 
    if on_epoch_end: 
        predict_per_epoch = PredictForEpoch(X_train, y_train, X_test, y_test, 
                                            idx_word_dct, save_filepath)
        callbacks.append(predict_per_epoch)
    
    model.fit(X, y, nb_epoch=nb_epoch, callbacks=callbacks,
                   validation_split=validation_split, batch_size=batch_size,   
                   shuffle=False)

    return model

def save_model_losses(model, save_filepath): 
    """Save model losses to the inputted filepath

    Args: 
    ----
        model: fitted keras.model.Model object
        save_filepath: str
    """

    history = model.history
    train_losses = history.history['loss']
    test_losses = history.history['val_loss']

    train_fp = '{}_train.txt'.format(save_filepath)
    test_fp = '{}_test.txt'.format(save_filepath)

    np.savetxt(train_fp, train_losses)
    np.savetxt(test_fp, test_losses)

if __name__ == '__main__': 
    try: 
        embed_dim = sys.argv[1]
    except: 
        raise Exception("Usage: {} embed_dim".format(sys.argv[0]))
        
    # Unfortunately, there are no real time savings from doing the following data   
    # loading and pre-processing ahead of time, which is why it's done here. 
    wrd_embedding = return_data("word_embedding", embed_dim=embed_dim)
    bodies, headlines = return_data("articles")
    bodies, headlines = bodies[:200], headlines[:200]

    word_idx_dct, idx_word_dct, word_vector_dct = \
            create_mapping_dicts(wrd_embedding, filter_corpus=True, bodies=bodies, 
                                 headlines=headlines)
    embedding_weights = gen_embedding_weights(word_idx_dct, word_vector_dct)

    vec_bodies, vec_headlines = vectorize_texts(bodies, headlines, word_idx_dct)
    vocab_size = len(embedding_weights)
    maxlen = 50
    X, y, filtered_bodies, filtered_hlines = format_inputs(vec_bodies,      
                                                              vec_headlines, 
                                                              vocab_size=vocab_size, 
                                                              maxlen=maxlen)

    X_test, y_test, X, y, test_hlines, filtered_hlines = return_xy_subset(X, y,
                                                                filtered_hlines, 
                                                                nobs=50, train=False)
    X_train, y_train, X, y, train_hlines, filtered_hlines = return_xy_subset(X, y,
                                                                  filtered_hlines, 
                                                                  nobs=50, train=True)
    maxlen += 1 # Account for newline characters added in. 
    preds_filepath = 'work/preds/glove_{}'.format(embed_dim)

    model = make_model(embedding_weights, input_length=maxlen)
    model = fit_model(model, X, y, X_train, train_hlines, X_test,
                           test_hlines, batch_size=32, nb_epoch=1,
                           early_stopping_tol=5, save_filepath=preds_filepath, 
                           on_epoch_end=True, idx_word_dct=idx_word_dct,
                           validation_split=0.10)
    
    weights_fname = 'work/weights/glove_{}.h5'.format(embed_dim)
    model.save_weights(weights_fname, overwrite=True)
    losses_fp = 'work/losses/glove_{}'.format(embed_dim)
    save_model_losses(model, losses_fp)
