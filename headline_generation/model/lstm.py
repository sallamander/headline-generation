"""A script for fitting an LSTM to generate headlines for article text."""

import datetime
import numpy as np
from keras.layers import Input
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense
from keras.models import Model
from keras.callbacks import EarlyStopping
from headline_generation.utils.preprocessing import gen_embedding_weights, \
        vectorize_texts, format_inputs
from headline_generation.utils.data_io import return_data
from headline_generation.utils.mappings import create_mapping_dicts, \
        map_idxs_to_str
from headline_generation.model.eval_model import generate_sequence

def make_model(embedding_weights, max_features=300, batch_size=32, input_length=50):
    """Build an LSTM based off the input parameters and return it compiled. 

    Args: 
    ----
        embedding_weights: 2d np.ndarray
        max_features (optional): int
            Holds the max number of features for the embedding layer
        batch_size (optional): int
            Holds how many article bodies to feed through at a time
        input_length (optional): int
            Holds how many words each article body will hold

    Return: 
    ------
        lstm_model: keras.model.Sequential compiled model
    """

    dict_size = embedding_weights.shape[0] # Num words in corpus
    embedding_dim = embedding_weights.shape[1] # Num dims in vec representation

    bodies = Input(shape=(input_length,), dtype='int32') 
    embeddings = Embedding(input_dim=dict_size, output_dim=embedding_dim,
                           weights=[embedding_weights])(bodies)
    layer = LSTM(32, return_sequences=True)(embeddings)
    layer = LSTM(32, return_sequences=True)(embeddings)
    layer = LSTM(32, return_sequences=False)(layer)
    layer = Dense(dict_size, activation='softmax')(layer)

    lstm_model = Model(input=bodies, output=layer)

    lstm_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return lstm_model

def fit_model(lstm_model, nb_epoch=10, early_stopping_tol=0, validation_split=0.0): 
    """Fit the inputted LSTM model according to the inputted specifications. 

    Args: 
    ----
        lstm_model: compiled keras.model.Model object
        nb_epoch (optional): int 
        early_stopping_tol (optional): int 
            Holds the `patience` to pass into a keras.callbacks.EarlyStopping object
        val_split (optional): float 
            Holds the `validation_split` to pass into the `fit` method on the 
            lstm_model

    Returns: 
    -------
        lstm_model: fitted keras.model.Model object
    """
    callbacks = []
    if early_stopping_tol: 
        monitor = 'loss' if not validation_split else 'val_loss'
        early_stopping = EarlyStopping(monitor=monitor, patience=early_stopping_tol)
        callbacks.append(early_stopping) 
    
    # Fit over a range to look at predictions per epoch. 
    for epoch in range(nb_epoch):
        lstm_model.fit(X, y, nb_epoch=1, callbacks=callbacks,
                       validation_split=validation_split)

    return lstm_model

def predict_w_model(lstm_model, X, y, headlines, idx_word_dct, save_filepath=None): 
    """Predict over the list of inputted `headlines` using the inputted model.

    Each headline corresponds to multiple rows in X (one for each of the words in the
    headline) - loop over all these rows and predict on them. If it's the first row, 
    then generate a sequence (e.g. try to replicate the entire headline), and 
    otherwise just predict the single word denoted by the one-hot encoding of the
    corresponding y. 

    Args: 
    ----
        lstm_model: keras.model.Model object
        X: 2d np.ndarray
            Contains 1 body of an article body pair to use for prediction
        y: 2d np.ndarray 
        headlines: list of list of strings  
        idx_word_dct: dct
        save_filepath (optional): str
    """
    
    row_idx = 0
    for hline in headlines: 
        for word_num, word in enumerate(hline): 
            x, y_test = X[row_idx], y[row_idx]
            if not word_num: 
                seq_length = len(hline)
                y_pred = generate_sequence(lstm_model, x)
            else: 
                x = x[np.newaxis]
                y_pred = lstm_model.predict(x)
                y_pred = [np.argmax(y_pred)]
                hline = np.where(y_test == 1)[0] 
            
            predicted_heading = map_idxs_to_str(y_pred, idx_word_dct) 
            actual_heading = map_idxs_to_str(hline, idx_word_dct) 

            with open(save_filepath, 'a+') as f: 
                out_str = '{} \n {} \n'.format(repr(predicted_heading),
                                               repr(actual_heading))
                f.write(out_str)

            row_idx += 1

if __name__ == '__main__': 
    # Unfortunately, there are no real time savings from doing the following data   
    # loading and pre-processing ahead of time, which is why it's done here. 
    wrd_embedding = return_data("word_embedding")
    bodies, headlines = return_data("articles")
    bodies, headlines = bodies[0:3], headlines[0:3]

    word_idx_dct, idx_word_dct, word_vector_dct = \
            create_mapping_dicts(wrd_embedding, filter_corpus=True, bodies=bodies, 
                                 headlines=headlines)
    embedding_weights = gen_embedding_weights(word_idx_dct, word_vector_dct)

    bodies_arr, headlines_arr = vectorize_texts(bodies, headlines, word_idx_dct)
    vocab_size = len(embedding_weights)
    maxlen = 50
    X, y, filtered_bodies, filtered_headlines = format_inputs(bodies_arr,     
                                                              headlines_arr, 
                                                              vocab_size=vocab_size, 
                                                              maxlen=maxlen)

    maxlen += 1 # Account for newline characters added in. 
    lstm_model = make_model(embedding_weights, input_length=maxlen)
    lstm_model = fit_model(lstm_model, nb_epoch=5000, early_stopping_tol=50)

    dt = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
    preds_filepath = 'work/preds/{}.txt'.format(dt)
    predict_w_model(lstm_model, X, y, filtered_headlines, idx_word_dct,
                    preds_filepath)
