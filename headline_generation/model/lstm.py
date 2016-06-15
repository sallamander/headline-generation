"""A script for fitting an LSTM to generate headlines for article text."""

import datetime
import numpy as np
from keras.layers import Input
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense
from keras.models import Model
from headline_generation.utils.preprocessing import gen_embedding_weights, \
        vectorize_texts, format_inputs
from headline_generation.utils.data_io import return_data
from headline_generation.utils.mappings import create_mapping_dicts


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
    layer = LSTM(32, return_sequences=False)(layer)
    layer = Dense(dict_size, activation='softmax')(layer)

    lstm_model = Model(input=bodies, output=layer)

    lstm_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return lstm_model

def predict_w_model(lstm_model, X_test, y_test, idx_word_dct, save_filepath): 
    """Predict on an individual X using the inputted model. 

    After predicting, save the final to a file with associated timestamp. 

    Args: 
    ----
        lstm_model: keras.model.Model object
        X_test: 1d np.ndarray
            Contains 1 body of an article body pair to use for prediction
        y_test: list of ints
        idx_word_dct: dct
            Holds mapping of indices to words. 
        save_filepath: str
    """
    
    input_lst = X_test.tolist()
    y_pred = []
    for _ in range(len(y_test)): 
        # This is either messy here or below when updating input_lst. There's 
        # probably a better way. 
        pred_X = np.array(input_lst)[np.newaxis]
        pred_vector = lstm_model.predict(pred_X)
        pred = np.argmax(pred_vector)

        y_pred.append(pred)
        input_lst = input_lst[1:]
        input_lst.append(pred)

    predicted_heading = ' '.join(idx_word_dct[idx] for idx in y_pred)
    actual_heading = ' '.join(idx_word_dct[idx] for idx in y_test)

    with open(save_filepath, 'a+') as f: 
        out_str = '{} \n {} \n'.format(predicted_heading, actual_heading)
        f.write(out_str)
        
if __name__ == '__main__': 
    # Unfortunately, there are no real time savings from doing the following data   
    # loading and pre-processing ahead of time, which is why it's done here. 
    wrd_embedding = return_data("word_embedding")
    bodies, headlines = return_data("articles")
    bodies, headlines = bodies[0:2], headlines[0:2]

    word_idx_dct, idx_word_dct, word_vector_dct = \
            create_mapping_dicts(wrd_embedding, filter_corpus=True, bodies=bodies, 
                                 headlines=headlines)
    embedding_weights = gen_embedding_weights(word_idx_dct, word_vector_dct)

    bodies_arr, headlines_arr = vectorize_texts(bodies, headlines, word_idx_dct)
    vocab_size = len(word_vector_dct)
    maxlen = 50
    X, y = format_inputs(bodies_arr, headlines_arr, vocab_size=vocab_size, 
                         maxlen=maxlen)

    lstm_model = make_model(embedding_weights, input_length=maxlen)
    lstm_model.fit(X, y, nb_epoch=2000)

    dt = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
    preds_filepath = 'work/preds/{}.txt'.format(dt)
    for idx in range(2): 
        predict_w_model(lstm_model, X[idx], headlines_arr[idx], idx_word_dct,
                        preds_filepath)
