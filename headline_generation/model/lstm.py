"""A script for fitting an LSTM to generate headlines for article text."""

import numpy as np
import pickle
from keras.preprocessing import sequence
from keras.layers import Input
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from gensim.models.word2vec import Word2Vec
from headline_generation.utils.preprocessing import create_mapping_dicts, \
        gen_embedding_weights, vectorize_texts, return_data 


def make_model(embedding_weights, max_features=300, batch_size=32, input_length=50):
    """Build an LSTM based off the input parameters and return it compiled. 

    Args: 
    ----
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

    input_dim = embedding_weights.shape[0]
    output_dim = embedding_weights.shape[1]

    lstm_model = Sequential()
    lstm_model.add(Embedding(input_dim=input_dim, output_dim=output_dim, 
                        input_length=input_length, weights=[embedding_weights]))

    lstm_model.compile('rmsprop', 'mse')

    return lstm_model

if __name__ == '__main__': 
    # Unfortunately, there are no real time savings from doing the following data   
    # loading and pre-processing ahead of time, which is why it's done here. 
    wrd_embedding = return_data("word_embedding")
    bodies, headlines = return_data("articles")

    idx_dct, vector_dct = create_mapping_dicts(wrd_embedding)
    embedding_weights = gen_embedding_weights(idx_dct, vector_dct)

    bodies_arr = vectorize_texts(bodies, idx_dct)
    headlines_arr = vectorize_texts(headlines, idx_dct)

    non_empty_idx = np.where(headlines_arr != -99)[0]
    X_s = bodies_arr[non_empty_idx]
    y_s = headlines_arr[non_empty_idx]

    input_length=50
    X_s = sequence.pad_sequences(X_s, input_length)
    lstm_model = make_model(embedding_weights, input_length=input_length)

    test_X = X_s[0:32]
    output_test = lstm_model.predict(test_X)

    
