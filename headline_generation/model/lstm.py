"""A script for fitting an LSTM to generate headlines for article text."""

from keras.preprocessing import sequence
from keras.layers import Input
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense
from keras.models import Model
from headline_generation.utils.preprocessing import create_mapping_dicts, \
        gen_embedding_weights, vectorize_texts, filter_empties
from headline_generation.utils.data_io import return_data


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
    layer = Dense(dict_size)(layer)

    lstm_model = Model(input=bodies, output=layer)
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
    X_s, y_s = filter_empties(bodies_arr, headlines_arr)

    input_length=50
    X_s = sequence.pad_sequences(X_s, input_length)
    '''
    lstm_model = make_model(embedding_weights, input_length=input_length)

    test_X = X_s[0:32]
    output_test = lstm_model.predict(test_X)
    ''' 
