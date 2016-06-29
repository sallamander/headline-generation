"""A module for formatting article/headline pairs for a Keras model. 

This module contains functions for running article/headline pairs through an
embedding to vectorize them. 
"""

import numpy as np
from keras.utils.np_utils import to_categorical 
from headline_generation.utils.mappings import map_idxs_to_str

def _vec_txt(words, word_idx_dct): 
    """Translate the inputted words into numbers using the `word_idx_dct`. 

    This is a helper function to `vectorize_txts`. 

    Args: 
    ----
        words: list of strings
        word_idx_dct: dct

    Return: 
    ------
        vectorized_words_lst: list of ints
    """

    vectorized_words_lst = []
    for word in words: 
        if word in word_idx_dct: 
            vectorized_words_lst.append(word_idx_dct[word])

    return vectorized_words_lst

def vectorize_texts(bodies, headlines, word_idx_dct): 
    """Translate each of the inputted text's words into numbers. 

    This calls the helper function `_vec_txt`. 

    Args: 
    ----
        bodies: list of lists of strings
        headlines: list of lists of strings
        word_idx_dct: dict

    Return: 
    ------
        vec_bodies: 1d np.ndarray of lists 
        vec_headlines: 1d np.ndarray of lists
    """

    vec_bodies = []
    vec_headlines = []
    for body, headline in zip(bodies, headlines):  
        vec_body = _vec_txt(body, word_idx_dct)
        vec_headline = _vec_txt(headline, word_idx_dct)
        if vec_body and vec_headline: 
            vec_bodies.append(vec_body)
            vec_headlines.append(vec_headline)
    
    return vec_bodies, vec_headlines 

def format_inputs(vec_bodies, vec_headlines, vocab_size, maxlen=50, step=1): 
    """Format the body and headline arrays into the X,y matrices fed into the model.

    Take the article bodies and headlines concatenated (e.g. a continuous array
    of words starting with the first word in the body and ending with the last
    word in the headline), and create (X, y) pairs to build up X and y matrices
    
    Building these (X, y) pairs includes: 
        - Dropping any body/article pairs where the body is less than the `maxlen` 
          plus the length of the heading (and accounting for step size); this allows 
          for the number of samples per body/article pair to be equal to the number  
          of words in the heading 
        - Taking the first `maxlen` + headline length words of the body and stepping
          through those by the `step` to obtain X's, and stepping through the 
          words of the heading by `step` to obtain the corresponding y
    
    Args: 
    ----
        vec_bodies: list of lists ints
        vec_headlines: list of lists ints
        vocab_size: int
        maxlen (optional): int
            How long to make the X sequences used for predicting. 
        step (optional): int
            How many words to step by when passing through the concatenated
            article + body and generating (X,y) pairs 

    Return: 
    ------
        Xs: 2d np.ndarray
        ys: 2d np.ndarray
        filtered_bodies: list 
        filtered_headlines: list
    """

    Xs, ys = [], []

    filtered_bodies = []
    filtered_headlines = []
    for body, hline in zip(vec_bodies, vec_headlines): 
        
        len_body, len_hline = len(body), len(hline)
        max_hline_len = (len_body - maxlen) // step
        hline.append(0) # Append the newline character. 

        if len_hline <= max_hline_len: 
            for idx, word in enumerate(hline): 
                X = body[idx:maxlen] + [0] + hline[:idx]
                y = hline[idx]

                Xs.append(X)
                ys.append(y)

            filtered_bodies.append(body)
            filtered_headlines.append(hline)
    
    # One-hot encode y.
    ys = to_categorical(ys, nb_classes=vocab_size)
    Xs = np.array(Xs, dtype='int32')

    return Xs, ys, filtered_bodies, filtered_headlines

