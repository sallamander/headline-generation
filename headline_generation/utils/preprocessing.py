"""A module for formatting article/headline pairs for a Keras model. 

This module contains functions for running article/headline pairs through an
embedding to vectorize them and prepare them to be run through an LSTM Keras model. 
"""

import numpy as np
from gensim.corpora.dictionary import Dictionary

def create_mapping_dicts(wrd_embedding, filter_corpus=False, bodies=None,
                         headlines=None): 
    """Generate word:index, word:vector, index:word dictionaries. 

    Args: 
    ----
        wrd_embedding: gensim.models.word2vec.Word2Vec fitted model
        filter_corpus (optional): boolean  
            Filter the corpus to only those words seen in the articles. Use
            to speed up iteration during intial building/training phases. 
        bodies (optional): list of lists 
            Must be passed in if `filter_corpus` is True. 
        headlines (optional): list of lists  
            Must be passed in if `filter_corpus` is True. 

    Return: 
    ------
        word_idx_dct: dict
        idx_word_dct: dict
        word_vector_dct: dict
    """

    if filter_corpus:
        if (not bodies or not headlines): 
            raise Exception('Must pass in bodies and headlines with filter_corpus as True!')
        else: 
            wrd_embedding = _filter_corpus(bodies, headlines, wrd_embedding)

    gensim_dct = Dictionary()
    gensim_dct.doc2bow(wrd_embedding.vocab.keys(), allow_update=True)

    word_idx_dct = {wrd: idx for idx, wrd in gensim_dct.items()}
    idx_word_dct = {idx: wrd for idx, wrd in gensim_dct.items()}
    word_vector_dct = {wrd: wrd_embedding[wrd] for idx, wrd in gensim_dct.items()}

    return word_idx_dct, idx_word_dct, word_vector_dct 

def _filter_corpus(bodies, headlines, wrd_embedding): 
    """Set the wrd_embeddding.vocab as the bag-of-words from bodies and headlines.

    Args: 
    ----
        bodies: list of lists
        headlines: list of lists 
        wrd_embedding: gensim.models.word2vec.Word2Vec fitted model

    Return: 
    ------
        wrd_embedding: gensim.models.word2vec.Word2Vec fitted model
            Original wrd_embedding passed in with `vocab` attribute changed. 
    """
    
    bodies_bow = set(word for body in bodies for word in body)
    headlines_bow = set(word for headline in headlines for word in headline)
    
    new_vocab = bodies_bow.union(headlines_bow)
    current_vocab = set(wrd_embedding.vocab.keys())
    filtered_vocab = current_vocab.intersection(new_vocab)
    
    new_vocab_dct = {}
    for word in filtered_vocab: 
        new_vocab_dct[word] = wrd_embedding.vocab[word]
    
    wrd_embedding.vocab = new_vocab_dct

    return wrd_embedding

def gen_embedding_weights(word_idx_dct, word_vector_dct): 
    """Generate the initial embedding weights to feed into Keras model.

    Args: 
    ----
        word_idx_dct: dict
        word_vector_dct: dict

    Return: 
    ------
        embedding_weights: 2d np.ndarry
    """

    n_words = len(word_idx_dct)
    # A little gross, but avoids loading all keys/values into memory. We need 
    # to access one of the lists and see how many dimensions each embedding has.
    n_dim = next(len(word_vector_dct[word]) for word in word_vector_dct)

    embedding_weights = np.zeros((n_words, n_dim))

    for wrd, idx in word_idx_dct.items():
        embedding_weights[idx, :] = word_vector_dct[wrd]

    return embedding_weights

def _vec_txt(words, word_idx_dct): 
    """Translate the inputted words into numbers using the index_dct. 

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

def vectorize_texts(texts, word_idx_dct): 
    """Translate each of the inputted text's words into numbers. 

    This calls the helper function `_vectorize_text`. 

    Args: 
    ----
        texts: list of lists 
        word_idx_dct: dct

    Return: 
    ------
        vectorized_words_arr: 1d np.ndarray
    """

    vec_texts = []
    for text in texts:  
        vec_text = _vec_txt(text, word_idx_dct)
        if vec_text: 
            vec_texts.append(vec_text)
        else: 
            # Used to later filter out empty vec_text.
            vec_texts.append(np.array(-99))

    vectorized_words_arr = np.array(vec_texts)

    return vectorized_words_arr

def filter_empties(bodies_arr, headlines_arr): 
    """Filter out bodies/headline pairs where the headline is empty.

    Args: 
    ----
        bodies_arr: 1d np.ndarray
        headlines_arr: 1d np.ndarray

    Return: 
    ------
        filtered_bodies: 1d np.ndarray
        filered_headlines: 1d np.ndarray
    """

    non_empty_idx = np.where(headlines_arr != -99)[0]
    filtered_bodies = bodies_arr[non_empty_idx]
    filtered_headlines = headlines_arr[non_empty_idx]

    return filtered_bodies, filtered_headlines

def format_inputs(bodies_arr, headlines_arr, vocab_size, maxlen=50, step=1): 
    """Format the body and headline arrays into the X,y matrices fed into the LSTM.

    Take the article bodies and headlines concatenated (e.g. a continuous array
    of words starting with the first word in the body and ending with the last
    word in the headline), and create (X, y) pairs to build up X and y matrices
    to feed into the LSTM. 
    
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
        bodies_arr: 1d np.ndarray of lists of strings
        headlines_arr: 1d np.ndarray of lists of strings
        vocab_size: int
        maxlen (optional): int
            How long to make the X sequences used for predicting. 
        step (optional): int
            How many words to skip over when passing through the concatenated
            article + body and generating (X,y) pairs 

    Return: 
    ------
        X_s: 2d np.ndarray
        y_s: 1d np.ndarray
    """

    X_s = np.zeros((0, maxlen)).astype('int32')
    ys = np.zeros((0, 1)).astype('int32')

    master_arr = []
    for body, hline in zip(bodies_arr, headlines_arr): 

        len_body, len_hline = len(body), len(hline)
        max_hline_len = (len_body - maxlen) // step

        if len_hline <= max_hline_len: 
            clipped_body = body[:(maxlen + len_hline)]
            clipped_body.extend(hline)
            master_arr.append((clipped_body, len_hline))


    for body_n_hline, len_hline in master_arr:
        for idx in range(0, len_hline, step): 
            X_start = idx
            X_end = X_start + maxlen
            X = np.array(body_n_hline[X_start:X_end])[np.newaxis]

            y_start = idx + maxlen + len_hline
            y_end = y_start + 1
            y = np.array(body_n_hline[y_start:y_end])[np.newaxis]

            X_s = np.concatenate([X_s, X])
            ys = np.concatenate([ys, y])

    # This is much faster than inserting in the above loop.
    y_s = np.zeros((X_s.shape[0], vocab_size)).astype('int32')
    idx = np.arange(X_s.shape[0])
    y_s[idx, ys] = 1

    return X_s, y_s

