"""A script to format article/headline pairs for a Keras model. 

This script takes in raw article/headline pairs, runs them through a Word2Vec 
embedding trained on GoogleNews, and then formats them to be later inputted into
a Keras LSTM. 
"""

import numpy as np
import pickle
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary

def create_mapping_dicts(wrd_embedding): 
    """Generate word:index and word:vector dictionaries. 

    Args: 
    ----
        word_embedding: gensim.models.word2vec.Word2Vec fitted model

    Return: 
    ------
        idx_dct: dict
        vector_dct: dict
    """

    gensim_dct = Dictionary()
    gensim_dct.doc2bow(wrd_embedding.vocab.keys(), allow_update=True)

    idx_dct = {wrd: idx + 1 for idx, wrd in gensim_dct.items()}
    vector_dct = {wrd: wrd_embedding[wrd] for idx, wrd in gensim_dct.items()}

    return idx_dct, vector_dct 

def _vec_txt(words, idx_dct): 
    """Translate the inputted words into numbers using the index_dct. 

    This is a helper function to `vectorize_txts`. 

    Args: 
    ----
        words: list of strings
        idx_dct: dct
            Holds a mapping of words to numbers (indices). 

    Return: 
    ------
        vectorized_words_lst: list of ints
    """

    vectorized_words_lst = []
    for word in words: 
        if word in idx_dct: 
            vectorized_words_lst.append(idx_dct[word])

    return vectorized_words_lst

def vectorize_texts(texts, idx_dct): 
    """Translate each of the inputted text's words into numbers. 

    This calls the helper function `_vectorize_text`. 

    Args: 
    ----
        texts: list of lists 
        idx_dct: dct
            Holds a mapping of words to number (indices). 

    Return: 
    ------
        vectorized_words_arr: 1d np.ndarray
    """

    vec_texts = []
    for text in texts:  
        vec_text = _vec_txt(text, idx_dct)
        if vec_text: 
            vec_texts.append(vec_text)
        else: 
            # Used to later filter out empty vec_text.
            vec_texts.append(np.array(-99))

    vectorized_words_arr = np.array(vec_texts)

    return vectorized_words_arr

if __name__ == '__main__': 

    embedding_fp = '../../data/word_embeddings/google_news_300dim.bin'
    wrd_embedding = Word2Vec.load_word2vec_format(embedding_fp, binary=True)
    
    body_fp = '../../data/articles/twenty_newsgroups/bodies.pkl'
    headline_fp = '../../data/articles/twenty_newsgroups/headlines.pkl'

    with open(body_fp, 'rb') as f: 
        bodies = pickle.load(f)
    with open(headline_fp, 'rb') as f: 
        headlines = pickle.load(f)

    idx_dct, vector_dct = create_mapping_dicts(wrd_embedding)
    bodies_arr = vectorize_texts(bodies, idx_dct)
    headlines_arr = vectorize_texts(headlines, idx_dct)

    non_empty_idx = np.where(headlines_arr != -99)[0]
    bodies_arr = bodies_arr[non_empty_idx]
    headlines_arr = headlines_arr[non_empty_idx]

    body_fp = '../../work/articles/twenty_newsgroups/bodies.pkl'
    headline_fp = '../../work/articles/twenty_newsgroups/headlines.pkl'

    with open(body_fp, 'wb+') as f: 
        pickle.dump(bodies_arr, f)
    with open(headline_fp, 'wb+') as f: 
        pickle.dump(headlines_arr, f)
