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
        index_dct: dict
        vector_dct: dict
    """

    gensim_dct = Dictionary()
    gensim_dct.doc2bow(wrd_embedding.vocab.keys(), allow_update=True)

    index_dct = {wrd: index + 1 for index, wrd in gensim_dct.items()}
    vector_dct = {wrd: wrd_embedding[wrd] + 1 for index, wrd in gensim_dct.items()}

    return index_dct, vector_dct 

if __name__ == '__main__': 

    embedding_fp = '../../data/word_embeddings/google_news_300dim.bin'
    wrd_embedding = Word2Vec.load_word2vec_format(embedding_fp, binary=True)
    
    body_fp = '../../data/articles/twenty_newsgroups/bodies.pkl'
    headline_fp = '../../data/articles/twenty_newsgroups/headlines.pkl'

    with open(body_fp, 'rb') as f: 
        bodies_arr = np.array(pickle.load(f))
    with open(headline_fp, 'rb') as f: 
        headlines_arr = np.array(pickle.load(f))

    index_dct, vector_dct = create_mapping_dicts(wrd_embedding)
