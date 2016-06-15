import pytest 
import numpy as np
from gensim.models.word2vec import Word2Vec
from headline_generation.utils.preprocessing import gen_embedding_weights, \
        vectorize_texts, format_inputs
from headline_generation.utils.mappings import create_mapping_dicts

class TestPreprocessing: 

    def setup_class(cls): 
        sentences = [['body1', 'words', 'and', 'stuff', 'words', '?'], 
                     ['body2', 'more', 'words', 'parrots', 'are', 'talkative',
                      'parrots', '?']]
        cls.bodies = [['body1', 'words', 'and', 'stuff'], 
                  ['body2', 'more', 'words', 'parrots', 'are' , 'talkative']]
        cls.headlines = [['words', '?'], ['parrots', '?']]

        bodies_set = set(word for body in cls.bodies for word in body)
        headlines_set = set(word for headline in cls.headlines for word in headline)

        cls.vocab = bodies_set.union(headlines_set)
        word2vec_model = Word2Vec(sentences, min_count=1, size=len(cls.vocab))
        cls.word_idx_dct, cls.idx_word_dct, cls.word_vector_dct = \
                create_mapping_dicts(word2vec_model)
    
        cls.vec_bodies_arr, cls.vec_headlines_arr = \
                vectorize_texts(cls.bodies, cls.headlines, cls.word_idx_dct)

    def teardown_class(cls): 
        del cls.vocab
        del cls.word_idx_dct
        del cls.idx_word_dct 
        del cls.word_vector_dct
        del cls.bodies
        del cls.headlines
        del cls.vec_bodies_arr
        del cls.vec_headlines_arr

    def test_gen_embedding_weights(self): 

        embedding_weights = gen_embedding_weights(self.word_idx_dct, 
                                                  self.word_vector_dct)

        assert (len(self.word_idx_dct) == embedding_weights.shape[0])
        assert (len(self.vocab) == embedding_weights.shape[1])        

    def test_vectorize_texts(self): 
         
        assert (type(self.vec_bodies_arr) == np.ndarray)
        assert (type(self.vec_headlines_arr) == np.ndarray)
        assert (self.vec_bodies_arr.shape[0] <= len(self.bodies))
        assert (self.vec_headlines_arr.shape[0] <= len(self.headlines))

    def test_format_inputs(self): 

        X, y, filtered_bodies, filtered_headlines = \
                format_inputs(self.vec_bodies_arr, self.vec_headlines_arr, 
                              len(self.vocab), maxlen=2, step=1)

        assert (len(filtered_bodies) <= self.vec_bodies_arr.shape[0])
        assert (len(filtered_headlines) <= self.vec_headlines_arr.shape[0])

        assert (X.shape[0] == 4)
        assert (X.shape[1] == 2)
        assert (y.shape[0] == 4)
        assert (y.shape[1] == len(self.vocab))
        
        x0_str = ' '.join(self.idx_word_dct[idx] for idx in X[0])
        x1_str = ' '.join(self.idx_word_dct[idx] for idx in X[1])
        x2_str = ' '.join(self.idx_word_dct[idx] for idx in X[2])
        x3_str = ' '.join(self.idx_word_dct[idx] for idx in X[3])

        assert (x0_str == "body1 words")
        assert (x1_str == "words words")
        assert (x2_str == "body2 more")
        assert (x3_str == "more parrots")
        
        y0_idx, y1_idx = np.where(y[0] == 1)[0][0], np.where(y[1] == 1)[0][0]
        y2_idx, y3_idx = np.where(y[2] == 1)[0][0], np.where(y[3] == 1)[0][0] 
        y0_str, y1_str = self.idx_word_dct[y0_idx], self.idx_word_dct[y1_idx] 
        y2_str, y3_str = self.idx_word_dct[y2_idx], self.idx_word_dct[y3_idx] 

        assert (y0_str == "words")
        assert (y1_str == "?")
        assert (y2_str == "parrots")
        assert (y3_str == "?")
