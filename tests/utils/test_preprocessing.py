import pytest 
import numpy as np
from gensim.models.word2vec import Word2Vec
from headline_generation.utils.preprocessing import create_mapping_dicts, \
        gen_embedding_weights, vectorize_texts, filter_empties

class TestPreprocessing: 

    def setup_class(cls): 
        sentence1 = ['This', 'is', 'a', 'first', 'sentence', 'of', 'words']
        sentence2 = ['This', 'is', 'a', 'second', 'sentence', 'of', 'words']
        sentence3 = ['Does', 'anybody', 'think', 'I', 'need', 'more', 'variety', '?']

        cls.sentences = [sentence1, sentence2, sentence3]
        
        cls.vocab = set(word for sentence in cls.sentences for word in sentence)
        cls.word2vec_model = Word2Vec(cls.sentences, min_count=1, size=len(cls.vocab))

        cls.word_idx_dct, cls.idx_word_dct, cls.word_vector_dct = \
                create_mapping_dicts(cls.word2vec_model)
    
    def teardown_class(cls): 
        del cls.vocab
        del cls.word2vec_model
        del cls.word_idx_dct
        del cls.idx_word_dct 
        del cls.word_vector_dct

    def test_mapping_dicts(self): 

        assert (len(self.word_idx_dct) == len(self.vocab))
        assert (len(self.idx_word_dct) == len(self.vocab))
        assert (len(self.word_vector_dct) == len(self.vocab))

        assert (set(self.word_idx_dct.keys()) == self.vocab)
        assert (set(self.idx_word_dct.values()) == self.vocab)
        assert (set(self.word_vector_dct.keys()) == self.vocab)

    def test_mapping_dicts_w_filter(self):
        
        bodies = [['body1', 'words', 'and', 'stuff'], 
                  ['body2', 'more', 'words', 'parrots', 'are' , 'talkative']]
        headlines = [['words', '?'], ['parrots', '?']]
        bodies_set = set(word for body in bodies for word in body)
        headlines_set = set(word for headline in headlines for word in headline)

        test_vocab = bodies_set.union(headlines_set)

        word_idx_dct, idx_word_dct, word_vector_dct = \
                create_mapping_dicts(self.word2vec_model, filter_corpus=True, 
                                     bodies=bodies, headlines=headlines)

        assert (len(word_idx_dct) <= len(test_vocab))
        assert (len(idx_word_dct) <= len(test_vocab))
        assert (len(word_vector_dct) <= len(test_vocab))

    def test_gen_embedding_weights(self): 

        embedding_weights = gen_embedding_weights(self.word_idx_dct, 
                                                  self.word_vector_dct)

        assert (len(self.word_idx_dct) == embedding_weights.shape[0])
        assert (len(self.vocab) == embedding_weights.shape[1])        

    def test_vectorize_texts(self): 
         
        vectorized_words_arr = vectorize_texts(self.sentences, self.word_idx_dct)

        assert (type(vectorized_words_arr) == np.ndarray)
        assert (vectorized_words_arr.shape[0] == len(self.sentences))
        
    def test_filter_empties(self): 
        
        bodies = np.array([[0], [1], [99], [63], [-99]])
        headlines = np.array([[-99], [42], [9], [100], [-99]])

        filtered_bodies, filtered_headlines = filter_empties(bodies, headlines) 

        assert (len(filtered_bodies) == 3)
        assert (len(filtered_headlines == 3))


        

