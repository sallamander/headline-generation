import pytest
import numpy as np
from gensim.models.word2vec import Word2Vec
from headline_generation.utils.mappings import create_mapping_dicts, \
        map_idxs_to_str, map_xy_to_str

class TestMappings: 

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
        cls.word2vec_model = Word2Vec(sentences, min_count=1, size=len(cls.vocab))
        cls.word_idx_dct, cls.idx_word_dct, cls.word_vector_dct = \
                create_mapping_dicts(cls.word2vec_model)
        cls.vocab.add('\n')
    
    def teardown_class(cls): 
        del cls.vocab
        del cls.word2vec_model
        del cls.word_idx_dct
        del cls.idx_word_dct 
        del cls.word_vector_dct
        del cls.bodies
        del cls.headlines

    def test_create_mapping_dicts(self): 

        assert (len(self.word_idx_dct) == len(self.vocab))
        assert (len(self.idx_word_dct) == len(self.vocab))
        assert (len(self.word_vector_dct) == len(self.vocab))

        assert (set(self.word_idx_dct.keys()) == self.vocab)
        assert (set(self.idx_word_dct.values()) == self.vocab)
        assert (set(self.word_vector_dct.keys()) == self.vocab)

    def test_create_mapping_dicts_w_filter(self):
        
        bodies = [['body', 'words', 'and', 'stuf'], 
                  ['body2', 'mo', 'words', 'parrot', 'are' , 'talkative']]
        headlines = [['words', '!!'], ['par', '?']]
        bodies_set = set(word for body in bodies for word in body)
        headlines_set = set(word for headline in headlines for word in headline)

        test_vocab = bodies_set.union(headlines_set)

        word_idx_dct, idx_word_dct, word_vector_dct = \
                create_mapping_dicts(self.word2vec_model, filter_corpus=True, 
                                     bodies=bodies, headlines=headlines)

        assert (len(word_idx_dct) <= len(test_vocab))
        assert (len(idx_word_dct) <= len(test_vocab))
        assert (len(word_vector_dct) <= len(test_vocab))

    def test_map_idxs_to_str(self): 

        idx_word_dct = {0: 'Hello', 1: 'Goodbye', 2: 'TSwift', 3: 'Kanye'}

        idx_lst1 = [0, 2]
        idx_lst2 = [1, 3]
        
        stringified1 = map_idxs_to_str(idx_lst1, idx_word_dct)
        stringified2 = map_idxs_to_str(idx_lst2, idx_word_dct)

        assert (stringified1 == 'Hello TSwift')
        assert (stringified2 == 'Goodbye Kanye')

    def test_map_xy_to_str(self): 
        
        idx_word_dct = {0: 'PCA', 1: 'SVD', 2: 'NMF', 3: '?', 4: '!'}

        x1, y1 = np.array([0, 1, 2, 3]), np.array([0, 0, 0, 0, 1])
        x2, y2 = np.array([1, 1, 3, 3]), np.array([1, 0, 0, 0, 0])
        x3, y3 = np.array([2, 1, 3, 3]), np.array([1, 0, 1, 0, 1])
        x4, y4 = np.array([2, 2, 2, 2]), np.array([1, 1, 1, 1, 1])
        x5, y5 = np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0, 0])

        xs = [x1, x2, x3, x4, x5]
        ys = [y1, y2, y3, y4, y5]

        act_x1, act_y1 = "PCA SVD NMF ?", "!"
        act_x2, act_y2 = "SVD SVD ? ?", "PCA"
        act_x3, act_y3 = "NMF SVD ? ?", "PCA NMF !"
        act_x4, act_y4 = "NMF NMF NMF NMF", "PCA SVD NMF ? !"
        act_x5, act_y5 = "PCA PCA PCA PCA", ""

        act_xs = [act_x1, act_x2, act_x3, act_x4, act_x5]
        act_ys = [act_y1, act_y2, act_y3, act_y4, act_y5]

        for raw_x, raw_y, act_x, act_y in zip(xs, ys, act_xs, act_ys): 
            map_x, map_y = map_xy_to_str(raw_x, raw_y, idx_word_dct)
            
            assert (map_x == act_x)
            assert (map_y == act_y)

