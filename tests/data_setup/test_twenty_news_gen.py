import pytest
import numpy as np
import string
from nltk.corpus import stopwords
from headline_generation.data_setup.twenty_news_gen import grab_body_headline, \
            clean_raw_txt

class TestTwentyNews:

    def setup_class(cls): 
        bag_of_words = np.array(['Random', 'words', 'used', 'to' 'generate', 'some', 
                        'random', 'subject', 'and', 'body', 'pairs', 'lets', 
                        'keep', 'filling', 'it', 'with', 'words', 'wohoo', 
                        'for', 'yoohoo', 'right', 'Adam', 'Sandlar', '?', '.', '!', 
                        ':'])

        cls.rand_txts = []
        cls.rand_bodies= []
        cls.rand_headings = []

        rand_int_pairs = np.random.randint(0, 10, (10, 2))
        for rand1, rand2 in rand_int_pairs: 
            body = ' '.join(np.random.choice(bag_of_words, size=rand1))
            headline = ' '.join(np.random.choice(bag_of_words, size=rand2))
            full_txt = "Subject:{}\nLines:{}".format(headline, body)

            cls.rand_txts.append(full_txt)
            cls.rand_bodies.append((body))
            cls.rand_headings.append((headline))

        cls.punct_set = set(string.punctuation)
        cls.punct_dct = str.maketrans({punct_mark: "" for punct_mark in \
                                   string.punctuation})
        cls.stopwrds_set = set(stopwords.words('english'))

    def teardown_class(cls): 
        del cls.rand_txts
        del cls.rand_bodies 
        del cls.rand_headings 
        del cls.punct_set
        del cls.punct_dct
        del cls.stopwrds_set

    def test_grab_body_headline(self): 
        
        for txt, tst_bdy, tst_heading in zip(self.rand_txts, self.rand_bodies, 
                self.rand_headings): 

            bdy, heading = grab_body_headline(txt)

            assert (bdy == tst_bdy)
            assert (heading == tst_heading)

    def test_clean_raw_txt(self):

        for txt in self.rand_txts: 
            body, heading = grab_body_headline(txt)
            clean_b, clean_h = clean_raw_txt(body, heading, self.punct_dct, 
                                             self.stopwrds_set)
            
            set_b, set_h = set(clean_b), set(clean_h)
            assert ((self.punct_set - set_b) == self.punct_set)
            assert ((self.punct_set - set_h) == self.punct_set)
            assert ((self.stopwrds_set - set_h) == self.stopwrds_set)
            assert ((self.stopwrds_set - set_h) == self.stopwrds_set)

