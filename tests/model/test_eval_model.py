import pytest
import numpy as np
from headline_generation.model.eval_model import return_xy_subset


class TestEvalModel: 
    
    def setup_class(cls): 
        cls.nobs, cls.n_feats = 50, 10
        
        cls.headlines = [['Two', 'words'], ['Three', 'words', '?'], ['Need', 'one'
                          'with', 'four']] 
        cls.X = np.random.randint(low=0, high=100, size=(cls.nobs, cls.n_feats)) 
        cls.y = np.random.randint(low=0, high=100, size=(cls.nobs, 1))

    def teardown_class(cls): 
        del cls.y
        del cls.X

    def test_return_xy_subset(self): 
        
        nobs = 2
        X_sub, y_sub, X0, y0, train_hlines, filtered_hlines = \
                return_xy_subset(self.X, self.y, self.headlines, nobs=nobs,     
                                 train=True)

        assert (X_sub.shape == (2, self.n_feats))
        assert (y_sub.shape == (2, 1))
        assert (X0.shape == (self.nobs, self.n_feats))
        assert (y0.shape == (self.nobs, 1))
        assert (len(train_hlines) == nobs)
        assert (len(filtered_hlines) == len(self.headlines))

        num_obs_removed = sum(len(headline) for headline in self.headlines[:nobs])
        X_sub, y_sub, X0, y0, test_hlines, filtered_hlines = \
                return_xy_subset(self.X, self.y, self.headlines, nobs=nobs, 
                                 train=False)

        assert (X_sub.shape == (2, self.n_feats))
        assert (y_sub.shape == (2, 1))
        assert (X0.shape == ((self.nobs - num_obs_removed), self.n_feats))
        assert (y0.shape == (self.nobs - num_obs_removed, 1))
        assert (len(test_hlines) == nobs)
        assert (len(filtered_hlines) == (len(self.headlines) - nobs))

