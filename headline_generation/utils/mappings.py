"""A module for generating word/idx/vector mappings and moving between them. """

from gensim.corpora.dictionary import Dictionary
import numpy as np

def create_mapping_dicts(wrd_embedding, filter_corpus=False, bodies=None,
                         headlines=None): 
    """Generate word:index, word:vector, index:word dictionaries. 

    Args: 
    ----
        wrd_embedding: gensim.models.word2vec.Word2Vec fitted model
        filter_corpus (optional): boolean  
            Filter the corpus to only those words seen in the articles. 
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
            excep_str = "Must pass in bodies and headlines with filter_corpus True!"
            raise Exception(excep_str)
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
            Original wrd_embedding with `vocab` attribute changed. 
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

def map_idxs_to_str(idx_lst, idx_word_dct): 
    """Return a string by mapping integers in the `idx_lst` to words. 

    Serves primarily as a helper function to `map_xy_to_str`, but is built alone to 
    allow external calls. 

    Args: 
    ----
        idx_lst: list of ints 
        idx_word_dct: dict 

    Return: 
    ------
        stringified: str
    """

    stringified = ' '.join(idx_word_dct[idx] for idx in idx_lst)
    return stringified 

def map_xy_to_str(x, y, idx_word_dct): 
    """Return a string for an inputted x and y. 

    Since the x vector contains all ints, simply run it through `map_idx_to_str`. 
    The y vector is one-hot encoded, meaning we need to grab the index corresponding
    to the 1 and then run that through `map_idxs_to_str`. 

    Args: 
    ----
        x: 1d np.ndarray
        y: 1d np.ndarray
        idx_word_dct: dict

    Return: 
    ------
        stringified_x: str
        stringified_y: str
    """

    stringified_x = map_idxs_to_str(x, idx_word_dct)
    y_hot_idx = np.where(y == 1)[0]
    stringified_y = map_idxs_to_str(y_hot_idx, idx_word_dct)

    return stringified_x, stringified_y

