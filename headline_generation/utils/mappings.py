"""A module for generating word/idx/vector mappings and moving between them. """

from gensim.corpora.dictionary import Dictionary

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
