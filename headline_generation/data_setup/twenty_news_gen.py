"""A script for reading in and cleaning the twenty news groups dataset."""

import re
import string
import pickle
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def grab_body_headline(article):
    """Grab the headline and body text from the inputted article text. 

    The article is one long string, and the parts to grab are stored within certain
    fields, denoted by the name of the field followed by a colon ("Subject:" for 
    the headline, "Lines:" for the body). 

    Args: 
    ----
        article: str

    Return: 
    ------
        (body, headline): tuple of strings
            Returns (None, None) if no body and/or headline is found. 
    """

    found_headline = re.findall("(?<=Subject:)(.*)", article)
    found_body = re.findall("(?<=Lines:)(?s)(.*)", article)

    # For 58 of the 18,846 articles in the data, no headline and/or body was found.  
    if found_headline and found_body: 
        headline = found_headline[0]
        body = found_body[0]
        return (body, headline)
    else: 
        # Return (None, None) to allow continuation of the pipline, and filter later.
        return (None, None)

def clean_raw_txt(body, headline, punct_dct=None, stopwrds_set=None): 
    """Clean the body and headline to remove punctuation, stopwords, etc.

    Clean the text according to the input specifications. 

    Args: 
    ----
        body: str
        headline: str
        punct_dct (optional): dict 
            Translation dict resulting from a `str.maketrans()` call             
        stopwords_set (optional): set  

    Return: 
    ------
        (body, headline): tuple
    """

    if punct_dct: 
        body = body.translate(punct_dct)
        headline = headline.translate(punct_dct)

    body_wrds = word_tokenize(body)
    headline_wrds = word_tokenize(headline)

    if stopwrds_set: 
        body_wrds = [wrd.lower() for wrd in body_wrds if wrd.lower() not in stopwrds_set] 
        headline_wrds = [wrd.lower() for wrd in headline_wrds if wrd.lower() not in stopwrds_set]

    return (body_wrds, headline_wrds)

if __name__ == '__main__': 
    articles = fetch_20newsgroups(subset='all').data
    bodies_n_headlines = [grab_body_headline(article) for article in articles]

    punct_dct = str.maketrans({punct_mark: "" for punct_mark in string.punctuation})
    stopwrds_set = set(stopwords.words('english'))
    
    cleaned_bodies = []
    cleaned_headlines = []
    for body, headline in bodies_n_headlines: 
        if body and headline: 
            body, headline = clean_raw_txt(body, headline, punct_dct, stopwrds_set)
            cleaned_bodies.append(body)
            cleaned_headlines.append(headline)
    
    body_fp = 'data/articles/twenty_newsgroups/bodies.pkl'
    headline_fp = 'data/articles/twenty_newsgroups/headlines.pkl'

    with open(body_fp, 'wb+') as f: 
        pickle.dump(cleaned_bodies, f)
    with open(headline_fp, 'wb+') as f: 
        pickle.dump(cleaned_headlines, f)
