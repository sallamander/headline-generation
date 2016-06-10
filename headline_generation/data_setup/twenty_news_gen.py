"""A script for reading in and cleaning the twenty news groups dataset."""

from sklearn.datasets import fetch_20newsgroups
import re

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
        (body, headline): tuple 
    """

    found_headline = re.findall("(?<=Subject:)(.*)", article)
    found_body = re.findall("(?<=Lines:)(?s)(.*)", article)

    # For 58 of the 18,846 articles in the data, no headline and/or body was found.  
    if found_headline and found_body: 
        headline = found_headline[0]
        body = found_body[0]
        return (body, headline)

if __name__ == '__main__': 
    articles = fetch_20newsgroups(subset='all').data
    bodies_n_headlines = [grab_body_headline(article) for article in articles]


    
