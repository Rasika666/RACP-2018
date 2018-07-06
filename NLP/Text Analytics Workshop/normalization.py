# -*- coding: utf-8 -*-
"""
Created July 2017

@author: arw
"""

from contractions import CONTRACTION_MAP # contractions.py should be in path
import re # Regular expression package
import nltk # The natural language toolkit
import string # Collection of string constants 
from nltk.stem import WordNetLemmatizer # English lemmatizer based on WordNet
from html.parser import HTMLParser # Package for parsing html
from unidecode import unidecode # Package to 'normalize' accented characters

# We use NLTK's stopword list for English - how many words does it have? What are stopwords?
stopword_list = nltk.corpus.stopwords.words('english')
# We can add some domain-specific stopwords as needed
stopword_list = stopword_list + ['mr', 'mrs', 'come', 'go', 'get',
                                 'tell', 'listen', 'one', 'two', 'three',
                                 'four', 'five', 'six', 'seven', 'eight',
                                 'nine', 'zero', 'join', 'find', 'make',
                                 'say', 'ask', 'tell', 'see', 'try', 'back',
                                 'also']

# For lemmatizing words, we use NLTK's WordNet Lemmatizer
wnl = WordNetLemmatizer()
# For stripping HTML and escape sequences we use html.parser
html_parser = HTMLParser()

# We define a function around NLTK's word tokenizer
def tokenize_text(text):
    tokens = nltk.word_tokenize(text) 
    tokens = [token.strip() for token in tokens]
    return tokens

# We replace tokens that match the LHS of CONTRACTION_MAP with its RHS
def expand_contractions(text, contraction_mapping):
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text
    

from nltk.corpus import wordnet as wn

# Annotate text tokens with POS tags
def pos_tag_text(text):    
    def penn_to_wn_tags(pos_tag):
        if pos_tag.startswith('J'):
            return wn.ADJ
        elif pos_tag.startswith('V'):
            return wn.VERB
        elif pos_tag.startswith('N'):
            return wn.NOUN
        elif pos_tag.startswith('R'):
            return wn.ADV
        else:
            return None
    
    tokens = nltk.word_tokenize(text)
    tagged_text = nltk.pos_tag(tokens)
    
    tagged_lower_text = [(word.lower(), penn_to_wn_tags(pos_tag))
                         for word, pos_tag in
                         tagged_text]
    return tagged_lower_text
    
# Lemmatize text based on POS tags    
def lemmatize_text(text):
    pos_tagged_text = pos_tag_text(text)
    lemmatized_tokens = [wnl.lemmatize(word, pos_tag) if pos_tag
                         else word                     
                         for word, pos_tag in pos_tagged_text]
    lemmatized_text = ' '.join(lemmatized_tokens)
    return lemmatized_text
    
# Remove special characters and punctuation
def remove_special_characters(text):
    tokens = tokenize_text(text)
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub(' ', token) for token in tokens])
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text
    
# Remove function words that bear little content    
def remove_stopwords(text):
    tokens = tokenize_text(text)
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

# Use this to remove any tokens that have numeric characters
def keep_text_characters(text):
    filtered_tokens = []
    tokens = tokenize_text(text)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

# We define a function to 'normalize' characters with accents (from other Latin languages)
def normalize_accented_characters(text):
    text = unidecode(text)
    return text


# Call all the functions defined above for each text in the corpus
def normalize_corpus(corpus, lemmatize=True, 
                     only_text_chars=False,
                     tokenize=False):
    
    normalized_corpus = []    
    for index, text in enumerate(corpus):
        # We call the function for normalizing accented Latin characters
        text = normalize_accented_characters(text)
        # Next we remove the 'unescaped characters'
        text = html_parser.unescape(text)
        # We now expand contractions using the CONTRACTION MAP
        text = expand_contractions(text, CONTRACTION_MAP)
        # If asked to lemmatize (optional) we call the WordNet Lemmatizer in NLTK
        if lemmatize:
            text = lemmatize_text(text)
        else:
            text = text.lower()
        # We now remove special characters
        text = remove_special_characters(text)
        text = remove_stopwords(text)
        if only_text_chars:
            text = keep_text_characters(text)
        
        if tokenize:
            text = tokenize_text(text)
            normalized_corpus.append(text)
        else:
            normalized_corpus.append(text)
            
    return normalized_corpus

