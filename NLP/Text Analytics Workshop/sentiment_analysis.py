# -*- coding: utf-8 -*-
"""
Created July 2017

@author: arw
"""

import pandas as pd
import numpy as np
from normalization import normalize_corpus
from utils import build_feature_matrix

# Load the cleaned movie reviews dataset
dataset = pd.read_csv(r'movie_reviews.csv')
# Check how big the dataset frame is using len() function
# Print the first few data points - note that data consists of 2 columns named 'review' and 'sentiment'
print(dataset.head())

# Divide data into training and testing sets
train_data = dataset[:25000]
test_data = dataset[25000:] 
# Check size (len) and first few elements (head()) of test_data (sub)frame


# Divide the data into the data (review) and the label (sentiment) in both training and testing sets
train_reviews = np.array(train_data['review'])
train_sentiments = np.array(train_data['sentiment'])
test_reviews = np.array(test_data['review'])
test_sentiments = np.array(test_data['sentiment'])

# Normalize the training review data using the normalization.py module
norm_train_reviews = normalize_corpus(train_reviews,
                                      lemmatize=False,
                                      only_text_chars=True)

# Extract features from these normalized training reviews
# - which features? Try other features using parameters provided in utils.py                                                                           
vectorizer, train_features = build_feature_matrix(documents=norm_train_reviews,
                                                  feature_type='tfidf',
                                                  ngram_range=(1, 1), 
                                                  min_df=0.0, max_df=1.0)                                      
                                      

from sklearn.linear_model import SGDClassifier

# Build/train an SVM classifier model with the train features extracted from reviews
svm = SGDClassifier(loss='hinge', n_iter=500)
svm.fit(train_features, train_sentiments) # We give the features and the correct labels



# Normalize the test reviews                        
norm_test_reviews = normalize_corpus(test_reviews,
                                     lemmatize=False,
                                     only_text_chars=True)  
# Extract features from the normalized test reviews                                   
test_features = vectorizer.transform(norm_test_reviews)         

# Predict the sentiment for test dataset movie reviews
predicted_sentiments = svm.predict(test_features)       

# Evaluate model prediction performance by comparing predicted sentiments and test sentiments
from utils import display_evaluation_metrics, display_confusion_matrix, display_classification_report

# Show performance metrics
display_evaluation_metrics(true_labels=test_sentiments,
                           predicted_labels=predicted_sentiments,
                           positive_class='positive')  

# Show confusion matrix
display_confusion_matrix(true_labels=test_sentiments,
                         predicted_labels=predicted_sentiments,
                         classes=['positive', 'negative'])

# Show detailed per-class classification report
display_classification_report(true_labels=test_sentiments,
                              predicted_labels=predicted_sentiments,
                              classes=['positive', 'negative'])


