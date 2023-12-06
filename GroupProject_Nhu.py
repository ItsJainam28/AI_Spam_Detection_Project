# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 08:30:21 2023

@author: Quynh Nhu
"""
# Input file containing data
import numpy as np
import pandas as pd
import os
path = "D:/MyCentennial/COMP237_IntroductionToAI/"
filename = 'Youtube01-Psy.csv'
fullpath = os.path.join(path,filename)
input_file = fullpath

# Load data from input file
data = pd.read_csv(fullpath)

# Basic data exploration
data.columns.values
data.shape
data.describe()
data.dtypes
data.head(5)

# Text preprocessing
import nltk 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
stop_words = set(stopwords.words('english'))

count_vectorizer = CountVectorizer()
train_tc = count_vectorizer.fit_transform(data['CONTENT'])
print("\nDimensions of training data:", train_tc.shape)

tfidf = TfidfTransformer()
train_tfidf = tfidf.fit_transform(train_tc)
type(train_tfidf)

# Shuffle the dataset
data = data.sample(frac=1)

# Split dataset into training and testing sets
train_size = int(0.75 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]

# Fit the training data into a Naive Bayes classifier
clf = MultinomialNB().fit(train_tfidf, data['CLASS'])

# Cross validate the model on the training data using 5-fold
scores = cross_val_score(clf, train_tfidf, data['CLASS'], cv=5)
print("Cross validation score: ", scores)
print("Mean cross validation score: ", np.mean(scores))

# Test the model on the test data
X_test_counts = count_vectorizer.transform(test_data['CONTENT'])
X_test_tfidf = tfidf.transform(X_test_counts)
predicted = clf.predict(X_test_tfidf)

# Print the confusion matrix and the accuracy of the model
print("Confusion matrix:\n",confusion_matrix(test_data['CLASS'], predicted))
print("Accuracy of the model on the test data: ", {np.mean(predicted == test_data['CLASS'])})