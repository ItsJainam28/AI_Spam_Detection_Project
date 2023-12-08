# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 08:30:21 2023

@author: Quynh Nhu
"""
# Input file containing data
import numpy as np
import pandas as pd
import os
import nltk 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
stop_words = set(stopwords.words('english'))

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
count_vectorizer = CountVectorizer()
train_tc = count_vectorizer.fit_transform(data['CONTENT'])
print("\nDimensions of training data:", train_tc.shape)

tfidf = TfidfTransformer()
train_tfidf = tfidf.fit_transform(train_tc)
type(train_tfidf)

# Shuffle the dataset
data = data.sample(frac=1)

# Split dataset into training and testing sets
train_size = int(0.75 * data.shape[0])
X_train, X_test = train_tfidf[:train_size], train_tfidf[train_size:]
y_train, y_test = data['CLASS'][:train_size], data['CLASS'][train_size:]

# Fit the training data into a Naive Bayes classifier
clf = MultinomialNB().fit(X_train, y_train)

# Cross validate the model on the training data using 5-fold
scores = cross_val_score(clf, X_train, y_train, cv=5)
print("Cross validation score: ", scores)
print("Mean cross validation score: ", scores.mean())

# Test the model on the test data
y_pred = clf.predict(X_test)
print("Confusion matrix:\n",confusion_matrix(y_test, y_pred))
print("Accuracy of the model on the test data: ", accuracy_score(y_test, y_pred))