# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 07:13:49 2023

@author: shiva
"""


import pandas as pd
import os
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score

filename = 'Youtube01-Psy.csv'
path = 'C:/Users/shiva/Downloads/youtube+spam+collection'
fullpath = os.path.join(path,filename)

#### Loaded the data into the dataframe
spam_collection = pd.read_csv(fullpath)

#### First 3 records
print(spam_collection.head(3))

#### Shape of the dataframe
print(spam_collection.shape)

#### Column Names
print(spam_collection.columns.values.tolist())

#### Types of columns
print(spam_collection.dtypes)

#### Missing values per column
print(spam_collection.isnull().sum())

### Info of the dataframe
print(spam_collection.info())

columns_to_drop = ['COMMENT_ID', 'AUTHOR', 'DATE']
spam_collection.drop(columns=columns_to_drop, inplace=True)

X = spam_collection['CONTENT']
y = spam_collection['CLASS']

# Tokenize and remove stopwords
stop_words = set(stopwords.words('english'))
X = X.apply(lambda text: ' '.join([word for word in text.split() if word not in stop_words]))


count_vectorizer = CountVectorizer()

# Fit and transform the cleaned text
X_vectorized = count_vectorizer.fit_transform(X)

print(X_vectorized.toarray())

print("\nDimensions of training data:", X_vectorized.shape)

# Size of the vocabulary
print("Vocabulary size:", len(count_vectorizer.vocabulary_)) 
# Displaying some example feature names
print("Example feature names:", list(count_vectorizer.vocabulary_.keys())[:10]) 

# Downscale data
tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_vectorized)

# Display highlights of the output after TF-IDF transformation
print("Shape of TF-IDF transformed data:", X_tfidf.shape)

# Shuffle the dataset
shuffled_collection = spam_collection.sample(frac=1)
print(shuffled_collection.head())

# Split the dataset into training and testing sets
train_size = int(0.75 * shuffled_collection.shape[0])
X_train, X_test = X_tfidf[:train_size], X_tfidf[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Display shapes of training and testing sets
print("Training Features Shape:", X_train.shape)
print("Training Labels Shape:", y_train.shape)
print("Testing Features Shape:", X_test.shape)
print("Testing Labels Shape:", y_test.shape)

# Fit the Naive Bayes classifier with the training data
naive_bayes = MultinomialNB().fit(X_train, y_train)

# Perform 5-fold cross-validation
scores = cross_val_score(naive_bayes, X_train, y_train, cv=5)
print("Mean accuracy of Naive Bayes model (5-fold cross-validation):", scores.mean())

# Predict labels for the test data
y_pred = naive_bayes.predict(X_test)

# Print the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:") 
print(conf_matrix)

# Calculate and print the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Test with new comments
new_comments = [
    "The animation on season 2 of Jujutsu kaisen is trash!",
    "Mr beast is giving away free money on telegram on this link: https://t.me/MrBeast1000000",
    "The movies are so boring these days...They just create same old stories again and again.",
    "Check out this video link: https://www.youtube.com/watch?v=lBO1L8pgR9s&ab_channel=UnfoldDataScience",
    "Win a free iPhone 15 pro, click here!",
]

new_comments_counts = count_vectorizer.transform(new_comments)
new_comments_tfidf = tfidf_transformer.transform(new_comments_counts)
print(naive_bayes.predict(new_comments_tfidf))




