# Step 1: Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Step 2: Load the data
df = pd.read_csv('Youtube01-Psy.csv')

# Step 3: Basic data exploration
print(df.head())
print(df.info())

# Step 4: Prepare the data
X = df['CONTENT']
y = df['CLASS']

# Tokenize and remove stopwords

stop_words = set(stopwords.words('english'))
X = X.apply(lambda x: ' '.join([word for word in word_tokenize(x) if word not in stop_words]))

# Step 5: Vectorize the data
count_vectorizer = CountVectorizer()
X_counts = count_vectorizer.fit_transform(X)
print(X_counts.shape)

# Step 6: Downscale data
tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_counts)
print(X_tfidf.shape)

# Step 7: Shuffle the dataset
df = df.sample(frac=1)

# Step 8: Split the data
train_size = int(0.75 * df.shape[0])
X_train, X_test = X_tfidf[:train_size], X_tfidf[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Step 9: Fit the data into a Naive Bayes classifier
clf = MultinomialNB().fit(X_train, y_train)

# Step 10: Cross validate the model
scores = cross_val_score(clf, X_train, y_train, cv=5)
print("Mean cross-validation accuracy: ", scores.mean())

# Step 11: Test the model
y_pred = clf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred))

# Step 12: Test with new comments
new_comments = [
    "I love this video, it's amazing!",
    "Great content, keep it up!",
    "https://www.youtube.com/watch?v=lBO1L8pgR9s&ab_channel=UnfoldDataScience",
    "Check out this video link: https://www.youtube.com/watch?v=lBO1L8pgR9s&ab_channel=UnfoldDataScience",
    "Win a free iPhone, click here!",
]
new_comments_counts = count_vectorizer.transform(new_comments)
new_comments_tfidf = tfidf_transformer.transform(new_comments_counts)
print(clf.predict(new_comments_tfidf))