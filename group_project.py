import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Step 1: Loading the data
df = pd.read_csv('Youtube01-Psy.csv')

# Step 3: Basic data exploration
print(df.head())
print(df.describe())


# Step 4: Preparing the data
X = df['CONTENT']
y = df['CLASS']

# Tokenize and remove stopwords
stop_words = set(stopwords.words('english'))
X = X.apply(lambda x: ' '.join([word for word in word_tokenize(x) if word not in stop_words]))

# Step 5: Vectorizing  the data
count_vectorizer = CountVectorizer()
X_count_vectorized = count_vectorizer.fit_transform(X)
print(X_count_vectorized.shape)

# Step 6: Downscaling the  data
tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_count_vectorized)
print(X_tfidf.shape)

# Step 7: Shuffle the dataset
# Does not make a difference in this case 
df = df.sample(frac=1)

# Step 8: Spliting  the data into training and testing sets
train_size = int(0.75 * df.shape[0])
X_train, X_test = X_tfidf[:train_size], X_tfidf[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Step 9: Fitting the data into a Naive Bayes classifier
classifier = MultinomialNB().fit(X_train, y_train)

# Step 10: Cross validating the model
scores = cross_val_score(classifier, X_train, y_train, cv=5)
print("Mean cross-validation accuracy: ", scores.mean())

# Step 11: Testing  the model
# Printing the confusion matrix and accuracy score
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred))

# Step 12: Testing with new comments created by the team
new_comments = [
    "The animation on season 2 of Jujutsu kaisen is trash!",
    "Mr beast is giving away free money on telegram on this link: https://t.me/MrBeast1000000",
    "The movies are so boring these days...They just create same old stories again and again.",
    "Check out this video link: https://www.youtube.com/watch?v=lBO1L8pgR9s&ab_channel=UnfoldDataScience",
    "Win a free iPhone 15 pro, click here!",
]
# Preprocess the new comments
new_comments_counts = count_vectorizer.transform(new_comments)
new_comments_tfidf = tfidf_transformer.transform(new_comments_counts)
# Predict the new comments
print(classifier.predict(new_comments_tfidf))
