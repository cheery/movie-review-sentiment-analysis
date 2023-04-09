import os
import glob
import itertools
from nltk_preprocess import preprocess_text

train_neg_files = glob.glob("aclImdb/train/neg/*.txt")
train_pos_files = glob.glob("aclImdb/train/pos/*.txt")
test_neg_files = glob.glob("aclImdb/test/neg/*.txt")
test_pos_files = glob.glob("aclImdb/test/pos/*.txt")

def read_file(filename):
    with open(filename, 'r', encoding="utf-8") as fd:
        return fd.read()

print("Preprocessing reviews")
train_reviews = []
test_reviews = []

for text in map(preprocess_text,
                map(read_file, train_neg_files)):
    train_reviews.append((text, "neg"))

for text in map(preprocess_text,
                map(read_file, train_pos_files)):
    train_reviews.append((text, "pos"))

for text in map(preprocess_text,
                map(read_file, test_neg_files)):
    test_reviews.append((text, "neg"))

for text in map(preprocess_text,
                map(read_file, test_pos_files)):
    test_reviews.append((text, "pos"))

import random
random.shuffle(train_reviews)
random.shuffle(test_reviews)

fe = "tfid_vectorizer"

print("Fitting vectorizer")
if fe == "bag_of_words":
    from sklearn.feature_extraction.text import CountVectorizer

    count_vectorizer = CountVectorizer()
    #bag_of_words = count_vectorizer.fit_transform(all_reviews)
    #vector = bag_of_words.toarray()

    count_vectorizer.fit([r[0] for r in train_reviews])

    #print("CountVectorizer Vocabulary:")
    #print(count_vectorizer.vocabulary_)

    def vectorize(review):
        # Transform the preprocessed review
        bag_of_words = count_vectorizer.transform([review])

        # Convert the matrix to an array
        return bag_of_words.toarray()[0]
elif fe == "tfid_vectorizer":
    from sklearn.feature_extraction.text import TfidfVectorizer

    tfid_vectorizer = TfidfVectorizer()
    tfid_vectorizer.fit([r[0] for r in train_reviews])

    def vectorize(review):
        # Transform the preprocessed review
        tfidf_matrix = tfid_vectorizer.transform([review])

        # Convert the matrix to an array
        return tfidf_matrix.toarray()[0]
elif fe == "word_embedding":
    import gensim.downloader as api
    # Load pre-trained Word2Vec model
    model = api.load('word2vec-google-news-300') 

    def vectorize(preprocessed_review):
        # Tokenize the preprocessed review
        tokenized_review = preprocessed_review.split()

        # Calculate the average word embeddings for the review
        review_embedding = sum([model[word] for word in tokenized_review if word in model]) / len(tokenized_review)

        return review_embedding


# Linear regression (didn't work)
print("Training linear regression with training data")
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# Create a logistic regression model with stochastic gradient descent (SGD) optimization
lr_model = SGDClassifier(loss='log')

# Define the batch size
batch_size = 1000

y_train = [r[1] for r in train_reviews]

for i in range(0, len(train_reviews), batch_size):
    print("batch:", i)
    X_train_batch = []
    y_train_batch = []
    for review, sentiment in train_reviews[i : i + batch_size]:
        X_train_batch.append(vectorize(review))
        y_train_batch.append(sentiment)
    lr_model.partial_fit(X_train_batch, y_train_batch, classes=np.unique(y_train))

print("Testing the model")
accuracies = []
precisions = []
recalls = []
f1_scores = []

for i in range(0, len(test_reviews), batch_size):
    print("batch:", i)
    X_batch = []
    y_batch = []
    for review, sentiment in test_reviews[i : i + batch_size]:
        X_batch.append(vectorize(review))
        y_batch.append(sentiment)
    y_pred = lr_model.predict(X_batch)
    accuracy = accuracy_score(y_batch, y_pred)
    accuracies.append(accuracy)

    precision, recall, f1_score, _ = precision_recall_fscore_support(y_batch, y_pred, average='binary', pos_label='pos')
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1_score)

# Calculate the overall metrics by averaging the metrics of each batch
overall_accuracy = np.mean(accuracies)
overall_precision = np.mean(precisions)
overall_recall = np.mean(recalls)
overall_f1_score = np.mean(f1_scores)

print("Overall accuracy:", overall_accuracy)
print("Overall precision:", overall_precision)
print("Overall recall:", overall_recall)
print("Overall F1-score:", overall_f1_score)


print("Training naive bayes with training data")
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

clf = MultinomialNB()

# Define the batch size
batch_size = 1000
 
y_train = [r[1] for r in train_reviews]

for i in range(0, len(train_reviews), batch_size):
    print("batch:", i)
    X_train_batch = []
    y_train_batch = []
    for review, sentiment in train_reviews[i : i + batch_size]:
        X_train_batch.append(vectorize(review))
        y_train_batch.append(sentiment)
    clf.partial_fit(X_train_batch, y_train_batch, classes=np.unique(y_train))

print("Testing the model")
accuracies = []
precisions = []
recalls = []
f1_scores = []

for i in range(0, len(test_reviews), batch_size):
    print("batch:", i)
    X_batch = []
    y_batch = []
    for review, sentiment in test_reviews[i : i + batch_size]:
        X_batch.append(vectorize(review))
        y_batch.append(sentiment)
    y_pred = clf.predict(X_batch)
    accuracy = accuracy_score(y_batch, y_pred)
    accuracies.append(accuracy)

    precision, recall, f1_score, _ = precision_recall_fscore_support(y_batch, y_pred, average='binary', pos_label='pos')
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1_score)

# Calculate the overall metrics by averaging the metrics of each batch
overall_accuracy = np.mean(accuracies)
overall_precision = np.mean(precisions)
overall_recall = np.mean(recalls)
overall_f1_score = np.mean(f1_scores)

print("Overall accuracy:", overall_accuracy)
print("Overall precision:", overall_precision)
print("Overall recall:", overall_recall)
print("Overall F1-score:", overall_f1_score)

