#!/usr/bin/python

import pickle
import numpy
numpy.random.seed(42)


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10) mini-project.
words_file = "../text_learning/your_word_data.pkl"
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "rb"))
authors = pickle.load( open(authors_file, "rb"))


from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(word_data, authors, test_size=0.1, random_state=42)


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test)


### a classic way to overfit is to use a small number of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150]
labels_train   = labels_train[:150]


from sklearn import tree, naive_bayes
from sklearn.metrics import accuracy_score

# clf = naive_bayes.GaussianNB()
clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)

labels_pred = clf.predict(features_test)
acc = accuracy_score(labels_test, labels_pred)
print(acc)

# Most important word --> text_learning/vectorize_text에서 highly powerful word 제거
for ind, imp in enumerate(clf.feature_importances_):
    if imp > 0.2:
        print('Important word - pos: {}, imp: {}, word : {}'.format(ind, imp, vectorizer.get_feature_names()[ind]))
