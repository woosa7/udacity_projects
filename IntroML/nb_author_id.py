#!/usr/bin/python
"""
    Use a Naive Bayes Classifier to identify emails by their authors
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
sys.path.append("./tools/")
from email_preprocess import preprocess

features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

clf = GaussianNB()
clf.fit(features_train, labels_train)

labels_pred = clf.predict(features_test)
acc = accuracy_score(labels_test, labels_pred)
print(acc)

#########################################################

