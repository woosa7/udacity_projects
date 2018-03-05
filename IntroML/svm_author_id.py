#!/usr/bin/python

""" 
    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
sys.path.append("./tools/")
from email_preprocess import preprocess

features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
from sklearn import svm
from sklearn.metrics import accuracy_score

# slen = int(len(features_train)/10)
# features_train = features_train[:slen]
# labels_train = labels_train[:slen]
# print(len(labels_train))

# clf = svm.SVC(C=1.0, kernel='linear')     # 0.984
clf = svm.SVC(C=10000.0, kernel='rbf')      # 0.990
clf.fit(features_train, labels_train)

labels_pred = clf.predict(features_test)
acc = accuracy_score(labels_test, labels_pred)
print(acc)

#########################################################
