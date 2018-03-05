#!/usr/bin/python

""" 
    Use a Decision Tree to identify emails from the Enron corpus by author:
    Sara has label 0
    Chris has label 1
"""
    
import sys
sys.path.append("./tools/")
from email_preprocess import preprocess

features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
from sklearn import tree
from sklearn.metrics import accuracy_score

clf = tree.DecisionTreeClassifier()     # 0.990
# clf = tree.DecisionTreeClassifier(min_samples_split=40) # 0.977
clf.fit(features_train, labels_train)

labels_pred = clf.predict(features_test)
acc = accuracy_score(labels_test, labels_pred)
print(acc)

#########################################################
