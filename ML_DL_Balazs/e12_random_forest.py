from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import datasets
import pandas as pd

#------------------------------------------------------------------------------------------
# 1. iris dataset

dataset = datasets.load_iris()

features = dataset.data
targets = dataset.target

feature_train, feature_test, target_train, target_test = train_test_split(features, targets, test_size=.3)

model = RandomForestClassifier(n_estimators=1000, max_features='sqrt')
fitted_model = model.fit(feature_train, target_train)
predictions = fitted_model.predict(feature_test)

print(confusion_matrix(target_test, predictions))
print(accuracy_score(target_test, predictions))
print('\n')

#------------------------------------------------------------------------------------------
# credit data

# Logistic regression accuracy: 93%
# we do better with knn: 97.5% !!!!!!!!
# we can achieve ~ 99% with random forests

credit_data = pd.read_csv("data/credit_data.csv")

features = credit_data[["income","age","loan"]]
targets = credit_data.default

feature_train, feature_test, target_train, target_test = train_test_split(features, targets, test_size=.2)

model = RandomForestClassifier(n_estimators=1000, max_features='sqrt')
fitted_model = model.fit(feature_train, target_train)
predictions = fitted_model.predict(feature_test)

print(confusion_matrix(target_test, predictions))
print(accuracy_score(target_test, predictions))
