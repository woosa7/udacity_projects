from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import datasets

dataset = datasets.load_iris()      # count = 150

features = dataset.data
targets = dataset.target

feature_train, feature_test, target_train, target_test = train_test_split(features, targets, test_size=.3)

model = DecisionTreeClassifier(criterion='gini')
model.fitted = model.fit(feature_train, target_train)
model.predictions = model.fitted.predict(feature_test)

print(confusion_matrix(target_test, model.predictions))
print(accuracy_score(target_test, model.predictions))

predicted = cross_val_predict(model, features, targets, cv=10)
print(accuracy_score(targets, predicted))
