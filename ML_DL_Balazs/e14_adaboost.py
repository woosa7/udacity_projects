from sklearn import datasets
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

dataset = datasets.load_iris()

features = dataset.data
targets = dataset.target

feature_train, feature_test, target_train, target_test = train_test_split(features, targets, test_size=.2)

model = AdaBoostClassifier(n_estimators=100, learning_rate=1, random_state=123)
model.fitted = model.fit(feature_train, target_train)
model.predictions = model.fitted.predict(feature_test)

print(confusion_matrix(target_test, model.predictions))
print(accuracy_score(target_test, model.predictions))
