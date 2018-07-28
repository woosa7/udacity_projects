import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import datasets

dataset = datasets.load_iris()      # count = 150

features = dataset.data
targets = dataset.target

# with grid search you can find an optimal parameter "parameter tuning" !!!
param_grid = {'max_depth': np.arange(1, 10)}

# In every iteration, data split randomly in cross validation + DecisionTreeClassifier
# initializes the tree randomly: that's why you get different results !!!
tree = GridSearchCV(DecisionTreeClassifier(), param_grid)

feature_train, feature_test, target_train, target_test = train_test_split(features, targets, test_size=.3)

tree.fit(feature_train, target_train)
tree_predictions = tree.predict_proba(feature_test)[:, 1]

print("Best parameter with Grid Search: ", tree.best_params_)

# --------------------------
param = tree.best_params_['max_depth']

model = DecisionTreeClassifier(max_depth=param)
model.fitted = model.fit(feature_train, target_train)
model.predictions = model.fitted.predict(feature_test)

print(confusion_matrix(target_test, model.predictions))
print(accuracy_score(target_test, model.predictions))
