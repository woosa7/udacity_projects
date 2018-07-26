import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler


def isTasty(quality):
    if quality >= 7:
        return 1
    else:
        return 0


data = pd.read_csv("data/wine.csv", sep=";")

print(data['quality'].describe())
print(data['quality'].value_counts())


features = data[["fixed acidity", "volatile acidity", "citric acid", "residual sugar","chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]]
data['tasty'] = data["quality"].apply(isTasty)
targets = data['tasty']

# print('\n', features.head())
features = MinMaxScaler().fit_transform(features)
feature_train, feature_test, target_train, target_test = train_test_split(features, targets, test_size=.2)


param_dist = {
 'n_estimators': [50,100,200],
 'learning_rate' : [0.01,0.05,0.1,0.3,1],
}

grid_search = GridSearchCV(estimator=AdaBoostClassifier(), param_grid=param_dist, cv=10)
grid_search.fit(feature_train, target_train)

print("Best parameter with Grid Search: ", grid_search.best_params_)

predictions = grid_search.predict(feature_test)

print(confusion_matrix(target_test, predictions))
print(accuracy_score(target_test, predictions))
