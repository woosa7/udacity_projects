import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score

data = pd.read_csv("data/credit_data.csv")

# Logistic regression accuracy: 93%
# we do better with knn: 99.1% !!!!!!!!

# data.features = data[["income","age","loan"]]
data.features = data[["income","age","loan", "LTI"]]
data.target = data.default

# Normalization = feature scaling
# 1. min-max normalization
# 2. z-score normalization - PCA에서 주로 사용 : X = (X - mean(X)) / Std(X)
data.features = preprocessing.MinMaxScaler().fit_transform(data.features) #HUGE DIFFERENCE !!!

feature_train, feature_test, target_train, target_test = train_test_split(data.features, data.target, test_size=0.3)

cross_valid_scores = []

# small k : underfitting / large k : overfitting

for k in range(1,50):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, data.features, data.target, cv=10, scoring='accuracy')
    cross_valid_scores.append(scores.mean())
    print('{:03d} : {:.5f}'.format(k, scores.mean()))


print('-----------')
kval = np.argmax(cross_valid_scores)+1
print("Optimal k with cross-validation: ", kval)

model = KNeighborsClassifier(n_neighbors=kval)  # k value !!!
fittedModel = model.fit(feature_train, target_train)
predictions = fittedModel.predict(feature_test)

print(confusion_matrix(target_test, predictions))
print(accuracy_score(target_test, predictions))
