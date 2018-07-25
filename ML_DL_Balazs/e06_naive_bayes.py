import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

data = pd.read_csv("data/credit_data.csv")

# Logistic regression accuracy: 93%
# we do better with knn: 97.5% !!!!!!!!

# data.features = data[["income","age","loan"]]
data.features = data[["income","age","loan","LTI"]]
data.target = data.default

data.features = preprocessing.MinMaxScaler().fit_transform(data.features)

feature_train, feature_test, target_train, target_test = train_test_split(data.features,data.target, test_size=0.3)

model = GaussianNB()  
fittedModel = model.fit(feature_train, target_train)
predictions = fittedModel.predict(feature_test)

print(confusion_matrix(target_test, predictions))
print(accuracy_score(target_test,predictions))