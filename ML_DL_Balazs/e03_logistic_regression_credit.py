import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Multinomial Logistic Regression
creditData = pd.read_csv("data/credit_data.csv")

print(creditData.head())
print(creditData.describe())
print('---------------')
print(creditData.corr())

# features = creditData[["income","age","loan"]]
features = creditData[["income","age","loan","LTI"]]
target = creditData.default

feature_train, feature_test, target_train, target_test = train_test_split(features,target, test_size=0.3)

model = LogisticRegression()
model.fit = model.fit(feature_train, target_train)
predictions = model.fit.predict(feature_test)

print('---------------')
print(confusion_matrix(target_test, predictions))
print(accuracy_score(target_test,predictions))