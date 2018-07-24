import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score

creditData = pd.read_csv("data/credit_data.csv")

features = creditData[["income","age","loan"]]
target = creditData.default

model = LogisticRegression()
predicted = cross_val_predict(model,features,target, cv=10)
print(accuracy_score(target, predicted))

scores = cross_val_score(model, features, target, cv=10)
print(scores)
print(np.mean(scores))


