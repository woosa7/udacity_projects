import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# -------------------------------------------------------------------
# Load data
df = pd.read_csv('data/stock_price.csv')

# create lag data
vclose = df['close']
df['Lag1'] = vclose / vclose.shift(1) - 1
df['Lag2'] = vclose / vclose.shift(2) - 1
df['Lag3'] = vclose / vclose.shift(3) - 1
df['Lag4'] = vclose / vclose.shift(4) - 1
df['Lag5'] = vclose / vclose.shift(5) - 1

df = df.dropna()
df['TLag'] = df['Lag1']+df['Lag2']+df['Lag3']+df['Lag4']+df['Lag5']
# df['direction'] = np.sign(df['TLag'])
df['direction'] = np.sign(df['Lag1'])

df.loc[df.direction == 0, 'direction'] = 1.0

print(df.shape)
print(df.head())

# -------------------------------------------------------------------
# split train / test
X = df[['Lag1','Lag2','Lag3','Lag4','Lag5']]
y = df['direction']

idx = 4595
X_train = X[:idx]
y_train = y[:idx]
X_test = X[idx:]
y_test = y[idx:]

print(X_train.shape)
print(X_test.shape)


# -------------------------------------------------------------------
# Logistic Regression

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print('\n===== Logistic Regression')
print(confusion_matrix(y_test, y_pred))
print(model.score(X_test, y_test))


# -------------------------------------------------------------------
# KNN

model_knn = KNeighborsClassifier(100)
model_knn.fit(X_train, y_train)

y_pred = model_knn.predict(X_test)

print('\n===== KNeighborsClassifier')
print(confusion_matrix(y_test, y_pred))
print(model_knn.score(X_test, y_test))


# -------------------------------------------------------------------
# SVM

model_svc = SVC(kernel='linear')
model_svc.fit(X_train, y_train)

y_pred = model_svc.predict(X_test)

print('\n===== KNeighborsClassifier')
print(confusion_matrix(y_test, y_pred))
print(model_svc.score(X_test, y_test))

