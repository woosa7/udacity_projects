#!/usr/bin/python

"""
    Draws a little scatterplot of the training/testing data
    You fill in the regression code where indicated:
"""    

import sys
import pickle
import numpy as np

sys.path.append("tools/")
from feature_format import featureFormat, targetFeatureSplit
dictionary = pickle.load( open("final_project/final_project_dataset_modified.pkl", "rb") )
# print(dictionary.keys())    # name list

### list the features you want to look at -- first item in the list will be the "target" feature
# features_list = ["bonus", "salary"]
features_list = ["bonus", "long_term_incentive"]
data = featureFormat( dictionary, features_list, remove_any_zeroes=True)
target, features = targetFeatureSplit( data )

### training-testing split
from sklearn.model_selection import train_test_split
feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5, random_state=42)
train_color = "b"
test_color = "r"


from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

reg = linear_model.LinearRegression()
reg.fit(feature_train, target_train)

target_test = np.array(target_test)
print(target_test.shape)

y_pred = reg.predict(feature_test)
print(y_pred.shape)

print("Mean squared error: %.2f" % mean_squared_error(target_test, y_pred))
print('Variance score: %.2f' % r2_score(target_test, y_pred))


### draw the scatterplot, with color-coded training and testing points
import matplotlib.pyplot as plt
for feature, target in zip(feature_test, target_test):
    plt.scatter( feature, target, color=test_color )
for feature, target in zip(feature_train, target_train):
    plt.scatter( feature, target, color=train_color )

plt.scatter(feature_test[0], target_test[0], color=test_color, label="test")
plt.scatter(feature_test[0], target_test[0], color=train_color, label="train")

### draw the regression line, once it's coded
try:
    plt.plot( feature_test, reg.predict(feature_test) )
except NameError:
    pass
plt.xlabel(features_list[1])
plt.ylabel(features_list[0])
plt.legend()
plt.show()
