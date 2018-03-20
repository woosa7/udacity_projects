#!/usr/bin/python

import random
import numpy
import matplotlib.pyplot as plt
import pickle
import sys

sys.path.append("tools/")
from outlier_cleaner import outlierCleaner


### load data
ages = pickle.load( open("tools/practice_outliers_ages.pkl","rb") )
net_worths = pickle.load( open("tools/practice_outliers_net_worths.pkl","rb") )

ages       = numpy.reshape( numpy.array(ages), (len(ages), 1))
net_worths = numpy.reshape( numpy.array(net_worths), (len(net_worths), 1))

# print(ages.shape)
# print(net_worths.shape)

from sklearn.model_selection import train_test_split
ages_train, ages_test, net_worths_train, net_worths_test = train_test_split(ages, net_worths, test_size=0.1, random_state=42)

# train & predict
from sklearn import linear_model
from sklearn.metrics import r2_score

reg = linear_model.LinearRegression()
reg.fit(ages_train, net_worths_train)

# print(reg.coef_[0])
# print(reg.intercept_)

y_pred = reg.predict(ages_test)

print(r2_score(net_worths_test, y_pred))


# plot of all data : line - predict / dot - real value
# try:
#     plt.plot(ages, reg.predict(ages), color="blue")
# except NameError:
#     pass
# plt.scatter(ages, net_worths)
# plt.show()


### identify and remove the most outlier-y points
cleaned_data = []
try:
    predictions = reg.predict(ages_train)
    c_ages, c_net_worths = outlierCleaner( predictions, ages_train, net_worths_train )
except NameError:
    print("your regression object doesn't exist, or isn't name reg")
    print("can't make predictions to use in identifying outliers")


### only run this code if cleaned_data is returning data
if len(c_ages) > 0:
    ages       = numpy.reshape( numpy.array(c_ages), (len(c_ages), 1))
    net_worths = numpy.reshape( numpy.array(c_net_worths), (len(c_net_worths), 1))

    ### refit your cleaned data!
    try:
        reg.fit(ages, net_worths)

        y_pred = reg.predict(ages_test)
        print(r2_score(net_worths_test, y_pred))

        plt.plot(ages, reg.predict(ages), color="blue")
    except NameError:
        print("you don't seem to have regression imported/created,")
        print("   or else your regression object isn't named reg")
        print("   either way, only draw the scatter plot of the cleaned data")
    plt.scatter(ages, net_worths)
    plt.xlabel("ages")
    plt.ylabel("net worths")
    plt.show()
else:
    print("outlierCleaner() is returning an empty list, no refitting to be done")
