#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot as plt
sys.path.append("./tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("./final_project/final_project_dataset.pkl", "rb") )


# all data - some outliers
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()


# remove outlers
keys = data_dict.keys()

out_keys = []
for key in keys:
    bonus = data_dict[key]['bonus']
    if  bonus == "NaN":
        bonus = 0
    else:
        bonus = int(bonus)

    if bonus > 5000000:
        print(key, bonus)
        out_keys.append(key)

print(out_keys)

for key in out_keys:
    del data_dict[key]


# no outliers
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()
