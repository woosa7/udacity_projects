#!/usr/bin/python 

""" 
    Skeleton code for k-means clustering mini-project.
"""
import pickle
import numpy
import matplotlib.pyplot as plt
import sys
sys.path.append("./tools/")
from feature_format import featureFormat, targetFeatureSplit


def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    colors = ["b", "c", "y", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()


# -----------------------------------------------------------------------------------------------------
### load Enron data
data_dict = pickle.load( open("final_project/final_project_dataset.pkl", "rb") )

### outlier -- remove it!
data_dict.pop("TOTAL", 0)

print(data_dict['METTS MARK'].keys())

feature_1 = "salary"
feature_2 = "exercised_stock_options"       # salary, exercised_stock_options, total_payments, bonus, long_term_incentive
feature_3 = "total_payments"
feature_4 = "bonus"
poi  = "poi"

features_list = [poi, feature_1, feature_2]
# features_list = [poi, feature_1, feature_2, feature_3]
# features_list = [poi, feature_1, feature_2, feature_3, feature_4]

data = featureFormat(data_dict, features_list, remove_NaN=True, remove_all_zeroes=True)
poi, finance_features = targetFeatureSplit( data )

# scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
finance_features = scaler.fit_transform(finance_features)


### cluster
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=0).fit(finance_features)
pred = kmeans.labels_

try:
    Draw(pred, finance_features, poi, mark_poi=False, name="clusters.pdf", f1_name=feature_1, f2_name=feature_2)
except NameError:
    print("no predictions object named pred found, no clusters to plot")
