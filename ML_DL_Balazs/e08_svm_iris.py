from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import datasets

"""
	Important parameters for SVC
		gamma -> defines how far the influence of a single training example reaches
				 Low value: influence reaches far      
				 High value: influence reaches close

		C -> trades off hyperplane surface simplicity + training examples mis-classifications
				 Low value: simple/smooth hyperplane surface 
				 High value: all training examples classified correctly but complex surface 
"""

dataset = datasets.load_iris()

features = dataset.data
targets = dataset.target

featureTrain, featureTest, targetTrain, targetTest = train_test_split(features, targets, test_size=0.3)

model = svm.SVC()
# model = svm.SVC(gamma=0.001, C=100)

fittedModel = model.fit(featureTrain, targetTrain)
predictions = fittedModel.predict(featureTest)

print(confusion_matrix(targetTest, predictions))
print(accuracy_score(targetTest, predictions))
