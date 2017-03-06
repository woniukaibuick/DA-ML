from sklearn import datasets
# from skimage import io
# iris = datasets.load_iris()
# print(iris);
# print(datasets.load_boston())

import numpy as np
import urllib.request

from sklearn import  preprocessing
from sklearn import  metrics
from sklearn.ensemble import  ExtraTreesClassifier
# url with dataset
url="http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
# download the file
raw_data = urllib.request.urlopen(url)
# load the CSV file as a numpy matrix
dataset = np.loadtxt(raw_data,delimiter=",")
# separate the data from the target attributes
X = dataset[:,0:8]
Y = dataset[:,8]
normalized_X = preprocessing.normalize(X)
standardized_X = preprocessing.scale(X)
model = ExtraTreesClassifier()
model.fit(X,Y)
print(model.feature_importances_)