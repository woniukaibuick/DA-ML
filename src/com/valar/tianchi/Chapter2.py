import os
import  csv
from sklearn.datasets import   load_iris
import numpy as np
from collections import  defaultdict
from operator import  itemgetter
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.cross_validation import  cross_val_score
from matplotlib import pyplot as pt
import numpy as np
# import matplotlib.pyplot as pt
from collections import  defaultdict
from sklearn.pipeline import  Pipeline
from sklearn.preprocessing import MinMaxScaler


x = np.arange(0 , 360)
y = np.sin( x * np.pi / 180.0)
pt.plot(x,y)
pt.xlim(0,360)
pt.ylim(-1.2,1.2)
pt.title("SIN function")
pt.show()
pt.clf()

# home_folder = os.path.expanduser("~")
# print("home_folder",home_folder)

# data_folder = os.path.join(home_folder,"Data","Ionosphere")
data_folder = "D:/software/Python/DA/PDF/Code_REWRITE/Chapter 2/";
data_filename = os.path.join(data_folder,"ionosphere.data")
print("data_filename",data_filename)
X = np.zeros((351,34),dtype="float")
y = np.zeros((351,),dtype="bool")
with open(data_filename,"r") as input_file:
    reader = csv.reader(input_file)
    for i,row in enumerate(reader):
        data = [float(datum) for datum in row[:-1]]
        X[i]=data
        y[i] = row[-1] == 'g'
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=14)
print("There are {} samples in the training dataset".format(X_train.shape[0]))
print("There are {} samples in the testing dataset".format(X_test.shape[0]))
print("Each sample has {} features".format(X_train.shape[1]))
estimator = KNeighborsClassifier()
estimator.fit(X_train,y_train)
y_predicted = estimator.predict(X_test)
accuracy = np.mean(y_test == y_predicted) * 100
print("The accuracy is {0:.1f}".format(accuracy))

avg_scores = []
all_scores = []
parameter_values = list(range(1,21))
for n_neighbors in parameter_values:
    estimator =  KNeighborsClassifier(n_neighbors=n_neighbors)
    scores = cross_val_score(estimator,X,y,scoring = 'accuracy')
    avg_scores.append(np.mean(scores))
    all_scores.append(scores)
print("now will plot:")
pt.figure(figsize=(32,20))
pt.plot(parameter_values,avg_scores,"-0",linewidth = 5, markersize = 24)
pt.show()




for parameter ,scores in zip(parameter_values,all_scores):
    n_scores = len(scores)
    pt.plot([parameter]*n_scores,scores,"-o")



pt.plot(parameter_values,all_scores,"bx")
pt.show()

all_scores = defaultdict(list)
parameter_values = list(range(1,21))
for n_neighbors in parameter_values:
    for i in range(100):
        estimator = KNeighborsClassifier(n_neighbors=n_neighbors)
        scores = cross_val_score(estimator,X,y,scoring="accuracy",cv=10)
        all_scores[n_neighbors].append(scores)
for parameter in parameter_values:
    scores = all_scores[parameter]
    n_scores = len(scores)
    pt.plot([parameter] * n_scores,scores,"-o")

pt.plot(parameter_values, avg_scores, '-o')
pt.show()






X_broken = np.array(X)
# 接下来，我们就要捣乱了，每隔一行，就把第二个特征的值除以10。
X_broken[:,::2] /= 10
X_transformed = MinMaxScaler().fit_transform(X_broken)
estimator = KNeighborsClassifier()
transformed_scores = cross_val_score(estimator, X_transformed, y,
scoring='accuracy')
scaling_pipline = Pipeline([('scale',MinMaxScaler()),('predict',KNeighborsClassifier())])
scores = cross_val_score(scaling_pipline,X,y,scoring='accuracy')
print("The average accuracy for is: {0:.1f}%".format(np.mean(transformed_scores) * 100))

















