from sklearn.datasets import   load_iris
import numpy as np
from collections import  defaultdict
from operator import  itemgetter
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report


dataset = load_iris()
X = dataset.data
y = dataset.target
print("data:",X)
# print(dataset.DESCR)
print("X.shape",X.shape)
n_samples, n_features = X.shape
attribute_means = X.mean(axis=0)
print("attribute_means",attribute_means)
assert attribute_means.shape == (n_features,)
X_d = np.array(X >= attribute_means, dtype='int')
print("X_d:",X_d)


# Set the random state to the same number to get the same results as in the book
random_state = 14

X_train, X_test, y_train, y_test = train_test_split(X_d, y, random_state=random_state)
print("There are {} y training samples".format(y_train.shape))
print("There are {} y testing samples".format(y_test.shape))

print("There are {} x training samples".format(X_train.shape))
print("There are {} x testing samples".format(X_test.shape))


def train(X,y_true,feature):
    n_samples,n_features = X.shape
    assert  0<=feature<n_features
    values = set(X[:,feature])
    predictors = dict()
    errors = []
    for current_value in values:
        most_frequent_class,error = train_feature_value(X,y_true,feature,current_value)
        predictors[current_value] = most_frequent_class
        errors.append(error)
    total_error = sum(errors)
    return predictors,total_error
def train_feature_value(X,y_true,feature,value):
    class_counts = defaultdict(int)
    for sample,y in zip(X,y_true):
        if sample[feature] == value:
            class_counts[y] += 1
    sorted_class_counts = sorted(class_counts.items(),key=itemgetter(1),reverse = True)
    most_frequent_class = sorted_class_counts[0][0]
    n_samples = X.shape[1]
    error = sum([class_count for class_value,class_count in class_counts.items()
                 if class_value != most_frequent_class])
    return most_frequent_class,error

# all_predictors = {variable: train(X_train, y_train, variable) for variable in range(X_train.shape[1])}
all_predictors = {variable: train(X_train, y_train, variable) for variable in range(X_train.shape[1])}
errors = {variable:error for variable , (mapping,error) in all_predictors.items()}
best_variable ,best_error = sorted(errors.items(),key= itemgetter(1))[0]
print("the best model is based on variable {0} ans has error {1:2f}".format(best_variable,best_error))
model = {"variable":best_variable,"predictor":all_predictors[best_variable][0]}
print("the model is:",model)

def predict(X_test,model):
    variable = model['variable']
    predictor = model['predictor']
    y_predicted = np.array([predictor[int(sample[variable])] for sample in X_test])
    return y_predicted

y_predicted = predict(X_test,model)
print("y_predicted:",y_predicted)
accuracy = np.mean(y_predicted == y_test) * 100
print("the test accuracy is {:.1f}%".format(accuracy))

print("classification_report:",(y_test,y_predicted))








