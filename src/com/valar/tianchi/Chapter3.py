import pandas as pd
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
from matplotlib import pyplot as plt
import numpy as np
# import matplotlib.pyplot as pt
from collections import  defaultdict
from sklearn.pipeline import  Pipeline
from sklearn.preprocessing import MinMaxScaler

#due to have no data , can not follow book
file_name = "";
dataset = pd.read_csv(file_name);
print("dataset:",dataset)

