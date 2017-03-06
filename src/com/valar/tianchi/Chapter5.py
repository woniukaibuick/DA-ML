import  os
import  pandas as pd
import numpy as np
from sklearn.feature_selection import  VarianceThreshold
adult_data_file  = "D:/software/Python/data/chapter5/adult.data"

adult = pd.read_csv(adult_data_file,header=None,
                         names=["Age", "Work-Class", "fnlwgt",
                        "Education", "Education-Num",
                        "Marital-Status", "Occupation",
                        "Relationship", "Race", "Sex",
                        "Capital-gain", "Capital-loss",
                        "Hours-per-week", "Native-Country",
                        "Earnings-Raw"])
# print(adult_data)
adult.dropna(how='all',inplace=True)
print(adult.columns)
describe = adult["Hours-per-week"].describe()
print(describe)

print(adult["Education-Num"].median())

print(adult["Work-Class"].unique())

adult["LongHours"] = adult["Hours-per-week"] > 40

x = np.arange(30).reshape((10,3))
print(x)
x[:,1] = 1
print(x)

vt = VarianceThreshold()
xt = vt.fit_transform(x)
print(xt)
print(vt.variances_)