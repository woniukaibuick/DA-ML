import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tests.test_simplification import nan
from numpy.core.defchararray import index
class PandasLearn:
    
    def createSeriesByList(self,list = None):
        if list is None:
            s = pd.Series(list);
            print(s)
        else:
            s = pd.Series([1,2,3,nan,4]);
            print(s)  
    def funcTest(self):
        s = pd.date_range('20170110',periods=6)
        df = pd.DataFrame(np.random.randn(6,4),index =s,columns=list('ABCD'));
        print(df)        
        