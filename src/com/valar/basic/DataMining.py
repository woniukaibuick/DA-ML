import json
from collections import  defaultdict
from test.test_trace import TestLineCounts
from pandas import DataFrame,Series
import pandas as pd; import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
class DataMining:
    __path = 'D:\software\Python\python code\pydata-book-master' ;
    __record = [];
    def __init__(self,path):
        print('path:'+path)
        self.__path = path;
        self.__record = [json.loads(line) for line in open(path)];
        
    def getCounts(self,sequence):
        counts = {}
        for x in sequence:
            if x in counts:
                counts[x] += 1
            else:
                counts[x] = 1
        return counts    
    
    
    def getCountsByPythonLibrary(self,sequence):
        counts = defaultdict(int)
        for x in sequence:
            counts[x] += 1
        return counts
    
    def getTopCounts(self,count_dict,n = 10):
        value_key_pairs = [(count,tz) for tz,count in count_dict.items()]
        value_key_pairs.sort()
        return value_key_pairs[-n:]
    
    def ch02(self):
        print("ch02 starting!")
        print(self.__record[0])
        print("self.__record[0]['tz']:"+self.__record[0]['tz'])
        time_zones = [rec['tz'] for rec in self.__record if 'tz' in rec]
        print("time_zones[:10]")
        print(time_zones[:10])
        counts = self.getCountsByPythonLibrary(time_zones)
        print(counts)
        print(counts["America/New_York"])
        print("len:")
        len(time_zones)
        print("get Top counts:")
        print(self.getTopCounts(counts))
        counts = Counter(time_zones)
        print("get Top Counts By libray Func:")
        counts.most_common(10)
        
    def ch02_frame(self):
        print("ch03 starting")    
        frame = DataFrame(self.__record);
#         print(frame)
        tz_counts = frame['tz'].value_counts();
#         print( tz_counts[:10])
        clean_tz = frame['tz'].fillna('Missing');
        clean_tz[clean_tz == ''] = 'unknown'
        tz_counts = clean_tz.value_counts()
#         print(tz_counts[:10])
#         print(tz_counts[:10].plot(kind='barh',rot=0))
        results = Series([]) 
        
#         print(frame['_heartbeat_'])
        print('operating_system:')
        operating_system = np.where(frame['a'].str.contains('windows'))
        print(operating_system)        
#         print(results)
    def drawPlot(self):
        x = np.linspace(0, 10, 1000)
        y = np.sin(x)
        z = np.cos(x**2)
         
        plt.figure(figsize=(10,6))
        plt.plot(x,y,label="$sin(x)$",color="red",linewidth=2)
        plt.plot(x,z,"b--",label="$cos(x^2)$")
        plt.xlabel("Time(s)")
        plt.ylabel("Volt")
        plt.title("PyPlot First Example")
        plt.ylim(-1.2,1.2)
        plt.legend()
        plt.show()
    def plotTest(self):
        plt.figure()
        plt.xlabel('xLabel')
        plt.ylabel('yLabel')
        plt.title('title')
        x=[1,2,3]
        y=[4,5,6]
        plt.plot(x,y)
        plt.show()
        
        