import numpy as np
class NumpyLearn:
    def testNumpy(self):
        a = np.arange(0,15).reshape(3,5)
        print(a)
        print(a.shape)
        print(a.dtype.name)
        print(a.itemsize)
        print(a.size)
        print(type(a))
        b = np.array([1,7,5])
        print(type(b))
        
        print("zeros ndarray:")
        print(np.zeros((3,4)))
        print(np.ones((2,3,4), dtype=np.int16))
        print(np.empty((2,3)))
        
    def testNumpy1(self):
        data = ["hi",'valar']
        arr = np.array(data)
        print(arr)
        print(arr.ndim)
        print(arr.astype(np.string_))
        print(np.empty(8))
        print('empty like:')
        a= [[1,2,3], [4,5,6]];
        print(np.empty_like(a))
        
        a = np.array([1,2,3])
        b = np.array([4,5,6])
        print(a * b)
        
    def testNumpy2(self):
        names = np.array(['Joe','valar','john'])
        print(names[1:3])
#         print(names)    
            
        data = np.empty((3,4)) 
        print('original data!')
        print(data)   
        data = np.array(data);
        arr = data
        print('other deals arr[:2,1:]')
        print(arr[:2,1:])  #shape (2,2)
        print('other deals arr[2,:]')
        print(arr[2,:])  #shape (2,1)
        print('other deals arr[:,:2]')
        print(arr[:,:2]) #shape(1,2)
    def testNumpy3(self):
        data = [1,2,3,4,5,6,7,8,9]
        data = np.array(data).reshape(3,3)
        data[1:1] = 2
        print('original data!')
        print(data)
        print('other deals')    
        print((data == 5) | (data == 6))  
#         print(data == 5 | data == 6) 
        data[data != 5] =1
        print(data)
    def testNumpy4(self):
#         test = np.empty((8,4))
#         for i in range(8):
#             #print(i)
#             test[i] = i*2
#         print(test)
#         print(test[1])
#         print(test[[-1,-2,-3]])
        myarr = np.arange(32).reshape(8,4)
        print(myarr)
        print('just a test')
#         print(myarr[[1,5,7,2],[0,3,1,2]])
#         print(myarr[[1,5,7,2]][:,[0,3,1,2]])
        print(myarr[np.ix_([1,5,7,2],[0,3,1,2])])
        print('happy new year')
        
    def testNumpy5(self):
        myarr = np.arange(32).reshape(8,4)
        myarrT = myarr.T
        print(myarr)
        print(myarrT)    
        print(np.dot(myarr,myarrT))
  
    def testNumpy6(self):
        myarr = np.arange(32).reshape(4,4,2)
        myarrT = myarr.transpose(1,0,2);
        print('myarr')
        print(myarr)
        print('myarrT')
        print(myarrT)
        print("myarr * myarrT")
        print(myarr*myarrT)    
#         print(np.dot(myarr,myarrT))      
        
        
        
        