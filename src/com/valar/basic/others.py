# print("hi python");
# var1 = 1;
# del var1; 
# print(abs(-100));
# 
# lis = ["one","two"];
# lis.clear();
# print(lis)

# a, b = 0, 1
# while b < 10:
#     print(b)
#     a, b = b, a+b

# import sys
# 
# def fibonacci(n): # 生成器函数 - 斐波那契
#     a, b, counter = 0, 1, 0
#     while True:
#         if (counter > n): 
#             return
#         yield a
#         a, b = b, a + b
#         counter += 1
# f = fibonacci(5) # f 是一个迭代器，由生成器返回生成
# 
# while True:
#     try:
#         print (next(f), end=" ")
#     except StopIteration:
#         sys.exit()
        
        
# def firstFunc(n):
#     print(n)
#     
# firstFunc(10) ;       
# 
# print("below:")
# for i in sys.argv:
#     print(i)
#     
# print('\n\nPython path is:',sys.path,'\n')    


# class people:
#     #定义基本属性
#     name = ''
#     age = 0
#     #定义私有属性,私有属性在类外部无法直接进行访问
#     __weight = 0
#     #定义构造方法
#     def __init__(self,n,a,w):
#         self.name = n
#         self.age = a
#         self.__weight = w
#     def speak(self):
#         print("%s 说: 我 %d 岁。" %(self.name,self.age))
# 
# #单继承示例
# class student(people):
#     grade = ''
#     def __init__(self,n,a,w,g):
#         #调用父类的构函
#         people.__init__(self,n,a,w)
#         self.grade = g
#     #覆写父类的方法
#     def speak(self):
#         print("%s 说: 我 %d 岁了，我在读 %d 年级"%(self.name,self.age,self.grade))
# 
# 
# 
# s = student('ken',10,60,3)
# s.speak()
