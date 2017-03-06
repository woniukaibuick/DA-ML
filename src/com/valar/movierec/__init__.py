import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from traitsui.editors.list_editor import columns_trait
class SimpleMovieRecommendation:
    def loadMatrix(self):
        matrix = {}
        f = open('data.txt')
        columns = f.readline().split(',')
        for line in f:
            scores = line.split(',')
            print(scores)
            for i in range(len(scores))[1:] :
                matrix[(scores[0], columns[i])] = scores[i].strip("\n")
        return matrix
    def simDistance(self, matrix, row1, row2):        
        columns = set(lambda l:l[1], matrix.keys())
        si = filter(lambda l:matrix.has_key(row1, 1) and matrix[(row1, 1)] != "" and matrix.has_key((row2, 1)) and matrix[(row2, 1)] != "", columns)
        if len(si) == 0 : return 0
        sum_of_distance = sum([pow(float(matrix[(row1, column)]) - float(matrix[(row2, column)]), 2)
                               for column in si])
        return 1 / (1 + np.sqrt(sum_of_distance))  # printsim_distance(matrix, "KaiZhou", "ShuaiGe")
        
           
    def top_matches(self, matrix, row, similarity=simDistance):
        rows = set(map(lambda l:l[0], matrix.keys()))
        scores = [(similarity(matrix, row, r), r)for r in rows if r != row]
        scores.sort()
        scores.reverse()
        return scores    
        
    def transform(self, matrix):
        rows = set(map(lambda l:l[0], matrix.keys()))
        columns = set(map(lambda l:l[1], matrix.keys()))
        transform_matrix = {}
        for row in rows:
            for column in columns:
                transform_matrix[(column, row)] = matrix[(row, column)]
        return transform_matrix     
    def get_recommendations(self, matrix, row, similarity=simDistance):
        rows = set(map(lambda l:l[0], matrix.keys()))
        columns = set(map(lambda l:l[1], matrix.keys()))
        sum_of_column_sim = {}
        sum_of_column = {}
        for r in rows:
            if r == row:
                continue
            sim = similarity(matrix, row, r)
            if sim <= 0:
                continue 
            for c in columns:
                if matrix[(r, c)] == "":
                    continue
        sum_of_column_sim.setdefault(c, 0)
        sum_of_column_sim[c] += sim
        sum_of_column.setdefault(c, 0)
        sum_of_column[c] += float(matrix[(r, c)]) * sim
        scores = [(sum_of_column[c] / sum_of_column_sim[c], c)for c in sum_of_column]
        scores.sort()
        scores.reverse()
        return scores   
              
              
              
                               
# mr = SimpleMovieRecommendation();
# matrix = mr.loadMatrix()
# mr.simDistance(matrix, 'KaiZhou', 'ShuaiGe');
# tran_matrix = mr.transform(matrix)
# film = 'Friends'
# print(mr.top_matches(tran_matrix, film))
# print(mr.loadMatrix())       
