import os
import  pandas as pd


ratting_file_data = "D:/software/Python/Movie data/ml-1m/ratings.dat"
movie_file_data = "D:/software/Python/Movie data/ml-1m/movies.dat"

all_ratings = pd.read_csv(ratting_file_data,delimiter="\t",
                          header=None,names={"UserID","MovieID","Ratting","Datetime"})

all_ratings["Datetime"] = pd.to_datetime(all_ratings["Datetime"],unit='s')
print(all_ratings[:5])