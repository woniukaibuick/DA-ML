import os
import  sys
import  pandas as pd
from collections import  defaultdict
from operator import  itemgetter

#APriori
## improve with FP-Growth

ratting_file_data = "D:/software/Python/data/ml-1m/ratings.dat" #UserID::MovieID::Rating::Timestamp
movie_file_data = "D:/software/Python/data/ml-1m/movies.dat"  #MovieID::Title::Genres

movie_name_data = pd.read_csv(ratting_file_data,delimiter="::",
                          header=None,names={"MovieID","Title","Genres"},engine='python')
all_ratings = pd.read_csv(ratting_file_data,delimiter="::",
                          header=None,names={"UserID","MovieID","Ratting","Datetime"},engine='python')
# print(all_ratings)
# print(all_ratings["Datetime"])
all_ratings["Datetime"] = pd.to_datetime(all_ratings["Datetime"],unit='s')
# print(all_ratings[:5])

all_ratings["Favorable"] = all_ratings["Ratting"]>3
# print(all_ratings[10:15])

rattings = all_ratings[all_ratings["UserID"].isin(range(200))]
print("rattings:")
print(rattings)

favorable_ratings = rattings[rattings["Favorable"]] # only collect the movie favorable field is true

favorable_reviews_by_users = dict((k,frozenset(v.values))
                                   for k,v in favorable_ratings.groupby ("UserID")["MovieID"]
                                  )
num_favorable_by_movie = rattings[["MovieID","Favorable"]].groupby("MovieID").sum()

# res = num_favorable_by_movie.sort("Favorable",ascending = False)[:5]
res = num_favorable_by_movie.sort_values("Favorable",ascending = False)[:5]
print("result:",res)

frequent_itemsets = {}
min_support = 50
frequent_itemsets[1] = dict((frozenset((movie_id,)),row["Favorable"])
                             for movie_id,row in num_favorable_by_movie.iterrows()
                             if row["Favorable"] > min_support)
def findFrequentItemSets(favorable_reviews_by_users,k_1_itemsets,min_support):
    counts = defaultdict(int)
    for user,reviews in favorable_reviews_by_users.items():
        for itemset in k_1_itemsets:
            if itemset.issubset(reviews):
                for other_reviewd_movie in reviews-itemset:
                    current_superset = itemset | frozenset((other_reviewd_movie,))
                    counts[current_superset] += 1
    return dict([(itemset,frequency) for itemset,frequency in counts.items() if frequency>min_support])


for k in range(2,20):
    cur_frequent_itemsets = findFrequentItemSets(favorable_reviews_by_users,frequent_itemsets[k-1],min_support)
    frequent_itemsets[k] = cur_frequent_itemsets
    if len(cur_frequent_itemsets) == 0:
        print("did not find any frequent itemsets of length {}".format(k))
        sys.stdout.flush()
        break
    else:
        print("I found {} frequent itemsets of length{}".format(len(cur_frequent_itemsets), k))
        sys.stdout.flush()
del frequent_itemsets[1]  # delete frequent[1] elements,dirty data

candidate_rules = []
for itemset_length,itemset_counts in frequent_itemsets.items():
    for itemset in itemset_counts.keys():
        for conclusion in itemset:
            premise = itemset - set((conclusion,))
            candidate_rules.append((premise,conclusion))
print("candidate_rules:")
print(candidate_rules[:5])
correct_counts = defaultdict(int)
incorrect_counts = defaultdict(int)
for user, reviews in favorable_reviews_by_users.items():
    for candidate_rule in candidate_rules:
        premise, conclusion = candidate_rule
        if premise.issubset(reviews):
            if conclusion in reviews:
                correct_counts[candidate_rule] += 1
            else:
                incorrect_counts[candidate_rule] += 1
rule_confidence = {candidate_rule: correct_counts[candidate_rule] / float(correct_counts[candidate_rule] + incorrect_counts[candidate_rule])
              for candidate_rule in candidate_rules}


min_confidence = 0.9
sorted_confidence = sorted(rule_confidence.items(), key=itemgetter(1), reverse=True)

# for index in range(5):
#     print("Rule #{0}".format(index + 1))
#     (premise, conclusion) = sorted_confidence[index][0]
#     print("Rule: If a person recommends {0} they will also recommend {1}".format(premise, conclusion))
#     print(" - Confidence: {0:.3f}".format(rule_confidence[(premise, conclusion)]))
#     print("")

def getMovieNameByID(movie_id):
    title_object = movie_name_data[movie_name_data["MovieID"] == movie_id]["Title"]
    title = title_object.values[0]
    return title


for index in range(5):
    print("Rule #{0}".format(index + 1))
    (premise, conclusion) = sorted_confidence[index][0]
    premise_names = ", ".join(getMovieNameByID(idx) for idx in premise)
    conclusion_name = getMovieNameByID(conclusion)
    print("Rule: If a person recommends {0} they will also recommend {1}".format(premise_names, conclusion_name))
    print(" - Confidence: {0:.3f}".format(rule_confidence[(premise, conclusion)]))
    print("")
# Evaluation using test data
test_dataset = all_ratings[~all_ratings['UserID'].isin(range(200))]
test_favorable = test_dataset[test_dataset["Favorable"]]
#test_not_favourable = test_dataset[~test_dataset["Favourable"]]
test_favorable_by_users = dict((k, frozenset(v.values)) for k, v in test_favorable.groupby("UserID")["MovieID"])
#test_not_favourable_by_users = dict((k, frozenset(v.values)) for k, v in test_not_favourable.groupby("UserID")["MovieID"])
#test_users = test_dataset["UserID"].unique()
correct_counts = defaultdict(int)
incorrect_counts = defaultdict(int)
for user, reviews in test_favorable_by_users.items():
    for candidate_rule in candidate_rules:
        premise, conclusion = candidate_rule
        if premise.issubset(reviews):
            if conclusion in reviews:
                correct_counts[candidate_rule] += 1
            else:
                incorrect_counts[candidate_rule] += 1

test_confidence = {candidate_rule: correct_counts[candidate_rule] / float(correct_counts[candidate_rule] + incorrect_counts[candidate_rule])
                   for candidate_rule in rule_confidence}
print(len(test_confidence))
sorted_test_confidence = sorted(test_confidence.items(), key=itemgetter(1), reverse=True)
print(sorted_test_confidence[:5])
for index in range(10):
    print("Rule #{0}".format(index + 1))
    (premise, conclusion) = sorted_confidence[index][0]
    premise_names = ", ".join(getMovieNameByID(idx) for idx in premise)
    conclusion_name = getMovieNameByID(conclusion)
    print("Rule: If a person recommends {0} they will also recommend {1}".format(premise_names, conclusion_name))
    print(" - Train Confidence: {0:.3f}".format(rule_confidence.get((premise, conclusion), -1)))
    print(" - Test Confidence: {0:.3f}".format(test_confidence.get((premise, conclusion), -1)))
    print("")