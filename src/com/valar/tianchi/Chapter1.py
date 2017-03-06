import numpy as np
import collections
from collections import  defaultdict
from pprint import pprint
from operator import itemgetter
from sklearn.datasets import   load_iris
# Now, we split into a training and test set
from sklearn.cross_validation import train_test_split

dataset_filename = "D:/software/Python/DA/PDF/Code_REWRITE/Chapter 1/affinity_dataset.txt";
x = np.loadtxt(dataset_filename)
n_samples, n_features = x.shape
features = ["bread","milk","cheese","apples","bananas"]
print("n_features:",n_features)
print("n_samples:",n_samples)
print("shape:",x.shape)
num_apple_purchases = 0
for sample in x:
    if sample[3] == 1:
        num_apple_purchases += 1
print(num_apple_purchases)

rule_valid = 0
rule_invalid = 0
valid_rules = defaultdict(int)
invalid_rules = defaultdict(int)
num_occurances = defaultdict(int)


# print(range[4])

for sample in x:
    if sample[3] == 1:
        if sample[4] == 1:
            rule_valid += 1
        else:
            rule_invalid +=1
support = rule_valid
confidence = rule_valid / num_apple_purchases
print("the suppport is:",support,"the confidence is:",confidence)
# import collections
# s = [('yellow', 1), ('blue', 2), ('yellow', 3), ('blue', 4), ('red', 1)]
# d = collections.defaultdict(list)
# for k, v in s:
#     d[k].append(v)
# print(list(d.items()))
#
# dd = collections.defaultdict(int)
# dd.setdefault('valar','yes')
# print(dd.keys())
# print(dd['ggg'])

for sample in  x:
    for premise in range(n_features):
        if sample[premise] == 0:continue
        num_occurances[premise] +=1
        for conclusion in range(n_features):
            if premise == conclusion:
                continue
            if sample[conclusion] == 1:
                valid_rules[(premise,conclusion)] += 1
            else:
                invalid_rules[(premise,conclusion)] += 1
support = valid_rules;
confidence = defaultdict(float)
for premise,conclusion in valid_rules.keys():
    confidence[(premise,conclusion)] = valid_rules[(premise,conclusion)] / num_occurances[premise]
for premise,conclusion in confidence:
    premise_name = features[premise]
    conclusion_name = features[conclusion]
    print("Rule:if a persion buys {0} they will also buy {1}".format(premise_name,conclusion_name))
    print("The confidence is:",confidence[(premise,conclusion)],"the support is:",support[(premise,conclusion)])


def print_rule(premise,conclusion,support,confidence,features):
    premise_name = features[premise]
    conclusion_name = features[conclusion]
    print("Rule: If a person buys {0} they will also buy {1}".format(premise_name, conclusion_name))
    print(" - Confidence: {0:.3f}".format(confidence[(premise, conclusion)]))
    print(" - Support: {0}".format(support[(premise, conclusion)]))
    print("")


premise = 1
conclusion = 3
print_rule(premise,conclusion,support,confidence,features)



print("ready to print the support items:")
pprint(list(support.items()))

sorted_support =sorted(support.items(),key = itemgetter(1),reverse = True)

print("ready to print the support items by sorted method:")
for index in range(5):
    (premise,conclusion) = sorted_support[index][0]
    print_rule(premise, conclusion, support, confidence, features)


sorted_confidence = sorted(confidence.items(),key = itemgetter(1),reverse = True)
print("ready to print the confidence items by sorted method:")
for index in range(5):
    print("Rule #{0}".format(index + 1))
    print("sorted_confidence:",sorted_confidence[index])
    print("sorted_confidence index,0 :", sorted_confidence[index][0])
    (premise, conclusion) = sorted_confidence[index][0]
    print_rule(premise, conclusion, support, confidence, features)


print("ready to print valar testing:")
# print(confidence.keys())
# print(confidence.values())
# print(confidence[0])
























