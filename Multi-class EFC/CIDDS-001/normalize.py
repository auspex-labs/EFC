import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import math
import os
import random
import sys

def get_normalization(file):
    intervals = []
    for feature in range(8):
        print(feature)
        data = pd.read_csv(file, usecols = [feature], header=None)
        data = list(data.iloc[:,0])
        intervals.append([min(data), max(data)])
        print(intervals[feature])
    return intervals

def normalize(data, dict):
    for feature in range(8):
        if feature not in [1,6]:
            min, max = dict[feature]
            if min != max:
                data.iloc[:, feature] = (np.array(data.iloc[:,feature]) - min) / (max - min)
                print(np.unique(data.iloc[:, feature]))
    return data


for fold in range(1,6):
    intervals = get_normalization("5-fold_sets/Non_discretized/Sets{}/reduced_train.csv".format(fold))
    np.save("5-fold_sets/Normalized/Sets{}/Min_max.npy".format(fold), intervals)

    # intervals = np.load("5-fold_sets/Normalized/Sets{}/Min_max.npy".format(fold), allow_pickle=True)
    reader = pd.read_csv("5-fold_sets/Non_discretized/Sets{}/reduced_train.csv".format(fold), chunksize=7000000, header=None)
    for chunk in reader:
        data = normalize(chunk, intervals)
        data.to_csv("5-fold_sets/Normalized/Sets{}/reduced_train.csv".format(fold), mode='a', header=False, index=False)

    reader = pd.read_csv("5-fold_sets/Non_discretized/Sets{}/test.csv".format(fold), chunksize=7000000, header=None)
    for chunk in reader:
        data = normalize(chunk, intervals)
        data.to_csv("5-fold_sets/Normalized/Sets{}/test.csv".format(fold), mode='a', header=False, index=False)