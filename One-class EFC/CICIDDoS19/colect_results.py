from math import sqrt
from statistics import mean, stdev

import numpy as np
import pandas as pd

algorithms = ["GaussianNB", "KNN", "DT", "Adaboost", "RF", "SVC", "MLP", "EFC"]
results = open("Data/Results/Mean_external_internal_results.txt", "w+")
time_results = open("Data/Results/Time_results.txt", "w+")

for j in algorithms:
    precision_internal = []
    recall_internal = []
    f1_internal = []
    roc_internal = []
    precision_external = []
    recall_external = []
    f1_external = []
    roc_external = []
    training_time = []
    testing_time = []
    for i in range(1, 11):
        metrics = np.load(f"Data/Results/Exp{i}/{j}_internal.npy")
        precision_internal.append(metrics[0])
        recall_internal.append(metrics[1])
        f1_internal.append(metrics[2])
        roc_internal.append(metrics[3])

        metrics = np.load(f"Data/Results/Exp{i}/{j}_external.npy")
        precision_external.append(metrics[0])
        recall_external.append(metrics[1])
        f1_external.append(metrics[2])
        roc_external.append(metrics[3])

        cron = np.load(f"Data/Results/Exp{i}/{j}_times.npy")
        training_time.append(cron[0])
        testing_time.append(cron[1])

    results.write(
        "{} & {:.3f} $\\pm$ {:.3f} & {:.3f} $\\pm$ {:.3f} & {:.3f} $\\pm$ {:.3f} & {:.3f} $\\pm$ {:.3f} \\\\ \n".format(
            j,
            mean(f1_internal),
            (stdev(f1_internal) / sqrt(len(f1_internal))) * 1.96,
            mean(roc_internal),
            (stdev(roc_internal) / sqrt(len(f1_internal))) * 1.96,
            mean(f1_external),
            (stdev(f1_external) / sqrt(len(f1_internal))) * 1.96,
            mean(roc_external),
            (stdev(f1_internal) / sqrt(len(roc_external))) * 1.96,
        )
    )
    time_results.write(
        "{} & {:.3f} $\\pm$ {:.3f} & {:.3f} $\\pm$ {:.3f} \\\\ \n".format(
            j,
            mean(training_time),
            (stdev(training_time) / sqrt(len(training_time))) * 1.96,
            mean(testing_time),
            (stdev(testing_time) / sqrt(len(testing_time))) * 1.96,
        )
    )

results.close()
time_results.close()
