import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from classification_functions import *
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

sys.path.append("../../../EFC")


def DT(sets):
    train = pd.read_csv(f"5-fold_sets/Normalized/Sets{sets}/X_train", header=None)
    train_labels = pd.read_csv(f"5-fold_sets/Normalized/Sets{sets}/y_train", header=None)
    test = pd.read_csv(f"5-fold_sets/Normalized/Sets{sets}/X_test", header=None)

    DT = DecisionTreeClassifier()
    start = time.time()
    DT.fit(train, train_labels)
    training_time = time.time() - start
    print("DT train: ", time.time() - start)
    start = time.time()
    predict_labels = DT.predict(test)
    testing_time = time.time() - start
    print("DT test: ", time.time() - start)
    np.save(f"5-fold_sets/Results/Sets{sets}/DT_predicted.npy", predict_labels)
    np.save(
        f"5-fold_sets/Results/Sets{sets}/DT_times.npy",
        [training_time, testing_time],
    )


def svc(sets):
    train = pd.read_csv(f"5-fold_sets/Normalized/Sets{sets}/X_train", header=None)
    train_labels = pd.read_csv(f"5-fold_sets/Normalized/Sets{sets}/y_train", header=None)
    test = pd.read_csv(f"5-fold_sets/Normalized/Sets{sets}/X_test", header=None)

    svc = SVC(kernel="poly", probability=True)
    start = time.time()
    svc.fit(train, train_labels)
    training_time = time.time() - start
    print("SVC train: ", training_time)
    start = time.time()
    predict_labels = svc.predict(test)
    testing_time = time.time() - start
    print("SVC test: ", testing_time)
    np.save(f"5-fold_sets/Results/Sets{sets}/SVC_predicted.npy", predict_labels)
    np.save(
        f"5-fold_sets/Results/Sets{sets}/SVC_times.npy",
        [training_time, testing_time],
    )


def mlp(sets):
    train = pd.read_csv(f"5-fold_sets/Normalized/Sets{sets}/X_train", header=None)
    train_labels = pd.read_csv(f"5-fold_sets/Normalized/Sets{sets}/y_train", header=None)
    test = pd.read_csv(f"5-fold_sets/Normalized/Sets{sets}/X_test", header=None)

    MLP = MLPClassifier(max_iter=300)
    start = time.time()
    MLP.fit(train, train_labels)
    training_time = time.time() - start
    print("MLP train: ", training_time)
    start = time.time()
    predict_labels = MLP.predict(test)
    testing_time = time.time() - start
    print("MLP test: ", testing_time)

    np.save(f"5-fold_sets/Results/Sets{sets}/MLP_predicted.npy", predict_labels)
    np.save(
        f"5-fold_sets/Results/Sets{sets}/MLP_times.npy",
        [training_time, testing_time],
    )


def EFC(sets):
    test = pd.read_csv(f"5-fold_sets/Discretized/Sets{sets}/X_test", header=None).astype("int")
    test_labels = pd.read_csv(f"5-fold_sets/Discretized/Sets{sets}/y_test", squeeze=True, header=None).astype("int")
    train = pd.read_csv(f"5-fold_sets/Discretized/Sets{sets}/X_train", header=None).astype("int")
    train_labels = pd.read_csv(f"5-fold_sets/Discretized/Sets{sets}/y_train", squeeze=True, header=None).astype("int")
    Q = 30
    LAMBDA = 0.5

    start = time.time()
    h_i_matrices, coupling_matrices, cutoffs_list = MultiClassFit(np.array(train), np.array(train_labels), Q, LAMBDA)
    training_time = time.time() - start
    print("EFC train: ", training_time)

    start = time.time()
    predicted, energies = MultiClassPredict(
        np.array(test),
        h_i_matrices,
        coupling_matrices,
        cutoffs_list,
        Q,
        np.unique(train_labels),
    )
    testing_time = time.time() - start
    print("EFC test: ", testing_time)

    np.save(f"5-fold_sets/Results/Sets{sets}/EFC_predicted.npy", predicted)
    np.save(
        f"5-fold_sets/Results/Sets{sets}/EFC_times.npy",
        [training_time, testing_time],
    )

    print(classification_report(test_labels, predicted, labels=np.unique(test_labels)))


# with ProcessPoolExecutor(max_workers=3) as executor:
#     executor.map(mlp, range(1, 3))
#     executor.map(DT, range(1, 3))
#     # executor.map(svc, range(1, 3))
#     # executor.map(EFC, range(1, 3))
sets = 1
DT(sets)
mlp(sets)
svc(sets)
EFC(sets)
