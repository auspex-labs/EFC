import os
import pickle
import time

import numpy as np
import pandas as pd
from classification_functions import *
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.svm import SVC, OneClassSVM
from sklearn.tree import DecisionTreeClassifier


def KNN(sets, exp):
    train = np.load(f"TimeData/Non_discretized/Size{sets}/train.npy", allow_pickle=True)
    test = np.load(f"TimeData/Non_discretized/Size{sets}/test.npy", allow_pickle=True)
    test_labels = np.load(f"TimeData/Non_discretized/Size{sets}/test_labels.npy", allow_pickle=True)
    train_labels = np.load(f"TimeData/Non_discretized/Size{sets}/train_labels.npy", allow_pickle=True)

    KNN = KNeighborsClassifier(algorithm="kd_tree")
    start = time.time()
    KNN.fit(train, train_labels)
    training_time = time.time() - start

    start = time.time()
    predict_labels_internal = KNN.predict(test)
    testing_time = time.time() - start
    print("Train:", training_time)
    print("Test:", testing_time)

    np.save(f"TimeData/Results/Size{sets}/Exp{exp}/KNN_times.npy", [training_time, testing_time])


def RF(sets, exp):
    train = np.load(f"TimeData/Non_discretized/Size{sets}/train.npy", allow_pickle=True)
    test = np.load(f"TimeData/Non_discretized/Size{sets}/test.npy", allow_pickle=True)
    test_labels = np.load(f"TimeData/Non_discretized/Size{sets}/test_labels.npy", allow_pickle=True)
    train_labels = np.load(f"TimeData/Non_discretized/Size{sets}/train_labels.npy", allow_pickle=True)

    RF = RandomForestClassifier()
    start = time.time()
    RF.fit(train, train_labels)
    training_time = time.time() - start

    start = time.time()
    predict_labels_internal = RF.predict(test)
    testing_time = time.time() - start
    print("Train:", training_time)
    print("Test:", testing_time)
    np.save(f"TimeData/Results/Size{sets}/Exp{exp}/RF_times.npy", [training_time, testing_time])


def GaussianNaiveB(sets, exp):
    train = np.load(f"TimeData/Non_discretized/Size{sets}/train.npy", allow_pickle=True)
    test = np.load(f"TimeData/Non_discretized/Size{sets}/test.npy", allow_pickle=True)
    test_labels = np.load(f"TimeData/Non_discretized/Size{sets}/test_labels.npy", allow_pickle=True)
    train_labels = np.load(f"TimeData/Non_discretized/Size{sets}/train_labels.npy", allow_pickle=True)

    NB = GaussianNB()
    start = time.time()
    NB.fit(train, train_labels)
    training_time = time.time() - start

    start = time.time()
    predict_labels_internal = NB.predict(test)
    testing_time = time.time() - start
    print("Train:", training_time)
    print("Test:", testing_time)
    np.save(f"TimeData/Results/Size{sets}/Exp{exp}/GaussianNB_times.npy", [training_time, testing_time])


def DT(sets, exp):
    train = np.load(f"TimeData/Non_discretized/Size{sets}/train.npy", allow_pickle=True)
    test = np.load(f"TimeData/Non_discretized/Size{sets}/test.npy", allow_pickle=True)
    test_labels = np.load(f"TimeData/Non_discretized/Size{sets}/test_labels.npy", allow_pickle=True)
    train_labels = np.load(f"TimeData/Non_discretized/Size{sets}/train_labels.npy", allow_pickle=True)

    DT = DecisionTreeClassifier()
    start = time.time()
    DT.fit(train, train_labels)
    training_time = time.time() - start

    start = time.time()
    predict_labels_internal = DT.predict(test)
    testing_time = time.time() - start
    print("DT train:", training_time)
    print("DT test:", testing_time)
    np.save(f"TimeData/Results/Size{sets}/Exp{exp}/DT_times.npy", [training_time, testing_time])


def Adaboost(sets, exp):
    train = np.load(f"TimeData/Non_discretized/Size{sets}/train.npy", allow_pickle=True)
    test = np.load(f"TimeData/Non_discretized/Size{sets}/test.npy", allow_pickle=True)
    test_labels = np.load(f"TimeData/Non_discretized/Size{sets}/test_labels.npy", allow_pickle=True)
    train_labels = np.load(f"TimeData/Non_discretized/Size{sets}/train_labels.npy", allow_pickle=True)

    AD = AdaBoostClassifier()
    start = time.time()
    AD.fit(train, train_labels)
    training_time = time.time() - start

    start = time.time()
    predict_labels_internal = AD.predict(test)
    testing_time = time.time() - start
    print("Train:", training_time)
    print("Test:", testing_time)
    np.save(f"TimeData/Results/Size{sets}/Exp{exp}/Adaboost_times.npy", [training_time, testing_time])


def svc(sets, exp):
    train = np.load(f"TimeData/Non_discretized/Size{sets}/train.npy", allow_pickle=True)
    test = np.load(f"TimeData/Non_discretized/Size{sets}/test.npy", allow_pickle=True)
    test_labels = np.load(f"TimeData/Non_discretized/Size{sets}/test_labels.npy", allow_pickle=True)
    train_labels = np.load(f"TimeData/Non_discretized/Size{sets}/train_labels.npy", allow_pickle=True)

    transformer = Normalizer().fit(train)
    train = transformer.transform(train)
    transformer = Normalizer().fit(test)
    test = transformer.transform(test)

    svc = SVC(kernel="poly", probability=True)
    start = time.time()
    svc.fit(train, train_labels)
    training_time = time.time() - start

    start = time.time()
    predict_labels_internal = svc.predict(test)
    testing_time = time.time() - start
    print("Train:", training_time)
    print("Test:", testing_time)
    np.save(f"TimeData/Results/Size{sets}/Exp{exp}/SVC_times.npy", [training_time, testing_time])


def mlp(sets, exp):
    train = np.load(f"TimeData/Non_discretized/Size{sets}/train.npy", allow_pickle=True)
    test = np.load(f"TimeData/Non_discretized/Size{sets}/test.npy", allow_pickle=True)
    test_labels = np.load(f"TimeData/Non_discretized/Size{sets}/test_labels.npy", allow_pickle=True)
    train_labels = np.load(f"TimeData/Non_discretized/Size{sets}/train_labels.npy", allow_pickle=True)

    transformer = Normalizer().fit(train)
    train = transformer.transform(train)
    transformer = Normalizer().fit(test)
    test = transformer.transform(test)

    MLP = MLPClassifier(max_iter=300)
    start = time.time()
    MLP.fit(train, train_labels)
    training_time = time.time() - start

    start = time.time()
    predict_labels_internal = MLP.predict(test)
    testing_time = time.time() - start
    print("Train:", training_time)
    print("Test:", testing_time)
    np.save(f"TimeData/Results/Size{sets}/Exp{exp}/MLP_times.npy", [training_time, testing_time])


def EFC(sets, exp):
    train = np.load(f"TimeData/Discretized/Size{sets}/train.npy", allow_pickle=True).astype("int")
    train_normal = train[: int(train.shape[0] / 2), :]
    test = np.load(f"TimeData/Discretized/Size{sets}/test.npy", allow_pickle=True).astype("int")
    test_labels = np.load(f"TimeData/Discretized/Size{sets}/test_labels.npy", allow_pickle=True).astype("int")

    Q = 32
    LAMBDA = 0.5

    # Creating model
    start = time.time()
    couplingmatrix, h_i, cutoff = create_oneclass_model(train_normal, Q, LAMBDA)
    training_time = time.time() - start

    start = time.time()
    predicted_labels_internal, energies_internal = test_oneclass_model(
        np.array(test, dtype=int), couplingmatrix, h_i, test_labels, CUTOFF, Q
    )
    testing_time = time.time() - start
    np.save(f"TimeData/Results/Size{sets}/Exp{exp}/EFC_times.npy", [training_time, testing_time])

    print("Train:", training_time)
    print("Test:", testing_time)


for size in [0, 1, 2, 3]:
    for exp in range(1, 11):
        os.makedirs(f"TimeData/Results/Size{size}/Exp{exp}/", exist_ok=True)
        EFC(size, exp)
        KNN(size, exp)
        RF(size, exp)
        DT(size, exp)
        Adaboost(size, exp)
        GaussianNaiveB(size, exp)
        svc(size, exp)
        mlp(size, exp)
