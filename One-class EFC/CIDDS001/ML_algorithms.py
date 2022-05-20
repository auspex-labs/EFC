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


def KNN(sets):
    train = np.load(f"Data/Non_discretized/Exp{sets}/train.npy", allow_pickle=True)
    test = np.load(f"Data/Non_discretized/Exp{sets}/test.npy", allow_pickle=True)
    test_labels = np.load(f"Data/Non_discretized/Exp{sets}/test_labels.npy", allow_pickle=True)
    train_labels = np.load(f"Data/Non_discretized/Exp{sets}/train_labels.npy", allow_pickle=True)

    external_test = np.load(f"External_test/Non_discretized/Exp{sets}/external_test.npy", allow_pickle=True)
    external_test_labels = np.load("External_test/Non_discretized/Exp{}/external_test_labels.npy".format(sets), allow_pickle=True)

    KNN = KNeighborsClassifier()
    start = time.time()
    KNN.fit(train, train_labels)
    training_time = time.time() - start

    predict_prob_internal = KNN.predict_proba(test)
    start = time.time()
    predict_labels_internal = KNN.predict(test)
    testing_time = time.time() - start

    np.save(f"Data/Results/Exp{sets}/KNN_times.npy", [training_time, testing_time])
    precision = precision_score(test_labels, predict_labels_internal)
    recall = recall_score(test_labels, predict_labels_internal)
    f1 = f1_score(test_labels, predict_labels_internal)
    roc = roc_auc_score(test_labels, predict_prob_internal[:, 1])
    np.save(f"Data/Results/Exp{sets}/KNN_internal.npy", [precision, recall, f1, roc])

    predict_prob_external = KNN.predict_proba(external_test)
    predict_labels_external = KNN.predict(external_test)
    precision = precision_score(external_test_labels, predict_labels_external)
    recall = recall_score(external_test_labels, predict_labels_external)
    f1 = f1_score(external_test_labels, predict_labels_external)
    roc = roc_auc_score(external_test_labels, predict_prob_external[:, 1])
    np.save(f"Data/Results/Exp{sets}/KNN_external.npy", [precision, recall, f1, roc])


def RF(sets):
    train = np.load(f"Data/Non_discretized/Exp{sets}/train.npy", allow_pickle=True)
    test = np.load(f"Data/Non_discretized/Exp{sets}/test.npy", allow_pickle=True)
    test_labels = np.load(f"Data/Non_discretized/Exp{sets}/test_labels.npy", allow_pickle=True)
    train_labels = np.load(f"Data/Non_discretized/Exp{sets}/train_labels.npy", allow_pickle=True)

    external_test = np.load(f"External_test/Non_discretized/Exp{sets}/external_test.npy", allow_pickle=True)
    external_test_labels = np.load("External_test/Non_discretized/Exp{}/external_test_labels.npy".format(sets), allow_pickle=True)

    RF = RandomForestClassifier()
    start = time.time()
    RF.fit(train, train_labels)
    training_time = time.time() - start

    predict_prob_internal = RF.predict_proba(test)
    start = time.time()
    predict_labels_internal = RF.predict(test)
    testing_time = time.time() - start
    np.save(f"Data/Results/Exp{sets}/RF_times.npy", [training_time, testing_time])
    precision = precision_score(test_labels, predict_labels_internal)
    recall = recall_score(test_labels, predict_labels_internal)
    f1 = f1_score(test_labels, predict_labels_internal)
    roc = roc_auc_score(test_labels, predict_prob_internal[:, 1])
    np.save(f"Data/Results/Exp{sets}/RF_internal.npy", [precision, recall, f1, roc])

    predict_prob_external = RF.predict_proba(external_test)
    predict_labels_external = RF.predict(external_test)
    precision = precision_score(external_test_labels, predict_labels_external)
    recall = recall_score(external_test_labels, predict_labels_external)
    f1 = f1_score(external_test_labels, predict_labels_external)
    roc = roc_auc_score(external_test_labels, predict_prob_external[:, 1])
    np.save(f"Data/Results/Exp{sets}/RF_external.npy", [precision, recall, f1, roc])


def GaussianNaiveB(sets):
    train = np.load(f"Data/Non_discretized/Exp{sets}/train.npy", allow_pickle=True)
    test = np.load(f"Data/Non_discretized/Exp{sets}/test.npy", allow_pickle=True)
    test_labels = np.load(f"Data/Non_discretized/Exp{sets}/test_labels.npy", allow_pickle=True)
    train_labels = np.load(f"Data/Non_discretized/Exp{sets}/train_labels.npy", allow_pickle=True)

    external_test = np.load(f"External_test/Non_discretized/Exp{sets}/external_test.npy", allow_pickle=True)
    external_test_labels = np.load("External_test/Non_discretized/Exp{}/external_test_labels.npy".format(sets), allow_pickle=True)

    NB = GaussianNB()
    start = time.time()
    NB.fit(train, train_labels)
    training_time = time.time() - start

    predict_prob_internal = NB.predict_proba(test)
    start = time.time()
    predict_labels_internal = NB.predict(test)
    testing_time = time.time() - start

    np.save(f"Data/Results/Exp{sets}/GaussianNB_times.npy", [training_time, testing_time])
    precision = precision_score(test_labels, predict_labels_internal)
    recall = recall_score(test_labels, predict_labels_internal)
    f1 = f1_score(test_labels, predict_labels_internal)
    roc = roc_auc_score(test_labels, predict_prob_internal[:, 1])
    np.save(f"Data/Results/Exp{sets}/GaussianNB_internal.npy", [precision, recall, f1, roc])

    predict_prob_external = NB.predict_proba(external_test)
    predict_labels_external = NB.predict(external_test)
    precision = precision_score(external_test_labels, predict_labels_external)
    recall = recall_score(external_test_labels, predict_labels_external)
    f1 = f1_score(external_test_labels, predict_labels_external)
    roc = roc_auc_score(external_test_labels, predict_prob_external[:, 1])
    np.save(f"Data/Results/Exp{sets}/GaussianNB_external.npy", [precision, recall, f1, roc])


def DT(sets):
    train = np.load(f"Data/Non_discretized/Exp{sets}/train.npy", allow_pickle=True)
    test = np.load(f"Data/Non_discretized/Exp{sets}/test.npy", allow_pickle=True)
    test_labels = np.load(f"Data/Non_discretized/Exp{sets}/test_labels.npy", allow_pickle=True)
    train_labels = np.load(f"Data/Non_discretized/Exp{sets}/train_labels.npy", allow_pickle=True)

    external_test = np.load(f"External_test/Non_discretized/Exp{sets}/external_test.npy", allow_pickle=True)
    external_test_labels = np.load("External_test/Non_discretized/Exp{}/external_test_labels.npy".format(sets), allow_pickle=True)

    DT = DecisionTreeClassifier()
    start = time.time()
    DT.fit(train, train_labels)
    training_time = time.time() - start

    predict_prob_internal = DT.predict_proba(test)
    start = time.time()
    predict_labels_internal = DT.predict(test)
    testing_time = time.time() - start
    np.save(f"Data/Results/Exp{sets}/DT_times.npy", [training_time, testing_time])

    precision = precision_score(test_labels, predict_labels_internal)
    recall = recall_score(test_labels, predict_labels_internal)
    f1 = f1_score(test_labels, predict_labels_internal)
    roc = roc_auc_score(test_labels, predict_prob_internal[:, 1])
    np.save(f"Data/Results/Exp{sets}/DT_internal.npy", [precision, recall, f1, roc])

    predict_prob_external = DT.predict_proba(external_test)
    predict_labels_external = DT.predict(external_test)
    precision = precision_score(external_test_labels, predict_labels_external)
    recall = recall_score(external_test_labels, predict_labels_external)
    f1 = f1_score(external_test_labels, predict_labels_external)
    roc = roc_auc_score(external_test_labels, predict_prob_external[:, 1])
    np.save(f"Data/Results/Exp{sets}/DT_external.npy", [precision, recall, f1, roc])


def Adaboost(sets):
    train = np.load(f"Data/Non_discretized/Exp{sets}/train.npy", allow_pickle=True)
    test = np.load(f"Data/Non_discretized/Exp{sets}/test.npy", allow_pickle=True)
    test_labels = np.load(f"Data/Non_discretized/Exp{sets}/test_labels.npy", allow_pickle=True)
    train_labels = np.load(f"Data/Non_discretized/Exp{sets}/train_labels.npy", allow_pickle=True)

    external_test = np.load(f"External_test/Non_discretized/Exp{sets}/external_test.npy", allow_pickle=True)
    external_test_labels = np.load("External_test/Non_discretized/Exp{}/external_test_labels.npy".format(sets), allow_pickle=True)

    AD = AdaBoostClassifier()
    start = time.time()
    AD.fit(train, train_labels)
    training_time = time.time() - start

    predict_prob_internal = AD.predict_proba(test)
    start = time.time()
    predict_labels_internal = AD.predict(test)
    testing_time = time.time() - start
    np.save(f"Data/Results/Exp{sets}/Adaboost_times.npy", [training_time, testing_time])

    precision = precision_score(test_labels, predict_labels_internal)
    recall = recall_score(test_labels, predict_labels_internal)
    f1 = f1_score(test_labels, predict_labels_internal)
    roc = roc_auc_score(test_labels, predict_prob_internal[:, 1])
    np.save(f"Data/Results/Exp{sets}/Adaboost_internal.npy", [precision, recall, f1, roc])

    predict_prob_external = AD.predict_proba(external_test)
    predict_labels_external = AD.predict(external_test)
    precision = precision_score(external_test_labels, predict_labels_external)
    recall = recall_score(external_test_labels, predict_labels_external)
    f1 = f1_score(external_test_labels, predict_labels_external)
    roc = roc_auc_score(external_test_labels, predict_prob_external[:, 1])
    np.save(f"Data/Results/Exp{sets}/Adaboost_external.npy", [precision, recall, f1, roc])


def svc(sets):
    train = np.load(f"Data/Non_discretized/Exp{sets}/train.npy", allow_pickle=True)
    test = np.load(f"Data/Non_discretized/Exp{sets}/test.npy", allow_pickle=True)
    test_labels = np.load(f"Data/Non_discretized/Exp{sets}/test_labels.npy", allow_pickle=True)
    train_labels = np.load(f"Data/Non_discretized/Exp{sets}/train_labels.npy", allow_pickle=True)

    external_test = np.load(f"External_test/Non_discretized/Exp{sets}/external_test.npy", allow_pickle=True)
    external_test_labels = np.load("External_test/Non_discretized/Exp{}/external_test_labels.npy".format(sets), allow_pickle=True)

    transformer = Normalizer().fit(train)
    train = transformer.transform(train)
    transformer = Normalizer().fit(test)
    test = transformer.transform(test)
    transformer = Normalizer().fit(external_test)
    external_test = transformer.transform(external_test)

    svc = SVC(kernel="poly", probability=True)
    start = time.time()
    svc.fit(train, train_labels)
    training_time = time.time() - start

    predict_prob_internal = svc.predict_proba(test)
    start = time.time()
    predict_labels_internal = svc.predict(test)
    testing_time = time.time() - start

    np.save(f"Data/Results/Exp{sets}/SVC_times.npy", [training_time, testing_time])
    precision = precision_score(test_labels, predict_labels_internal)
    recall = recall_score(test_labels, predict_labels_internal)
    f1 = f1_score(test_labels, predict_labels_internal)
    roc = roc_auc_score(test_labels, predict_prob_internal[:, 1])
    np.save(f"Data/Results/Exp{sets}/SVC_internal.npy", [precision, recall, f1, roc])

    predict_prob_external = svc.predict_proba(external_test)
    predict_labels_external = svc.predict(external_test)
    precision = precision_score(external_test_labels, predict_labels_external)
    recall = recall_score(external_test_labels, predict_labels_external)
    f1 = f1_score(external_test_labels, predict_labels_external)
    roc = roc_auc_score(external_test_labels, predict_prob_external[:, 1])
    np.save(f"Data/Results/Exp{sets}/SVC_external.npy", [precision, recall, f1, roc])


def mlp(sets):
    train = np.load(f"Data/Non_discretized/Exp{sets}/train.npy", allow_pickle=True)
    test = np.load(f"Data/Non_discretized/Exp{sets}/test.npy", allow_pickle=True)
    test_labels = np.load(f"Data/Non_discretized/Exp{sets}/test_labels.npy", allow_pickle=True)
    train_labels = np.load(f"Data/Non_discretized/Exp{sets}/train_labels.npy", allow_pickle=True)

    external_test = np.load(f"External_test/Non_discretized/Exp{sets}/external_test.npy", allow_pickle=True)
    external_test_labels = np.load("External_test/Non_discretized/Exp{}/external_test_labels.npy".format(sets), allow_pickle=True)

    transformer = Normalizer().fit(train)
    train = transformer.transform(train)
    transformer = Normalizer().fit(test)
    test = transformer.transform(test)
    transformer = Normalizer().fit(external_test)
    external_test = transformer.transform(external_test)

    MLP = MLPClassifier(max_iter=300)
    start = time.time()
    MLP.fit(train, train_labels)
    training_time = time.time() - start

    predict_prob_internal = MLP.predict_proba(test)
    start = time.time()
    predict_labels_internal = MLP.predict(test)
    testing_time = time.time() - start
    np.save(f"Data/Results/Exp{sets}/MLP_times.npy", [training_time, testing_time])
    precision = precision_score(test_labels, predict_labels_internal)
    recall = recall_score(test_labels, predict_labels_internal)
    f1 = f1_score(test_labels, predict_labels_internal)
    roc = roc_auc_score(test_labels, predict_prob_internal[:, 1])
    np.save(f"Data/Results/Exp{sets}/MLP_internal.npy", [precision, recall, f1, roc])

    predict_prob_external = MLP.predict_proba(external_test)
    predict_labels_external = MLP.predict(external_test)
    precision = precision_score(external_test_labels, predict_labels_external)
    recall = recall_score(external_test_labels, predict_labels_external)
    f1 = f1_score(external_test_labels, predict_labels_external)
    roc = roc_auc_score(external_test_labels, predict_prob_external[:, 1])
    np.save(f"Data/Results/Exp{sets}/MLP_external.npy", [precision, recall, f1, roc])


def EFC(sets):
    train = np.load(f"Data/Discretized/Exp{sets}/train.npy", allow_pickle=True).astype("int")
    train_normal = train[: int(train.shape[0] / 2), :]
    test = np.load(f"Data/Discretized/Exp{sets}/test.npy", allow_pickle=True).astype("int")
    test_labels = np.load(f"Data/Discretized/Exp{sets}/test_labels.npy", allow_pickle=True).astype("int")

    external_test = np.load(f"External_test/Discretized/Exp{sets}/external_test.npy", allow_pickle=True).astype("int")
    external_test_labels = np.load(
        "External_test/Discretized/Exp{}/external_test_labels.npy".format(sets), allow_pickle=True
    ).astype("int")

    Q = 32
    LAMBDA = 0.5

    # Creating model
    start = time.time()
    couplingmatrix, h_i, cutoff = create_oneclass_model(train_normal, Q, LAMBDA)
    training_time = time.time() - start
    np.save(f"Data/Discretized/Exp{sets}/cutoff.npy", np.array(cutoff))
    np.save(f"Data/Discretized/Exp{sets}/h_i.npy", h_i)
    np.save(f"Data/Discretized/Exp{sets}/couplingmatrix.npy", couplingmatrix)

    start = time.time()
    predicted_labels_internal, energies_internal = test_oneclass_model(
        np.array(test, dtype=int), couplingmatrix, h_i, test_labels, cutoff, Q
    )
    testing_time = time.time() - start
    np.save(f"Data/Results/Exp{sets}/EFC_times.npy", [training_time, testing_time])

    print("Train:", training_time)
    print("Test:", testing_time)

    np.save(f"Data/Discretized/Exp{sets}/energies_internal.npy", np.array(energies_internal))
    predict_prob = [x for x in MinMaxScaler().fit_transform(np.array(energies_internal).reshape(-1, 1))]
    precision = precision_score(test_labels, predicted_labels_internal)
    recall = recall_score(test_labels, predicted_labels_internal)
    f1 = f1_score(test_labels, predicted_labels_internal)
    roc = roc_auc_score(test_labels, predict_prob)
    np.save(f"Data/Results/Exp{sets}/EFC_internal.npy", np.array([precision, recall, f1, roc]))
    print(f1, roc)

    predicted_labels_external, energies_external = test_oneclass_model(
        np.array(external_test, dtype=int), couplingmatrix, h_i, external_test_labels, cutoff, Q
    )
    np.save(f"Data/Discretized/Exp{sets}/energies_external.npy", np.array(energies_external))
    predict_prob = [x for x in MinMaxScaler().fit_transform(np.array(energies_external).reshape(-1, 1))]
    precision = precision_score(external_test_labels, predicted_labels_external)
    recall = recall_score(external_test_labels, predicted_labels_external)
    f1 = f1_score(external_test_labels, predicted_labels_external)
    roc = roc_auc_score(external_test_labels, predict_prob)
    print(f1, roc)
    np.save(f"Data/Results/Exp{sets}/EFC_external.npy", np.array([precision, recall, f1, roc]))


for i in range(1, 11):
    EFC(i)
    KNN(i)
    RF(i)
    DT(i)
    Adaboost(i)
    GaussianNaiveB(i)
    svc(i)
    mlp(i)
