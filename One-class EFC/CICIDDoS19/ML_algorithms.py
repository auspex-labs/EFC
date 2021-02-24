import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC, OneClassSVM
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from classification_functions import *
import pickle
import os
import time

def KNN(sets):
    train = np.load("Data/Non_discretized/Exp{}/train.npy".format(sets), allow_pickle=True)
    test = np.load("Data/Non_discretized/Exp{}/test.npy".format(sets), allow_pickle=True)
    test_labels = np.load("Data/Non_discretized/Exp{}/test_labels.npy".format(sets), allow_pickle=True)
    train_labels = np.load("Data/Non_discretized/Exp{}/train_labels.npy".format(sets), allow_pickle=True)
    test_labels = [0 if x=='BENIGN' else 1 for x in test_labels]
    train_labels = [0 if x=='BENIGN' else 1 for x in train_labels]

    external_test = np.load("../CICIDS17/External_test/Non_discretized/Exp{}/external_test.npy".format(sets), allow_pickle=True)
    external_test_labels = np.load("../CICIDS17/External_test/Non_discretized/Exp{}/external_test_labels.npy".format(sets), allow_pickle=True)
    external_test_labels = [0 if x=='BENIGN' else 1 for x in external_test_labels]

    f = open("Times.txt", 'a')
    KNN = KNeighborsClassifier()
    start = time.time()
    KNN.fit(train, train_labels)
    training_time = time.time()-start

    predict_prob_internal = KNN.predict_proba(test)
    start = time.time()
    predict_labels_internal = KNN.predict(test)
    testing_time = time.time()-start
    f.write("KNN & {} & {} \\\\ \n".format(training_time, testing_time))
    f.close()

    precision = precision_score(test_labels, predict_labels_internal)
    recall = recall_score(test_labels, predict_labels_internal)
    f1 = f1_score(test_labels, predict_labels_internal)
    roc = roc_auc_score(test_labels, predict_prob_internal[:,1])
    np.save("Data/Results/Exp{}/KNN_internal.npy".format(sets), [precision, recall, f1, roc])

    predict_prob_external = KNN.predict_proba(external_test)
    predict_labels_external = KNN.predict(external_test)
    precision = precision_score(external_test_labels, predict_labels_external)
    recall = recall_score(external_test_labels, predict_labels_external)
    f1 = f1_score(external_test_labels, predict_labels_external)
    roc = roc_auc_score(external_test_labels, predict_prob_external[:,1])
    np.save("Data/Results/Exp{}/KNN_external.npy".format(sets), [precision, recall, f1, roc])



def RF(sets):
    train = np.load("Data/Non_discretized/Exp{}/train.npy".format(sets), allow_pickle=True)
    test = np.load("Data/Non_discretized/Exp{}/test.npy".format(sets), allow_pickle=True)
    test_labels = np.load("Data/Non_discretized/Exp{}/test_labels.npy".format(sets), allow_pickle=True)
    train_labels = np.load("Data/Non_discretized/Exp{}/train_labels.npy".format(sets), allow_pickle=True)
    test_labels = [0 if x=='BENIGN' else 1 for x in test_labels]
    train_labels = [0 if x=='BENIGN' else 1 for x in train_labels]

    external_test = np.load("../CICIDS17/External_test/Non_discretized/Exp{}/external_test.npy".format(sets), allow_pickle=True)
    external_test_labels = np.load("../CICIDS17/External_test/Non_discretized/Exp{}/external_test_labels.npy".format(sets), allow_pickle=True)
    external_test_labels = [0 if x=='BENIGN' else 1 for x in external_test_labels]

    f = open("Times.txt", 'a')
    RF = RandomForestClassifier()
    start = time.time()
    RF.fit(train, train_labels)
    training_time = time.time()-start

    predict_prob_internal = RF.predict_proba(test)
    start = time.time()
    predict_labels_internal = RF.predict(test)
    testing_time = time.time()-start
    f.write("RF & {} & {} \\\\ \n".format(training_time, testing_time))
    f.close()
    precision = precision_score(test_labels, predict_labels_internal)
    recall = recall_score(test_labels, predict_labels_internal)
    f1 = f1_score(test_labels, predict_labels_internal)
    roc = roc_auc_score(test_labels, predict_prob_internal[:,1])
    np.save("Data/Results/Exp{}/RF_internal.npy".format(sets), [precision, recall, f1, roc])

    predict_prob_external = RF.predict_proba(external_test)
    predict_labels_external = RF.predict(external_test)
    precision = precision_score(external_test_labels, predict_labels_external)
    recall = recall_score(external_test_labels, predict_labels_external)
    f1 = f1_score(external_test_labels, predict_labels_external)
    roc = roc_auc_score(external_test_labels, predict_prob_external[:,1])
    np.save("Data/Results/Exp{}/RF_external.npy".format(sets), [precision, recall, f1, roc])



def GaussianNaiveB(sets):
    train = np.load("Data/Non_discretized/Exp{}/train.npy".format(sets), allow_pickle=True)
    test = np.load("Data/Non_discretized/Exp{}/test.npy".format(sets), allow_pickle=True)
    test_labels = np.load("Data/Non_discretized/Exp{}/test_labels.npy".format(sets), allow_pickle=True)
    train_labels = np.load("Data/Non_discretized/Exp{}/train_labels.npy".format(sets), allow_pickle=True)
    test_labels = [0 if x=='BENIGN' else 1 for x in test_labels]
    train_labels = [0 if x=='BENIGN' else 1 for x in train_labels]

    external_test = np.load("../CICIDS17/External_test/Non_discretized/Exp{}/external_test.npy".format(sets), allow_pickle=True)
    external_test_labels = np.load("../CICIDS17/External_test/Non_discretized/Exp{}/external_test_labels.npy".format(sets), allow_pickle=True)
    external_test_labels = [0 if x=='BENIGN' else 1 for x in external_test_labels]

    f = open("Times.txt", 'a')
    NB = GaussianNB()
    start = time.time()
    NB.fit(train, train_labels)
    training_time = time.time()-start

    predict_prob_internal = NB.predict_proba(test)
    start = time.time()
    predict_labels_internal = NB.predict(test)
    testing_time = time.time()-start
    f.write("NB & {} & {} \\\\ \n".format(training_time, testing_time))
    f.close()

    precision = precision_score(test_labels, predict_labels_internal)
    recall = recall_score(test_labels, predict_labels_internal)
    f1 = f1_score(test_labels, predict_labels_internal)
    roc = roc_auc_score(test_labels, predict_prob_internal[:,1])
    np.save("Data/Results/Exp{}/GaussianNB_internal.npy".format(sets), [precision, recall, f1, roc])

    predict_prob_external = NB.predict_proba(external_test)
    predict_labels_external = NB.predict(external_test)
    precision = precision_score(external_test_labels, predict_labels_external)
    recall = recall_score(external_test_labels, predict_labels_external)
    f1 = f1_score(external_test_labels, predict_labels_external)
    roc = roc_auc_score(external_test_labels, predict_prob_external[:,1])
    np.save("Data/Results/Exp{}/GaussianNB_external.npy".format(sets), [precision, recall, f1, roc])

def DT(sets):
    train = np.load("Data/Non_discretized/Exp{}/train.npy".format(sets), allow_pickle=True)
    test = np.load("Data/Non_discretized/Exp{}/test.npy".format(sets), allow_pickle=True)
    test_labels = np.load("Data/Non_discretized/Exp{}/test_labels.npy".format(sets), allow_pickle=True)
    train_labels = np.load("Data/Non_discretized/Exp{}/train_labels.npy".format(sets), allow_pickle=True)
    test_labels = [0 if x=='BENIGN' else 1 for x in test_labels]
    train_labels = [0 if x=='BENIGN' else 1 for x in train_labels]

    external_test = np.load("../CICIDS17/External_test/Non_discretized/Exp{}/external_test.npy".format(sets), allow_pickle=True)
    external_test_labels = np.load("../CICIDS17/External_test/Non_discretized/Exp{}/external_test_labels.npy".format(sets), allow_pickle=True)
    external_test_labels = [0 if x=='BENIGN' else 1 for x in external_test_labels]

    f = open("Times.txt", 'a')
    DT = DecisionTreeClassifier()
    start = time.time()
    DT.fit(train, train_labels)
    training_time = time.time()-start


    predict_prob_internal = DT.predict_proba(test)
    start = time.time()
    predict_labels_internal = DT.predict(test)
    testing_time = time.time()-start
    f.write("DT & {} & {} \\\\ \n".format(training_time, testing_time))
    f.close()

    precision = precision_score(test_labels, predict_labels_internal)
    recall = recall_score(test_labels, predict_labels_internal)
    f1 = f1_score(test_labels, predict_labels_internal)
    roc = roc_auc_score(test_labels, predict_prob_internal[:,1])
    np.save("Data/Results/Exp{}/DT_internal.npy".format(sets), [precision, recall, f1, roc])

    predict_prob_external = DT.predict_proba(external_test)
    predict_labels_external = DT.predict(external_test)
    precision = precision_score(external_test_labels, predict_labels_external)
    recall = recall_score(external_test_labels, predict_labels_external)
    f1 = f1_score(external_test_labels, predict_labels_external)
    roc = roc_auc_score(external_test_labels, predict_prob_external[:,1])
    np.save("Data/Results/Exp{}/DT_external.npy".format(sets), [precision, recall, f1, roc])

def Adaboost(sets):
    train = np.load("Data/Non_discretized/Exp{}/train.npy".format(sets), allow_pickle=True)
    test = np.load("Data/Non_discretized/Exp{}/test.npy".format(sets), allow_pickle=True)
    test_labels = np.load("Data/Non_discretized/Exp{}/test_labels.npy".format(sets), allow_pickle=True)
    train_labels = np.load("Data/Non_discretized/Exp{}/train_labels.npy".format(sets), allow_pickle=True)
    test_labels = [0 if x=='BENIGN' else 1 for x in test_labels]
    train_labels = [0 if x=='BENIGN' else 1 for x in train_labels]

    external_test = np.load("../CICIDS17/External_test/Non_discretized/Exp{}/external_test.npy".format(sets), allow_pickle=True)
    external_test_labels = np.load("../CICIDS17/External_test/Non_discretized/Exp{}/external_test_labels.npy".format(sets), allow_pickle=True)
    external_test_labels = [0 if x=='BENIGN' else 1 for x in external_test_labels]

    f = open("Times.txt", 'a')
    AD = AdaBoostClassifier()
    start = time.time()
    AD.fit(train, train_labels)
    training_time = time.time()-start

    predict_prob_internal = AD.predict_proba(test)
    start = time.time()
    predict_labels_internal = AD.predict(test)
    testing_time = time.time()-start
    f.write("KNN & {} & {} \\\\ \n".format(training_time, testing_time))
    f.close()

    precision = precision_score(test_labels, predict_labels_internal)
    recall = recall_score(test_labels, predict_labels_internal)
    f1 = f1_score(test_labels, predict_labels_internal)
    roc = roc_auc_score(test_labels, predict_prob_internal[:,1])
    np.save("Data/Results/Exp{}/Adaboost_internal.npy".format(sets), [precision, recall, f1, roc])

    predict_prob_external = AD.predict_proba(external_test)
    predict_labels_external = AD.predict(external_test)
    precision = precision_score(external_test_labels, predict_labels_external)
    recall = recall_score(external_test_labels, predict_labels_external)
    f1 = f1_score(external_test_labels, predict_labels_external)
    roc = roc_auc_score(external_test_labels, predict_prob_external[:,1])
    np.save("Data/Results/Exp{}/Adaboost_external.npy".format(sets), [precision, recall, f1, roc])

def svc(sets):
    train = np.load("Data/Non_discretized/Exp{}/train.npy".format(sets), allow_pickle=True)
    test = np.load("Data/Non_discretized/Exp{}/test.npy".format(sets), allow_pickle=True)
    test_labels = np.load("Data/Non_discretized/Exp{}/test_labels.npy".format(sets), allow_pickle=True)
    train_labels = np.load("Data/Non_discretized/Exp{}/train_labels.npy".format(sets), allow_pickle=True)
    test_labels = [0 if x=='BENIGN' else 1 for x in test_labels]
    train_labels = [0 if x=='BENIGN' else 1 for x in train_labels]

    external_test = np.load("../CICIDS17/External_test/Non_discretized/Exp{}/external_test.npy".format(sets), allow_pickle=True)
    external_test_labels = np.load("../CICIDS17/External_test/Non_discretized/Exp{}/external_test_labels.npy".format(sets), allow_pickle=True)
    external_test_labels = [0 if x=='BENIGN' else 1 for x in external_test_labels]

    transformer = Normalizer().fit(train)
    train = transformer.transform(train)
    transformer = Normalizer().fit(test)
    test = transformer.transform(test)
    transformer = Normalizer().fit(external_test)
    external_test = transformer.transform(external_test)

    f = open("Times.txt", 'a')
    svc = SVC(kernel='poly', probability=True)
    start = time.time()
    svc.fit(train, train_labels)
    training_time = time.time()-start

    predict_prob_internal = svc.predict_proba(test)
    start = time.time()
    predict_labels_internal = svc.predict(test)
    testing_time = time.time()-start
    f.write("SVC & {} & {} \\\\ \n".format(training_time, testing_time))
    f.close()

    precision = precision_score(test_labels, predict_labels_internal)
    recall = recall_score(test_labels, predict_labels_internal)
    f1 = f1_score(test_labels, predict_labels_internal)
    roc = roc_auc_score(test_labels, predict_prob_internal[:,1])
    np.save("Data/Results/Exp{}/SVC_internal.npy".format(sets), [precision, recall, f1, roc])

    predict_prob_external = svc.predict_proba(external_test)
    predict_labels_external = svc.predict(external_test)
    precision = precision_score(external_test_labels, predict_labels_external)
    recall = recall_score(external_test_labels, predict_labels_external)
    f1 = f1_score(external_test_labels, predict_labels_external)
    roc = roc_auc_score(external_test_labels, predict_prob_external[:,1])
    np.save("Data/Results/Exp{}/SVC_external.npy".format(sets), [precision, recall, f1, roc])

def mlp(sets):
    train = np.load("Data/Non_discretized/Exp{}/train.npy".format(sets), allow_pickle=True)
    test = np.load("Data/Non_discretized/Exp{}/test.npy".format(sets), allow_pickle=True)
    test_labels = np.load("Data/Non_discretized/Exp{}/test_labels.npy".format(sets), allow_pickle=True)
    train_labels = np.load("Data/Non_discretized/Exp{}/train_labels.npy".format(sets), allow_pickle=True)
    test_labels = [0 if x=='BENIGN' else 1 for x in test_labels]
    train_labels = [0 if x=='BENIGN' else 1 for x in train_labels]

    external_test = np.load("../CICIDS17/External_test/Non_discretized/Exp{}/external_test.npy".format(sets), allow_pickle=True)
    external_test_labels = np.load("../CICIDS17/External_test/Non_discretized/Exp{}/external_test_labels.npy".format(sets), allow_pickle=True)
    external_test_labels = [0 if x=='BENIGN' else 1 for x in external_test_labels]

    transformer = Normalizer().fit(train)
    train = transformer.transform(train)
    transformer = Normalizer().fit(test)
    test = transformer.transform(test)
    transformer = Normalizer().fit(external_test)
    external_test = transformer.transform(external_test)

    f = open("Times.txt", 'a')
    MLP = MLPClassifier(max_iter=300)
    start = time.time()
    MLP.fit(train, train_labels)
    training_time = time.time()-start

    predict_prob_internal = MLP.predict_proba(test)
    start = time.time()
    predict_labels_internal = MLP.predict(test)
    testing_time = time.time()-start
    f.write("MLP & {} & {} \\\\ \n".format(training_time, testing_time))
    f.close()

    precision = precision_score(test_labels, predict_labels_internal)
    recall = recall_score(test_labels, predict_labels_internal)
    f1 = f1_score(test_labels, predict_labels_internal)
    roc = roc_auc_score(test_labels, predict_prob_internal[:,1])
    np.save("Data/Results/Exp{}/MLP_internal.npy".format(sets), [precision, recall, f1, roc])

    predict_prob_external = MLP.predict_proba(external_test)
    predict_labels_external = MLP.predict(external_test)
    precision = precision_score(external_test_labels, predict_labels_external)
    recall = recall_score(external_test_labels, predict_labels_external)
    f1 = f1_score(external_test_labels, predict_labels_external)
    roc = roc_auc_score(external_test_labels, predict_prob_external[:,1])
    np.save("Data/Results/Exp{}/MLP_external.npy".format(sets), [precision, recall, f1, roc])

def EFC(sets):
    train = np.load("Data/Discretized/Exp{}/train.npy".format(sets), allow_pickle=True).astype('int')
    train_normal = train[:int(train.shape[0]/2), :]
    test = np.load("Data/Discretized/Exp{}/test.npy".format(sets), allow_pickle=True).astype('int')
    test_labels = np.load("Data/Discretized/Exp{}/test_labels.npy".format(sets), allow_pickle=True).astype('int')
    test_labels = np.array([0 if x==0 else 1 for x in test_labels], dtype=int)

    external_test = np.load("../CICIDS17/External_test/Discretized/Exp{}/external_test.npy".format(sets), allow_pickle=True).astype('int')
    external_test_labels = np.load("../CICIDS17/External_test/Discretized/Exp{}/external_test_labels.npy".format(sets), allow_pickle=True).astype('int')
    external_test_labels = np.array([0 if x==0 else 1 for x in external_test_labels], dtype=int)

    Q = 14
    LAMBDA = 0.5

    # Creating model
    couplingmatrix, h_i = create_model(train_normal, Q, LAMBDA)
    CUTOFF = define_cutoff(train_normal, h_i, couplingmatrix, Q)

    np.save("Data/Discretized/Exp{}/h_i.npy".format(sets), h_i)
    np.save("Data/Discretized/Exp{}/couplingmatrix.npy".format(sets), couplingmatrix)
    np.save("Data/Discretized/Exp{}/cutoff.npy".format(sets), np.array(CUTOFF))

    # Testing model in same context
    predicted_labels_internal, energies_internal = test_model(test, couplingmatrix, h_i, test_labels, CUTOFF, Q)

    np.save("Data/Discretized/Exp{}/energies_internal.npy".format(sets),energies_internal)
    predict_prob = [x for x in MinMaxScaler().fit_transform(np.array(energies_internal).reshape(-1,1))]
    precision = precision_score(test_labels, predicted_labels_internal)
    recall = recall_score(test_labels, predicted_labels_internal)
    f1 = f1_score(test_labels, predicted_labels_internal)
    roc = roc_auc_score(test_labels, predict_prob)
    np.save("Data/Results/Exp{}/EFC_internal.npy".format(sets), np.array([precision, recall, f1, roc]))

    # Testing model in different context
    predicted_labels_external, energies_external = test_model(external_test, couplingmatrix, h_i, external_test_labels, CUTOFF, Q)
    np.save("Data/Discretized/Exp{}/energies_external.npy".format(sets), energies_external)
    predict_prob = [x for x in MinMaxScaler().fit_transform(np.array(energies_external).reshape(-1,1))]
    precision = precision_score(external_test_labels, predicted_labels_external)
    recall = recall_score(external_test_labels, predicted_labels_external)
    f1 = f1_score(external_test_labels, predicted_labels_external)
    roc = roc_auc_score(external_test_labels, predict_prob)
    np.save("Data/Results/Exp{}/EFC_external.npy".format(sets), np.array([precision, recall, f1, roc]))


for i in range(1,11):
    EFC(i)
    KNN(i)
    RF(i)
    DT(i)
    Adaboost(i)
    GaussianNaiveB(i)
    svc(i)
    mlp(i)
