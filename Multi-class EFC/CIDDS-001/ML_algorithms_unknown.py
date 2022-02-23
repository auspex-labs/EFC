import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import sys
sys.path.append('../../../efc')
from classification_functions import *
import time
import threading

def RF(removed, sets):
    train = pd.read_csv("5-fold_sets/Normalized/Sets{}/X_train".format(sets), header=None)
    train_labels = pd.read_csv("5-fold_sets/Normalized/Sets{}/y_train".format(sets), header=None)
    test = pd.read_csv("5-fold_sets/Normalized/Sets{}/X_test".format(sets), header=None)

    valid_indexes = np.where(train_labels == removed)[0]
    train.drop(valid_indexes, axis=0, inplace=True)
    train_labels.drop(valid_indexes, axis=0, inplace=True)

    RF = RandomForestClassifier(n_jobs=-1)
    start = time.time()
    RF.fit(train, train_labels)
    print("RF train: ", time.time()-start)
    start = time.time()
    predict_labels = RF.predict(test)
    print("RF test: ", time.time()-start)
    np.save("5-fold_sets/Results_removing{}/Sets{}/RF_predicted.npy".format(removed, sets), predict_labels)

def SVC(removed, sets):
    train = pd.read_csv("5-fold_sets/Normalized/Sets{}/X_train".format(sets), header=None)
    train_labels = pd.read_csv("5-fold_sets/Normalized/Sets{}/y_train".format(sets), header=None)
    test = pd.read_csv("5-fold_sets/Normalized/Sets{}/X_test".format(sets), header=None)

    valid_indexes = np.where(train_labels == removed)[0]
    train.drop(valid_indexes, axis=0, inplace=True)
    train_labels.drop(valid_indexes, axis=0, inplace=True)

    svc = SVC(kernel='poly')
    start = time.time()
    svc.fit(train, train_labels)
    train_time = time.time()-start
    start = time.time()
    predict_labels = svc.predict(test)
    test_time = time.time()-start
    np.save("5-fold_sets/Results_removing{}/Sets{}/SVC_predicted.npy".format(removed, sets), predict_labels)
    np.save("5-fold_sets/Results_removing{}/Sets{}/SVC_times.npy".format(removed, sets), [train_time, test_time])


def EFC(removed, sets):
    test = pd.read_csv("5-fold_sets/Discretized/Sets{}/X_test".format(sets), header=None).astype('int')
    test_labels = pd.read_csv("5-fold_sets/Discretized/Sets{}/y_test".format(sets), squeeze=True, header=None).astype('int')
    train = pd.read_csv("5-fold_sets/Discretized/Sets{}/X_train".format(sets), header=None).astype('int')
    train_labels = pd.read_csv("5-fold_sets/Discretized/Sets{}/y_train".format(sets), squeeze=True, header=None).astype('int')

    valid_indexes = np.where(train_labels == removed)[0]
    train.drop(valid_indexes, axis=0, inplace=True)
    train_labels.drop(valid_indexes, axis=0, inplace=True)

    Q = 32
    LAMBDA = 0.5
    start = time.time()
    h_i_matrices, coupling_matrices, cutoffs_list = MultiClassFit(np.array(train), np.array(train_labels), Q, LAMBDA)
    train_time = time.time()-start

    start = time.time()
    predicted = MultiClassPredict(np.array(test), h_i_matrices, coupling_matrices, cutoffs_list, Q, np.unique(train_labels))
    test_time = time.time()-start

    np.save("5-fold_sets/Results_removing{}/Sets{}/EFC_predicted.npy".format(removed, sets), predicted)
    np.save("5-fold_sets/Results_removing{}/Sets{}/EFC_times.npy".format(removed, sets), [train_time, test_time])
    print(classification_report(test_labels, predicted, labels=np.unique(test_labels)))


for removed in [0, 1, 3, 4]:
    for sets in range(1,6):
        threading.Thread(EFC, args=(removed, sets,)).start()
        threading.Thread(SVC, args=(removed, sets,)).start()
