import os
import shutil

import numpy as np
import pandas as pd

PATH = "CIDDS-001/test_sets/"

for i in range(1, 11):
    test = np.array(pd.read_csv(PATH + f"non-discretized/{i}_test_cidds_ext.csv"))
    test_labels = test[:, 0]
    test_labels = [1 if x == "suspicious" else 0 for x in test_labels]

    test = np.delete(test, 0, axis=1)
    np.save(f"External_test/Non_discretized/Exp{i}/external_test.npy", np.array(test))
    np.save("External_test/Non_discretized/Exp{}/external_test_labels.npy".format(i), np.array(test_labels))

    test = np.load(PATH + f"discretized/{i}_test_ext.npy", allow_pickle=True)
    test_labels = np.load(PATH + f"discretized/{i}_labels_ext.npy", allow_pickle=True)
    test_labels = [1 if x == "suspicious" else 0 for x in test_labels]
    np.save(f"External_test/Discretized/Exp{i}/external_test.npy", np.array(test))
    np.save(f"External_test/Discretized/Exp{i}/external_test_labels.npy", np.array(test_labels))

    set = np.array(pd.read_csv(PATH + f"non-discretized/{i}_test_cidds_os.csv"))
    set_labels = list(set[:, 0])
    set_labels = [0 if x == "normal" else 1 for x in set_labels]

    set = list(np.delete(set, 0, axis=1))
    train = set[:8000] + set[10000:18000]
    train_labels = set_labels[:8000] + set_labels[10000:18000]
    test = set[8000:10000] + set[18000::]
    test_labels = set_labels[8000:10000] + set_labels[18000::]
    np.save(f"Data/Non_discretized/Exp{i}/train.npy", np.array(train))
    np.save(f"Data/Non_discretized/Exp{i}/train_labels.npy", np.array(train_labels))
    np.save(f"Data/Non_discretized/Exp{i}/test.npy", np.array(test))
    np.save(f"Data/Non_discretized/Exp{i}/test_labels.npy", np.array(test_labels))

    set = list(np.load(PATH + f"discretized/{i}_test_os.npy", allow_pickle=True))
    set_labels = np.load(PATH + f"discretized/{i}_labels_os.npy", allow_pickle=True)
    set_labels = [0 if x == "normal" else 1 for x in set_labels]
    print(np.unique(set_labels))

    train = set[:8000] + set[10000:18000]
    train_labels = set_labels[:8000] + set_labels[10000:18000]
    test = set[8000:10000] + set[18000::]
    test_labels = set_labels[8000:10000] + set_labels[18000::]

    np.save(f"Data/Discretized/Exp{i}/train.npy", np.array(train))
    np.save(f"Data/Discretized/Exp{i}/train_labels.npy", np.array(train_labels))
    np.save(f"Data/Discretized/Exp{i}/test.npy", np.array(test))
    np.save(f"Data/Discretized/Exp{i}/test_labels.npy", np.array(test_labels))
