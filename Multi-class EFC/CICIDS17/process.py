import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer, MaxAbsScaler, OrdinalEncoder

malicious_names = [
    "BENIGN",
    "Bot",
    "DDoS",
    "DoS GoldenEye",
    "DoS Hulk",
    "DoS Slowhttptest",
    "DoS slowloris",
    "FTP-Patator",
    "Heartbleed",
    "Infiltration",
    "PortScan",
    "SSH-Patator",
    "Web Attack",
]

# group continuos and symbolic features indexes
symbolic = [2, 79]
continuous = [x for x in range(79) if x not in symbolic]

# load data
for fold in range(1, 6):
    train = np.array(pd.read_csv(f"5-fold_sets/Raw/Sets{fold}/reduced_train.csv", header=None))
    test = np.array(pd.read_csv(f"5-fold_sets/Raw/Sets{fold}/test.csv", header=None))

    # encode symbolic features
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
    enc.fit(train[:, symbolic])
    train[:, symbolic] = enc.transform(train[:, symbolic])
    test[:, symbolic] = enc.transform(test[:, symbolic])
    test[:, symbolic] = np.nan_to_num(test[:, symbolic].astype("float"), nan=np.max(test[:, symbolic]) + 1)

    np.savetxt(f"5-fold_sets/Encoded/Sets{fold}/X_train", train[:, :-1], delimiter=",")
    np.savetxt(f"5-fold_sets/Encoded/Sets{fold}/y_train", train[:, -1], delimiter=",")
    np.savetxt(f"5-fold_sets/Encoded/Sets{fold}/X_test", test[:, :-1], delimiter=",")
    np.savetxt(f"5-fold_sets/Encoded/Sets{fold}/y_test", test[:, -1], delimiter=",")

    # normalize continuos features
    norm = MaxAbsScaler()
    norm.fit(train[:, continuous])
    train[:, continuous] = norm.transform(train[:, continuous])
    test[:, continuous] = norm.transform(test[:, continuous])

    np.savetxt(f"5-fold_sets/Normalized/Sets{fold}/X_train", train[:, :-1], delimiter=",")
    np.savetxt(f"5-fold_sets/Normalized/Sets{fold}/y_train", train[:, -1], delimiter=",")
    np.savetxt(f"5-fold_sets/Normalized/Sets{fold}/X_test", test[:, :-1], delimiter=",")
    np.savetxt(f"5-fold_sets/Normalized/Sets{fold}/y_test", test[:, -1], delimiter=",")

    # discretize continuos features
    disc = KBinsDiscretizer(n_bins=30, encode="ordinal", strategy="quantile")
    disc.fit(train[:, continuous])
    train[:, continuous] = disc.transform(train[:, continuous])
    test[:, continuous] = disc.transform(test[:, continuous])

    np.savetxt(f"5-fold_sets/Discretized/Sets{fold}/X_train", train[:, :-1], delimiter=",")
    np.savetxt(f"5-fold_sets/Discretized/Sets{fold}/y_train", train[:, -1], delimiter=",")
    np.savetxt(f"5-fold_sets/Discretized/Sets{fold}/X_test", test[:, :-1], delimiter=",")
    np.savetxt(f"5-fold_sets/Discretized/Sets{fold}/y_test", test[:, -1], delimiter=",")
