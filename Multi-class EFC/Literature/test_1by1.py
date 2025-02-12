import pickle
import resource
import sys

import numpy as np
import pandas as pd
from classification_functions import *

sys.path.append("../../../EFC")


def predict(folder, attacks):
    n_bins = 30
    train_labels = np.load(f"{folder}/Models/train_labels.npy", allow_pickle=True)
    preprocessor = pickle.load(open(f"{folder}/Models/preprocessor.pkl", mode="rb"))
    for i in attacks:
        test = pd.read_csv(f"{folder}/1by1_sets/{i}.csv", header=None).values
        test = preprocessor.transform(test).astype("int64")
        print(test.shape)

        h_i = np.load(f"{folder}/Models/h_i.npy", allow_pickle=True)
        couplings = np.load(f"{folder}/Models/couplings.npy", allow_pickle=True)
        cutoffs = np.load(f"{folder}/Models/cutoffs.npy", allow_pickle=True)

        y_pred, y_score = MultiClassPredict(
            np.array(test[:, :-1]),
            h_i,
            couplings,
            cutoffs,
            n_bins,
            np.unique(train_labels),
        )

        np.save(f"{folder}/Results/1by1/EFC_{i}.npy", [y_pred, y_score])


def main():
    unknown = [
        "DoS-Slowloris",
        "Heartbleed-Port",
        "Infiltration",
        "Botnet",
        "DoS-Slowhttptest",
        "WebAttacks",
    ]
    predict("Data", unknown)


def memory_limit():
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 * 0.8, hard))


def get_memory():
    with open("/proc/meminfo") as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ("MemFree:", "Buffers:", "Cached:"):
                free_memory += int(sline[1])
    return free_memory


if __name__ == "__main__":
    memory_limit()  # Limitates maximun memory usage to half
    try:
        main()
    except MemoryError:
        print("Memory error")
        sys.stderr.write("\n\nERROR: Memory Exception\n")
        sys.exit(1)
