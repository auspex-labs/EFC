import os

import numpy as np
import pandas as pd


def drop_duplicates(dict_dataset):
    data = pd.read_csv(f"Discretized_{dict_dataset}.csv")
    data.drop_duplicates(subset=data.columns[1:-1], inplace=True)
    np.save(f"Discretized_unique_{dict_dataset}.npy", data)


drop_duplicates("CICIDS17")
drop_duplicates("CICDDoS19")
