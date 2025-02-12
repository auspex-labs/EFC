import os

import numpy as np
import pandas as pd


def pre_process(day, files, concat):
    for file in files:
        data = pd.read_csv(f"Pre_processed/{day}/{file}.csv")
        concat = pd.concat([concat, data], axis=0)

    concat.iloc[:, 0] = [x for x in range(0, concat.shape[0])]
    return concat


files_day1 = [
    "DrDoS_LDAP",
    "DrDoS_MSSQL",
    "DrDoS_NetBIOS",
    "DrDoS_NTP",
    "DrDoS_SNMP",
    "DrDoS_SSDP",
    "DrDoS_UDP",
    "Syn",
    "TFTP",
    "UDPLag",
]
files_day2 = ["LDAP", "MSSQL", "NetBIOS", "Portmap", "Syn", "UDP", "UDPLag"]

concat = pd.read_csv("Pre_processed/{}/{}.csv".format("01-12", "DrDoS_DNS"))
concat = pre_process("01-12", files_day1, concat)
concat = pre_process("03-11", files_day2, concat)

concat.to_csv("Pre_processed.csv", index=False)
