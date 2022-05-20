import os

os.mkdir("Data/")
os.mkdir("External_test/")

os.mkdir("Data/Discretized")
os.mkdir("Data/Non_discretized")
os.mkdir("Data/Results")
os.mkdir("External_test/Discretized")
os.mkdir("External_test/Non_discretized")

for exp in range(1, 11):
    os.mkdir(f"Data/Discretized/Exp{exp}/")
    os.mkdir(f"Data/Non_discretized/Exp{exp}/")
    os.mkdir(f"Data/Results/Exp{exp}/")
    os.mkdir(f"External_test/Discretized/Exp{exp}/")
    os.mkdir(f"External_test/Non_discretized/Exp{exp}/")
