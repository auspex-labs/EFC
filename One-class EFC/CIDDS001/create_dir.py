import os

os.makedirs("Data/", exist_ok=True)
os.makedirs("External_test/", exist_ok=True)

os.makedirs("Data/Discretized", exist_ok=True)
os.makedirs("Data/Non_discretized", exist_ok=True)
os.makedirs("Data/Results", exist_ok=True)
os.makedirs("External_test/Discretized", exist_ok=True)
os.makedirs("External_test/Non_discretized", exist_ok=True)

for exp in range(1, 11):
    os.makedirs(f"Data/Discretized/Exp{exp}/", exist_ok=True)
    os.makedirs(f"Data/Non_discretized/Exp{exp}/", exist_ok=True)
    os.makedirs(f"Data/Results/Exp{exp}/", exist_ok=True)
    os.makedirs(f"External_test/Discretized/Exp{exp}/", exist_ok=True)
    os.makedirs(f"External_test/Non_discretized/Exp{exp}/", exist_ok=True)
