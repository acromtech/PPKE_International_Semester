import numpy as np
import matplotlib.pyplot as plt
import openpyxl

fname = 'kin1.xlsx'
test_ratio = 0.8
n = 2

# Read data from Excel file
workbook = openpyxl.load_workbook(fname)
sheet = workbook.active
data = np.array([[cell.value for cell in row] for row in sheet.iter_rows()])[2:, 2:]

data[np.isnan(data)] = 0  # Data cleaning

Num_total = len(data)
Num_test = round(Num_total * test_ratio)

data_calibration = data[2:Num_total - Num_test, :]
data_test = data[Num_total - Num_test:, :]

CovData = np.cov(data_calibration, rowvar=False)
_, V = np.linalg.eig(CovData)
PM = V[:, :n]

C = np.zeros((Num_test, n))

for k in range(Num_test):
    S = data_test[k, :]
    C[k, :] = S @ PM

    if n == 2:
        plt.plot(C[k, 0], C[k, 1], 'b.', markersize=20)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.pause(0.03)

    elif n == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(C[k, 0], C[k, 1], C[k, 2], 'b.', markersize=10)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.pause(0.01)

plt.show()
