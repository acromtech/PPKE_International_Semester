import numpy as np
import matplotlib.pyplot as plt
from visualise import visualise_nonlin

# Load the dataset
data = np.load('./nonlin.data/rings.npz')
X, Y = data['X'], data['Y']

# Define a hypothesis function (your model)
def hypothesis(x):
    # Here, you need to define your hypothesis function
    # This function should output the class (-1 or +1) for each sample in x
    # You can use any nonlinear model or algorithm you want
    # For example, you can use logistic regression or a neural network
    # For now, let's assume it's a simple linear classifier
    w = np.array([1, -1])  # Example weights
    return np.sign(x @ w)

# Visualize the dataset with the provided visualise_nonlin function
fig, ax = visualise_nonlin(hypothesis, X, Y)
ax.set_title('Data points with classes')
plt.show()