import numpy as np
import matplotlib.pyplot as plt

def visualise_nonlin(fun, X, Y, resolution=50):
    """
    Visualises the dataset together with the separator

    Example usage:

        def hypothesis(x):
            return np.sign(x @ w)
        visualise_nonlin(hypothesis, X, Y)

    # Parameters
    fun (callable): a function that outputs, for each sample in the argument
        array X, the class it belongs to (this is your hypothesis in Python
        function form). The class should be -1 or +1.
    X (array-like): an Nx2 matrix of input elements
    Y (array-like): an Nx1 matrix of labels
    """
    fig, ax = plt.subplots()
    # Data
    plot_data(X, Y, ax)
    # Separator
    x = np.linspace(-5, 5, resolution)
    xx, yy = np.meshgrid(x, x)
    grid = np.c_[xx.flatten(), yy.flatten()]
    decisions = fun(grid)
    ax.contour(
        xx, yy, decisions.reshape(resolution, resolution), levels=np.array([0]))
    # Annotations
    ax.legend()
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.grid(True)
    fig.tight_layout()
    return fig, ax

def plot_data(X, Y, ax):
    positive_samples = X[Y[:, 0] == 1, :]
    negative_samples = X[Y[:, 0] == -1, :]
    ax.plot(positive_samples[:, 0], positive_samples[:, 1], 'x', label='$+1$')
    ax.plot(negative_samples[:, 0], negative_samples[:, 1], 'o', label='$-1$')
