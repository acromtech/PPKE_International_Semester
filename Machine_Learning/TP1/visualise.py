import matplotlib.pyplot as plt
import numpy as np

def visualise(w, X, Y, block=True, fig=None, ax=None, title=None):
    """
    Visualises the dataset together with the separator

    # Parameters
    w (array-like): the model weights (a vector with 3 elements)
    X (array-like): an Nx2 matrix of input elements
    Y (array-like): an Nx1 matrix of labels
    block (bool): blocking show
    fig, ax: Matplotlib figure and Axes to plot on
    title (str): title of the plot
    """

    def separator_points(w):
        l = -w[2] / (w[0]**2 + w[1]**2)
        x0 = np.array([l * w[0], l * w[1]])
        v = np.array([-w[1], w[0]])
        v /= np.linalg.norm(v, ord=2)
        return np.stack([x0 + 5*v, x0 - 5*v])

    w = w.reshape((-1,))
    if not fig:
        fig, ax = plt.subplots()
    ax.cla()
    positive_samples = X[Y[:, 0] == 1, :]
    negative_samples = X[Y[:, 0] == -1, :]
    ax.plot(positive_samples[:, 0], positive_samples[:, 1], 'x', label='$+1$')
    ax.plot(negative_samples[:, 0], negative_samples[:, 1], 'o', label='$-1$')
    if np.linalg.norm(w) < 1e-10 or np.linalg.norm(w[:2]) < 1e-10:
        print('Warning: the norm of the weight vector is to small to plot.')
    else:
        s = separator_points(w)
        ax.plot(s[:, 0], s[:, 1], '-', label='separator')
    ax.legend()
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    if title:
        ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()
    plt.show(block=block)
    return fig, ax
