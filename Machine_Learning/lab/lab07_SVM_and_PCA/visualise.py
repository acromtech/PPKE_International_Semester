import numpy as np
import matplotlib.pyplot as plt

def visualise_nonlin(fun, X, Y, resolution=50, annotation=None, ax=None):
    """
    Visualises the dataset together with the separator and margins

    The hypothesis function should give 0 for the decision boundary, and -1 and
    +1 for the margin boundaries on the negative and the positive sides,
    respectively.

    Example usage:

        w = np.array([0.42, 0.24])
        b = 0.042
        def hypothesis(x):
            return x @ w + b
        visualise_nonlin(hypothesis, X, Y)

    # Parameters
    fun (callable): a function that outputs, for each sample in the argument
        array X, the class it belongs to (this is your hypothesis in Python
        function form). The class should be -1 or +1.
    X (array-like): an Nx2 matrix of input elements
    Y (array-like): an Nx1 matrix of labels
    annotation (list of optional strings): a list of short texts to print with
        each point (in X). If None for a point, no text will be printed.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    # Data
    plot_data(X, Y, ax)
    # Separator
    x = np.linspace(-5, 5, resolution)
    xx, yy = np.meshgrid(x, x)
    grid = np.c_[xx.ravel(), yy.ravel()]
    decisions = fun(grid)
    ax.contour(
        xx,
        yy,
        decisions.reshape(resolution, resolution),
        levels=np.array([-1, 0, 1]),
        linestyles=['--', '-', '--'])
    # Annotations
    ax.legend()
    if annotation:
        assert len(annotation) == X.shape[0]
        for p, a in zip(X, annotation):
            if a:
                ax.text(p[0], p[1], a)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.grid(True)
    fig.tight_layout()
    return fig, ax

def plot_data(X, Y, ax, name=''):
    positive_samples = X[Y == 1, :]
    negative_samples = X[Y == -1, :]
    ax.plot(
        positive_samples[:, 0],
        positive_samples[:, 1],
        'x',
        label=name + ' $+1$')
    ax.plot(
        negative_samples[:, 0],
        negative_samples[:, 1],
        'o',
        label=name + ' $-1$')
