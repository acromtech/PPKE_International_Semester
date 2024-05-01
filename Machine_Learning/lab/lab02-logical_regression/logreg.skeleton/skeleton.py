from typing import NamedTuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from visualise import visualise

def sigmoid(h):
    # TODO
    pass

def task1():
    sigmoid_test_values = np.array([0., -1, 1, -10, 10, -1000, 1000])
    sigmoid(sigmoid_test_values)


class LogisticRegression:
    """
    Implements the logistic regression algorithm (binary classification model)
    """

    def predict(self, X: np.array) -> np.array:
        """
        Predicts class for each input in X
        """
        # TODO

    def predict_proba(self, X: np.array) -> np.array:
        """
        Predicts positive class probabilities for each input in X
        """
        # TODO

    def accuracy(self, X: np.array, Y: np.array) -> np.array:
        """
        Calculates the accuracy on the given dataset
        """
        # TODO

    def crossentropy_error(self, X: np.array, Y: np.array) -> np.array:
        """
        Calculates the (mean) cross-entropy error on the given dataset
        """
        # TODO

    def _error_gradient(self, X: np.array, Y: np.array,
                        w: np.array) -> np.array:
        """
        Calculate the (mean) gradient of the cross-entropy error (with respect
        to the parameters)
        """
        # TODO

    def train(self,
              X: np.array,
              Y: np.array,
              *,
              max_iteration: int = 100,
              batch_size: Optional[int] = None,
              step_size: float = 0.5,
              plot_every: int = 1,
              seed: int = 1):
        """
        Train the model on the given dataset
        """
        # TODO

