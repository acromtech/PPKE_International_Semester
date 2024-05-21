from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    """
    Implements the logistic regression algorithm (binary classification model)
    """

    def sigmoid(self, h: np.array) -> np.array:
        """
        Compute the sigmoid function for the given input.
        
        Parameters:
        - h (np.array): Input array.
        
        Returns:
        - np.array: Sigmoid function applied to each element of the input array.
        """
        return 1 / (1 + np.exp(-h))

    def predict(self, X: np.array, weights: np.array) -> np.array:
        """
        Predicts class labels for each input in X.
        
        Parameters:
        - X (np.array): Input features.
        - weights (np.array): Model weights.
        
        Returns:
        - np.array: Predicted class labels.
        """
        return np.where(self.predict_proba(X, weights) >= 0.5, 1, -1)

    def predict_proba(self, X: np.array, weights: np.array) -> np.array:
        """
        Predicts the probability of the positive class for each input in X.
        
        Calculates the dot product of the input features (X) and the model weights (weights), which gives us the linear combination of features. 
#       Then, we applies the sigmoid function (sigmoid) to the linear combination to obtain the probabilities.
        
        Parameters:
        - X (np.array): Input features.
        - weights (np.array): Model weights.
        
        Returns:
        - np.array: Predicted probabilities of the positive class.
        """
        return self.sigmoid(X @ weights)

    def accuracy(self, X: np.array, Y: np.array, weights: np.array) -> float:
        """
        Calculate the accuracy of the model on the given dataset.
        
        Parameters:
        - X (np.array): Input features.
        - Y (np.array): True labels.
        - weights (np.array): Model weights.
        
        Returns:
        - float: Accuracy of the model.
        """
        y_pred = self.predict(X, weights)
        return np.mean(y_pred == Y)

    def crossentropy_error(self, X: np.array, Y: np.array, weights: np.array) -> float:
        """
        Calculate the cross-entropy error of the model on the given dataset.
        
        Parameters:
        - X (np.array): Input features.
        - Y (np.array): True labels.
        - weights (np.array): Model weights.
        
        Returns:
        - float: Cross-entropy error.
        """
        y_pred_proba = self.predict_proba(X, weights)
        return -np.mean(Y * np.log(y_pred_proba) + (1 - Y) * np.log(1 - y_pred_proba))

    def _error_gradient(self, X: np.array, Y: np.array, weights: np.array) -> np.array:
        """
        Calculate the gradient of the error function with respect to the model parameters (weights) using the formula for the gradient of the cross-entropy loss. 
        
        It computes the dot product of the transpose of the input features (X.T) and the difference between the predicted probabilities and the true labels (y_pred - y_true), then divides by the number of samples to obtain the average gradient.
        During each iteration of gradient descent, the error gradient is computed using this function, and the model parameters are updated in the opposite direction of the gradient to minimize the error function.
        
        Parameters:
        - X (np.array): Input features.
        - Y (np.array): True labels.
        - weights (np.array): Model weights.
        
        Returns:
        - np.array: Gradient of the cross-entropy error.
        """
        y_pred_proba = self.predict_proba(X, weights)
        return np.dot(X.T, (y_pred_proba - Y)) / len(Y)
    
    def _error_gradient2(self, X: np.array, Y: np.array, weights: np.array, epsilon: float = 1e-5) -> np.array:
        """
        Calculate the gradient of the error function with respect to the model parameters (weights) using the formula for the gradient of the cross-entropy loss. 
            
        Parameters:
        - X (np.array): Input features.
        - Y (np.array): True labels.
        - weights (np.array): Model weights.
        - epsilon (float): Small value to compute finite differences.
            
        Returns:
        - np.array: Gradient of the cross-entropy error.
        """
        y_pred_proba = self.predict_proba(X, weights)
        gradient_exact = np.dot(X.T, (y_pred_proba - Y)) / len(Y)

        # Estimation using finite differences
        gradient_estimated = np.zeros_like(weights)
        for i in range(len(weights)):
            weights_plus = weights.copy()
            weights_plus[i] += epsilon / 2
            error_plus = self.crossentropy_error(X, Y, weights_plus)

            weights_minus = weights.copy()
            weights_minus[i] -= epsilon / 2
            error_minus = self.crossentropy_error(X, Y, weights_minus)

            gradient_estimated[i] = (error_plus - error_minus) / epsilon

        print("Exact Gradient:", gradient_exact)
        print("Gradient Estimated with Finite Differences:", gradient_estimated)

        return gradient_exact  # Return the exact gradient

    
    def batch_gradient_descent(self, X, Y, learning_rate, num_iterations):
        num_features = X.shape[1]
        weights = np.random.rand(num_features)
        
        plt.figure(1)
        
        for i in range(num_iterations):
            error = self.crossentropy_error(X, Y, weights)
            gradient = self._error_gradient(X, Y, weights)
            weights -= learning_rate * gradient
            if i % 100 == 0: 
                print(f"Iteration {i}: Error = {error}")

                # Plot the data points
                plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
                plt.xlabel('Feature 1')
                plt.ylabel('Feature 2')
                plt.title(f"Iteration {i}: Error = {error}")

                # Plot the decision boundary
                plot_x = np.array([np.min(X[:, 0]), np.max(X[:, 0])])
                plot_y = (-1 / weights[1]) * (weights[0] * plot_x)
                plt.plot(plot_x, plot_y, '-r')
                
                # Set the axes limits based on the min and max points
                plt.xlim(np.min(X[:, 0]), np.max(X[:, 0]))
                plt.ylim(np.min(X[:, 1]), np.max(X[:, 1]))
                
                plt.pause(0.1)
                plt.clf()

        plt.close()
        return weights
    
    def stochastic_gradient_descent(self, X, Y, learning_rate, num_iterations, batch_size=None):
        num_samples, num_features = X.shape
        weights = np.random.rand(num_features)

        if batch_size is None or batch_size >= num_samples:
            # Full batch descent
            batch_size = num_samples

        num_batches = num_samples // batch_size

        plt.figure()

        for i in range(num_iterations):
            # Shuffle the data before each epoch
            indices = np.random.permutation(num_samples)
            X_train_shuffled = X[indices]
            y_train_shuffled = Y[indices]

            for j in range(num_batches):
                # Select batch
                start = j * batch_size
                end = min((j + 1) * batch_size, num_samples)
                X_batch = X_train_shuffled[start:end]
                y_batch = y_train_shuffled[start:end]
                error = self.crossentropy_error(X_batch, y_batch, weights)
                gradient = self._error_gradient(X_batch, y_batch, weights)
                weights -= learning_rate * gradient

            if i % 100 == 0:
                print(f"Iteration {i}: Error = {error}")
                
                # Plot the data points
                plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
                plt.xlabel('Feature 1')
                plt.ylabel('Feature 2')
                plt.title(f"Iteration {i}: Error = {error}")

                # Plot the decision boundary
                plot_x = np.array([np.min(X[:, 0]), np.max(X[:, 0])])
                plot_y = (-1 / weights[1]) * (weights[0] * plot_x)
                plt.plot(plot_x, plot_y, '-r')
                
                # Set the axes limits based on the min and max points
                plt.xlim(np.min(X[:, 0]), np.max(X[:, 0]))
                plt.ylim(np.min(X[:, 1]), np.max(X[:, 1]))
                
                plt.pause(0.1)
                plt.clf()

        plt.close()

        return weights
    
    def stochastic_gradient_descent2(self, X, Y, learning_rate, num_iterations):
        d = X.shape[1]
        w = np.random.randn(d)

        plt.figure(figsize=(18, 6))

        for i in range(num_iterations):
            train_error = self.crossentropy_error(X, Y, w)
            grad = self._error_gradient(X, Y, w)
            w -= learning_rate * grad

            if (i + 1) % 100 == 0:
                print(f"Iteration {i + 1}: Error = {train_error}")
                train_scores = np.dot(X, w)
                sigmoid_values = self.sigmoid(train_scores)
                
                plt.subplot(1, 3, 1)
                plt.plot(train_scores, sigmoid_values, '*')
                plt.title('Sigmoid')
                plt.xlabel('Score (s)')
                plt.ylabel('Sigmoid Value')

                plt.subplot(1, 3, 2)
                plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.coolwarm, s=10, edgecolors='b')
                plot_x = np.array([np.min(X[:, 0]), np.max(X[:, 0])])
                plot_y = (-1 / w[1]) * (w[0] * plot_x)
                plt.plot(plot_x, plot_y, '-r')
                plt.title('Linear Separator')
                plt.xlabel('Feature 1')
                plt.ylabel('Feature 2')

                plt.subplot(1, 3, 3)
                plt.scatter(train_scores, Y, c=Y, cmap=plt.cm.coolwarm, s=10, edgecolors='b')
                plt.title('Ground Truth vs. Score')
                plt.xlabel('Score (s)')
                plt.ylabel('Ground Truth (y)')

                # Set the axes limits based on the min and max points
                plt.xlim(np.min(X[:, 0]), np.max(X[:, 0]))
                plt.ylim(np.min(X[:, 1]), np.max(X[:, 1]))

                plt.tight_layout()
                plt.pause(0.1)
                plt.clf()
        plt.close()

        return w
    
    def stochastic_gradient_descent3(self, X, Y, learning_rate, num_iterations):
        d = X.shape[1]
        w = np.random.randn(d)

        plt.figure(figsize=(18, 6))

        for i in range(num_iterations):
            train_error = self.crossentropy_error(X, Y, w)
            grad = self._error_gradient2(X, Y, w)
            w -= learning_rate * grad

            if (i + 1) % 100 == 0:
                print(f"Iteration {i + 1}: Error = {train_error}")
                train_scores = np.dot(X, w)
                sigmoid_values = self.sigmoid(train_scores)
                
                plt.subplot(1, 3, 1)
                plt.plot(train_scores, sigmoid_values, '*')
                plt.title('Sigmoid')
                plt.xlabel('Score (s)')
                plt.ylabel('Sigmoid Value')

                plt.subplot(1, 3, 2)
                plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.coolwarm, s=10, edgecolors='b')
                plot_x = np.array([np.min(X[:, 0]), np.max(X[:, 0])])
                plot_y = (-1 / w[1]) * (w[0] * plot_x)
                plt.plot(plot_x, plot_y, '-r')
                plt.title('Linear Separator')
                plt.xlabel('Feature 1')
                plt.ylabel('Feature 2')

                plt.subplot(1, 3, 3)
                plt.scatter(train_scores, Y, c=Y, cmap=plt.cm.coolwarm, s=10, edgecolors='b')
                plt.title('Ground Truth vs. Score')
                plt.xlabel('Score (s)')
                plt.ylabel('Ground Truth (y)')

                # Set the axes limits based on the min and max points
                plt.xlim(np.min(X[:, 0]), np.max(X[:, 0]))
                plt.ylim(np.min(X[:, 1]), np.max(X[:, 1]))

                plt.tight_layout()
                plt.pause(0.1)
                plt.clf()
        plt.close()

        return w

    def train(self,
              X: np.array,
              Y: np.array,
              weights: np.array,
              *,
              max_iteration: int = 100,
              batch_size: Optional[int] = None,
              step_size: float = 0.5,
              plot_every: int = 1,
              seed: int = 1):
        """
        Train the logistic regression model on the given dataset using gradient descent.
        
        Parameters:
        - X (np.array): Input features.
        - Y (np.array): True labels.
        - weights (np.array): Initial model weights.
        - max_iteration (int): Maximum number of iterations for training.
        - batch_size (Optional[int]): Size of mini-batches for stochastic gradient descent.
        - step_size (float): Learning rate for gradient descent.
        - plot_every (int): Plot decision boundary every `plot_every` iterations. Set to 0 to disable plotting.
        - seed (int): Seed for random number generator.
        """
        np.random.seed(seed)
        num_samples, num_features = X.shape

        for i in range(max_iteration):
            # Shuffle the data before each epoch
            indices = np.random.permutation(num_samples)
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]

            for j in range(0, num_samples, batch_size or num_samples):
                X_batch = X_shuffled[j:j + batch_size]
                Y_batch = Y_shuffled[j:j + batch_size]

                gradient = self._error_gradient(X_batch, Y_batch, weights)
                weights -= step_size * gradient

            if plot_every and (i + 1) % plot_every == 0:
                train_error = self.crossentropy_error(X, Y, weights)
                print(f"Iteration {i + 1}: Error = {train_error}")

                # Optionally plot decision boundary
                if plot_every > 0:
                    plt.figure()
                    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
                    plt.xlabel('Feature 1')
                    plt.ylabel('Feature 2')
                    plot_x = np.array([np.min(X[:, 0]), np.max(X[:, 0])])
                    plot_y = (-1 / weights[1]) * (weights[0] * plot_x)
                    plt.plot(plot_x, plot_y, '-r')
                    plt.title(f"Iteration {i + 1}: Error = {train_error}")
                    plt.show()