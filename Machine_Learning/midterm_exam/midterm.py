import numpy as np
import matplotlib.pyplot as plt

def load_dataset(file):
    data = np.load(file)
    return data['X'], data['Y'], data['X_test'], data['Y_test']

def feature_transformations(x):
    transformations = [
        np.sin(2 * x),
        np.exp(-(x - 2) ** 2),

        # It was the most strange and instable
            # Unvalid results like a division by 0 for severals values of x
        #np.exp(-x) / (x ** 2 + 1),

        # Modification try 1: Average good but not keep the same shape
        # np.log(np.abs(x) + 1), 

        # Modification try 2 : Better but we need to add a little something
        x ** 2, 

        # Modification try 3 : Mean and shape seems correct ... but i think x**2 was better
        # (x ** 2) + (x * 100), 

        np.sin(x) * np.cos(x)
    ]
    return transformations

def plot_feature_transformations(X_train):
    plt.figure(figsize=(12, 8))
    x_values = np.linspace(np.min(X_train), np.max(X_train), 100)
    transformations = feature_transformations(x_values)
    for i, transformation in enumerate(transformations):
        plt.plot(x_values, transformation, label=f'Transformation {i+1}')
    plt.xlabel('x')
    plt.ylabel('Transformed feature')
    plt.title('Feature Transformations on Training Dataset')
    plt.legend()
    plt.grid(True)
    plt.show()

def nonlinear_regression(X_train, Y_train, X_test, Y_test, alpha=0.01):
    # Apply feature transformations to training and test data
    X_train_transformed = np.column_stack(feature_transformations(X_train))
    X_test_transformed = np.column_stack(feature_transformations(X_test))

    # Regularization term
    I = np.eye(X_train_transformed.shape[1])
    regularization_term = alpha * I

    # Calculate weights using gradient ascent with regularization
    w = np.linalg.inv(X_train_transformed.T @ X_train_transformed + regularization_term) @ X_train_transformed.T @ Y_train

    # Calculate predicted outputs
    Y_train_predicted = X_train_transformed @ w
    Y_test_predicted = X_test_transformed @ w

    # Calculate mean squared errors
    mse_train = np.mean((Y_train - Y_train_predicted) ** 2)
    mse_test = np.mean((Y_test - Y_test_predicted) ** 2)

    return w, mse_train, mse_test, Y_train_predicted, Y_test_predicted


# a. Load datasets
X_train, Y_train, X_test, Y_test = load_dataset('./midterm.npz')

# b. Plot each feature transformation on the domain of the training dataset
plot_feature_transformations(X_train)

# c. Perform nonlinear regression on the datasets
w_train, mse_train, mse_test, Y_train_predicted, Y_test_predicted = nonlinear_regression(X_train, Y_train, X_test, Y_test)

# d. Print in-sample and test errors
print("In-sample Mean Squared Error:", mse_train)
print("Test Mean Squared Error:", mse_test)

# e. Plot the training and test datasets with both the ground truth and predicted outputs
plt.figure(figsize=(12, 6))
plt.scatter(X_train, Y_train, color='blue', label='Ground Truth (Train)')
plt.scatter(X_test, Y_test, color='green', label='Ground Truth (Test)')
plt.scatter(X_train, Y_train_predicted, color='red', label='Predicted (Train)')
plt.scatter(X_test, Y_test_predicted, color='orange', label='Predicted (Test)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Nonlinear Regression with Nonlinear Transformations')
plt.legend()
plt.grid(True)
plt.show()