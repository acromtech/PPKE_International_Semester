import numpy as np
import matplotlib.pyplot as plt
from visualise import visualise_nonlin

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.metrics import accuracy_score

import sys
sys.path.append('./Machine_Learning/lab/lab02_logical_regression')  # Chemin vers le répertoire parent du module
from LogisticRegression import LogisticRegression as lr

def task1(X, Y, X_test, Y_test):
    """
    The dataset in rings.npz is a binary classification problem.
    If you do not yet have a working logistic regression implementation, you can use that of scikit learn; see the documentation and the user guide for more information.
    """
    
    def task1a(X, Y, X_test, Y_test):
        """
        a. Plot the data points with their classes (you can use the visualise.py provided (this is a new one)).
        """
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

    def task1b(X, Y, X_test, Y_test):
        """
        b. Try the classification with a linear classifier (eg. PLA or logistic regression), and print the training/test errors, and plot the resulting classifier.
        
        The analysis results indicate that the logistic regression model used for classifying the "rings" dataset demonstrates relatively modest performance.
            - Classification error on the training data: approximately 45.6% (0.45599999999999996)
            - Classification error on the test data: approximately 49% (0.49)

        The relatively high classification errors on both the training and test sets indicate that the model fails to effectively capture the underlying structure of the data.

        So, the logistic regression model appears to struggle in accurately modeling the relationship between features and labels in the "rings" dataset.
        It's possible that the classes in the "rings" dataset are not linearly separable, which limits the effectiveness of logistic regression in this context.
        To achieve better classification performance, it may be necessary to explore more complex models, such as **neural networks**, capable of capturing nonlinear relationships between features and labels.
        """ 
        # Train the linear classifier (Perceptron)
        #clf = Perceptron()  # You can also use LogisticRegression instead of Perceptron
        clf = LogisticRegression()
        clf.fit(X, Y.ravel())

        # Predict classes for the training and test data
        Y_train_pred = clf.predict(X)
        Y_test_pred = clf.predict(X_test)

        # Calculate training and test errors
        train_error = 1 - accuracy_score(Y, Y_train_pred) # Return 1 if match
        test_error = 1 - accuracy_score(Y_test, Y_test_pred)

        print("Training error:", train_error)
        print("Test error:", test_error)

        # Plot the resulting classifier
        def hypothesis(x):
            return clf.predict(x)

        fig, ax = visualise_nonlin(hypothesis, X, Y)
        ax.set_title('Linear Classifier (Logistic Regression)')
        plt.show()
    
    def task1c(X, Y, X_test, Y_test):
        """
        c. Try the classification with the norm of the original input samples (‖x‖2) added as a new feature. (Again, print the errors, plot the classifier.)
            ->  For the b. question, the classification errors are relatively high, with a training error of 45.6% and a test error of 49%. 
                This suggests that the linear model (logistic regression) struggles to capture the underlying structure of the "rings" dataset, indicating that the classes may not be linearly separable.
                But now with the **adding of the L2 norm** as a new feature seems to have improved the model's performance. **The classification errors decreased for both the training** (24.6%) and test (27%) sets. 
                This suggests that adding this additional feature allowed the model to **better capture the data's structure and generalize better to test data**.
        """
        # Calculate the L2 norm of each input sample
        X_train_norm = np.linalg.norm(X, axis=1).reshape(-1, 1)
        X_test_norm = np.linalg.norm(X_test, axis=1).reshape(-1, 1)

        # Append the L2 norm as a new feature to the dataset
        X_train_ext = np.concatenate((X, X_train_norm), axis=1)
        X_test_ext = np.concatenate((X_test, X_test_norm), axis=1)

        # Train logistic regression classifier
        clf = LogisticRegression()
        clf.fit(X_train_ext, Y.ravel())

        # Make predictions
        Y_train_pred = clf.predict(X_train_ext)
        Y_test_pred = clf.predict(X_test_ext)

        # Calculate classification errors
        train_error = 1 - accuracy_score(Y, Y_train_pred)
        test_error = 1 - accuracy_score(Y_test, Y_test_pred)

        # Print errors
        print("Training error:", train_error)
        print("Test error:", test_error)

        # Define a custom prediction function
        def predict_with_clf(x):
            return clf.predict(np.hstack((x, np.linalg.norm(x, axis=1, keepdims=True))))

        # Plot the classifier
        fig, ax = visualise_nonlin(predict_with_clf, X, Y)
        ax.set_title('Classifier with L2 norm as a new feature')
        plt.show()

    def task1d(X, Y, X_test, Y_test):
        """
        d. Try the classification with polynomials of degree 2 made from the inputs as new features.
        (Again, print the errors and plot the classifier.)
        If you have two variables x1 and x2, then the degree 2 polinomials are: x21, x1x2 and x22.
        Use these as new features.
            ->  Such a result indicate that the classes are well-separated in the feature space after the transformation induced by the polynomial features.
                The problem is highly linearly separable in the space defined by these polynomial features.
        """
        # Generate degree 2 polynomial features
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_train_poly = poly.fit_transform(X)
        X_test_poly = poly.transform(X_test)

        # Train logistic regression classifier
        clf = LogisticRegression()
        clf.fit(X_train_poly, Y.ravel())

        # Make predictions
        Y_train_pred = clf.predict(X_train_poly)
        Y_test_pred = clf.predict(X_test_poly)

        # Calculate classification errors
        train_error = 1 - accuracy_score(Y, Y_train_pred)
        test_error = 1 - accuracy_score(Y_test, Y_test_pred)

        # Print errors
        print("Training error:", train_error)
        print("Test error:", test_error)

        # Plot the classifier
        def predict_with_clf(x):
            x_poly = poly.transform(x)
            return clf.predict(x_poly)

        fig, ax = visualise_nonlin(predict_with_clf, X, Y)
        ax.set_title('Classifier with Degree 2 Polynomial Features')
        plt.show()

    task1a(X, Y, X_test, Y_test)
    task1b(X, Y, X_test, Y_test)
    task1c(X, Y, X_test, Y_test)
    task1d(X, Y, X_test, Y_test)
    
def task2():
    """
    **2. What is the $E_{out}$ for this hypothesis (using accuracy as an error measure)?**

    To calculate the error rate (E_out) for the hypothesis $ f(x) = \text{sign}(x_1^2 + x_2^2 - 1.1) $ using accuracy as the error measure, we need to determine how many points in the dataset are misclassified by this hypothesis.

    Given that the negative class is defined in the set $ \{x : x_1^2 + x_2^2 \leq 1\} $ and the positive class is defined in the set $ \{x : 1 < x_1^2 + x_2^2 \leq 2\} $, we can infer the following:
    - For points with $ x_1^2 + x_2^2 \leq 1 $, the true label is negative.
    - For points with $ 1 < x_1^2 + x_2^2 \leq 2 $, the true label is positive.

    Now, let's evaluate the classification performance of the hypothesis:
    - For points with $ x_1^2 + x_2^2 \leq 1 $, if $ x_1^2 + x_2^2 - 1.1 \leq 0 $, the hypothesis will correctly classify them as negative. Otherwise, it will misclassify them as positive.
    - For points with $ 1 < x_1^2 + x_2^2 \leq 2 $, if $ x_1^2 + x_2^2 - 1.1 > 0 $, the hypothesis will correctly classify them as positive. Otherwise, it will misclassify them as negative.

    Given the nature of the hypothesis, we can infer that it will misclassify all points that are within the circle of radius $ \sqrt{1.1} $ centered at the origin, and all points that are outside the circle of radius $ \sqrt{2.1} $ centered at the origin. 

    Thus, the error rate (E_out) for this hypothesis using accuracy as the error measure would be the proportion of points misclassified, which is the ratio of the area of the annulus between the two circles to the total area of the unit circle.

    Mathematically, this can be expressed as:

    $$ E_{out} = \frac{{\text{Area of the annulus}}}{\text{Total area of the unit circle}} = \frac{{\pi \cdot (2.1 - 1.1)}}{\pi \cdot 1^2} = \frac{1}{1} = 1 $$

    So, the error rate for this hypothesis is 1 or 100%. This indicates that all points in the dataset are misclassified by the hypothesis.
    """
    pass

def task4(X, Y):
    """
    When implementing the gradient calculation by hand,1 one sometimes makes mistakes.
    One easy way to verify your calculations is to compare the derivatives to derivatives computed by the method of finite differences. 
    Derived from the definition of the derivative, the estimation is:

    $$f'(x)=(f(x+\epsilon/2) - f(x-\epsilon/2)) / \epsilon$$

    For some suitable $\epsilon$: it shouldn't be too small to avoid numerical errors (eg. from rounding), but it should be small enough to approximate the derivative.
    For functions with vector input, this only estimates the partial derivative, and needs to be repeated for each dimension of the input.

    a. In your logistic regression implementation, put an estimation of the finite difference of the error (estimating the gradient). (Here, the $f$ in (1) is the 𝐸𝑖𝑛 with $w$ as the input.)
    Print out both values: the exact gradient ∇𝐸𝑖𝑛 and the finite difference.
    Modify the formula calculating the gradient so it has an error in it. 
    Compare the two values again.
    """
    # Train logistic regression
    model = lr()
    model.stochastic_gradient_descent(X, Y, learning_rate=0.001, num_iterations=1000)

data = np.load('./Machine_Learning/lab/lab03_non-linear_regression/data/rings.npz')
X, Y, X_test, Y_test = data['X'], data['Y'], data['X_test'], data['Y_test']

# task1(X,Y,X_test,Y_test)
# task2()
task4(X,Y)
