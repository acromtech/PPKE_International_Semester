import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds, LinearConstraint

def task1(path_file_svm_linear):
    """
    Implement the SVM algorithm using the unconstrained optimisation equations from the lectures
    and the scipy.optimizers.minimize method in Scipy, which is a black box optimiser.
    For this, you will need to implement a function that returns the value of the objective given
    the current alpha vector. See more in the documentation. Calculate the Q matrix beforehand, do
    not calculate it in every call of this function
    """
    # Generic functions ------------------------------------
    def gen_objective(X, Y, C):
        N = len(X)
        Q = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                Q[i, j] = Y[i] * Y[j] * np.dot(X[i], X[j])

        def objective(alpha):
            return 0.5 * np.dot(alpha, np.dot(Q, alpha)) - np.sum(alpha)

        return objective

    def train(X, Y, C):
        fun = gen_objective(X, Y, C)
        N = len(X)
        alpha0 = np.zeros(N)  # Initial guess for alphas
        bounds = Bounds(0, C)  # Bounds for alpha_i

        # Construct constraint matrix and bounds
        A_eq = Y.reshape(1, -1)  # reshape Y to a row vector
        b_eq = np.array([0])

        constraint = LinearConstraint(A_eq, lb=b_eq, ub=b_eq)

        res = minimize(fun=fun, x0=alpha0, bounds=bounds, constraints=constraint)
        if not res.success:
            print('Training failed: ', res.message)
            return None
        return res.x

    def decision_boundary(alpha, X, Y, C):
        support_vector_index = np.where((alpha > 1e-5) & (alpha < C))[0][0]  # Selecting a support vector on the margin boundary
        w = np.dot((alpha * Y), X)
        b = Y[support_vector_index] - np.dot(X[support_vector_index], w)
        return w, b
    
    # Tasks ------------------------------------------------
    def task1a(X, Y):
        """
        a. Using the algorithm, classify the dataset found in svm_linear.npz. 
        Use C values of 1, 10 and 100.
        """
        print("Task 1a: Classifying the dataset...")
        C_values = [1, 10, 100]
        for C in C_values:
            alpha = train(X, Y, C)

            # Count nonzero elements in alpha (considering rounding/computational errors)
            num_support_vectors = np.sum(np.abs(alpha) > 1e-5)  # Adjust the threshold based on your requirements
            print("Number of support vectors (C={}): {}".format(C, num_support_vectors))

    def task1b(X, Y):
        """
        b. How many nonzero (accounting for rounding/computational errors) elements does the alpha vector have? 
        I.e. how many support vectors did the algorithm find?
        """
        print("\nTask 1b: Counting nonzero elements in the alpha vector...")
        C_values = [1, 10, 100]
        for C in C_values:
            alpha = train(X, Y, C)
            num_support_vectors = np.sum(np.abs(alpha) > 1e-5)
            print("Number of support vectors (C={}): {}".format(C, num_support_vectors))

    def task1c(X, Y):
        """
        c. Plot the data and the separator line (as usual), and mark the support vectors on your plot. 
        What are the scores (i.e. y(wT x + b)) for these samples?
        """
        print("\nTask 1c: Plotting the data and the separator line...")
        C_values = [1, 10, 100]
        for C in C_values:
            alpha = train(X, Y, C)
            w, b = decision_boundary(alpha, X, Y, C)

            # Plot the data points
            plt.figure()
            plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired, edgecolors='k', label='Data')

            # Plot the decision boundary
            slope = -w[0] / w[1]
            intercept = -b / w[1]
            x_plot = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
            y_plot = slope * x_plot + intercept
            plt.plot(x_plot, y_plot, '-r', label='Separator')

            # Mark support vectors
            support_vectors = X[alpha > 1e-5]
            support_alphas = alpha[alpha > 1e-5]
            plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=100, facecolors='none', edgecolors='g', label='Support Vectors')

            # Annotate support vectors with their corresponding alphas
            for i, (x, y) in enumerate(support_vectors):
                plt.annotate('α = {:.2f}'.format(support_alphas[i]), (x, y), textcoords="offset points", xytext=(10,-10), ha='center')

            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.title('SVM with C = {}'.format(C))
            plt.legend()
            plt.show()

    def task1d(X, Y):
        """
        d. Print the alpha is for the support vectors on your plot.
        """
        print("\nTask 1d: Printing alpha values for support vectors...")
        C_values = [1, 10, 100]
        for C in C_values:
            alpha = train(X, Y, C)
            support_alphas = alpha[alpha > 1e-5]
            print("Alpha values for support vectors (C={}):".format(C), support_alphas)

    def task1e(X, Y):
        """
        e. Compare the result of the SVM with the PLA and the Logistic Regression. 
        Which one had the best generalisation (in this case)?
        """
        print("\nTask 1e: Comparing SVM with other classifiers...")
        from sklearn.svm import SVC
        from sklearn.linear_model import Perceptron, LogisticRegression
        from sklearn.metrics import accuracy_score

        # SVM with different C values
        C_values = [1, 10, 100]
        for C in C_values:
            svm = SVC(kernel='linear', C=C)
            svm.fit(X, Y)
            svm_accuracy = accuracy_score(Y, svm.predict(X))
            print("SVM accuracy (C={}): {:.2f}".format(C, svm_accuracy))

        # Perceptron
        perceptron = Perceptron(max_iter=1000)
        perceptron.fit(X, Y)
        perceptron_accuracy = accuracy_score(Y, perceptron.predict(X))
        print("Perceptron accuracy:", perceptron_accuracy)

        # Logistic Regression
        logistic_regression = LogisticRegression(max_iter=1000)
        logistic_regression.fit(X, Y)
        logistic_accuracy = accuracy_score(Y, logistic_regression.predict(X))
        print("Logistic Regression accuracy:", logistic_accuracy)

    def task1e_numpy(X, Y):
        """
        e. Compare the result of the SVM with the PLA and the Logistic Regression using only NumPy. 
        Which one had the best generalisation (in this case)?
        """
        print("\nTask 1e: Comparing SVM, Perceptron, and Logistic Regression (using only NumPy)...")
        
        # PLA implementation
        def pla(X, Y, max_iterations=1000):
            N, d = X.shape
            w = np.zeros(d)
            converged = False
            iterations = 0
            
            while not converged and iterations < max_iterations:
                converged = True
                for i in range(N):
                    if Y[i] * np.dot(w, X[i]) <= 0:
                        w += Y[i] * X[i]
                        converged = False
                iterations += 1
            
            return w

        # Logistic Regression implementation
        def sigmoid(z):
            return 1 / (1 + np.exp(-z))

        def logistic_regression(X, Y, lr=0.01, epochs=1000):
            N, d = X.shape
            w = np.zeros(d)
            
            for epoch in range(epochs):
                y_pred = sigmoid(np.dot(X, w))
                gradient = np.dot(X.T, (Y - y_pred))
                w += lr * gradient
            
            return w

        # Test PLA and Logistic Regression
        pla_w = pla(X, Y)
        logistic_w = logistic_regression(X, Y)

        # Evaluate performance
        def accuracy(w, X, Y):
            y_pred = np.sign(np.dot(X, w))
            return np.mean(y_pred == Y)

        C_values = [1, 10, 100]
        for C in C_values:
            alpha = train(X, Y, C)
            w, b = decision_boundary(alpha, X, Y, C)
            svm_accuracy = accuracy(w, X, Y)
            print("SVM accuracy (C={}): {:.2f}".format(C, svm_accuracy))

        pla_accuracy = accuracy(pla_w, X, Y)
        print("PLA accuracy:", pla_accuracy)

        logistic_accuracy = accuracy(logistic_w, X, Y)
        print("Logistic Regression accuracy:", logistic_accuracy)

    data = np.load(path_file_svm_linear)
    X = data['X']
    Y = data['Y']
    task1a(X, Y)
    task1b(X, Y)
    task1c(X, Y)
    task1d(X, Y)
    task1e(X, Y)
    task1e_numpy(X, Y)

def task2(path_file_ring):
    """
    Using the kernel trick, classify the dataset found in rings.npz. 
    I would suggest using a third degree polynomial kernel for this. 
    What is the test/train accuracy of the classification? 
    Plot the data and the separator curve.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score

    print("\nTask 2: Classifying the dataset using a polynomial kernel...")

    # Load data
    data = np.load(path_file_ring)
    X = data['X']
    Y = data['Y']
    
    # Split data into train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Train SVM with polynomial kernel
    svm = SVC(kernel='poly', degree=3)
    svm.fit(X_train, Y_train)
    
    # Predict on test set
    Y_pred = svm.predict(X_test)
    
    # Calculate test accuracy
    test_accuracy = accuracy_score(Y_test, Y_pred)
    print("Test accuracy:", test_accuracy)
    
    # Plot data and separator curve
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired, edgecolors='k', label='Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    # Create grid for plotting separator curve
    h = .02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, cmap=plt.cm.Paired)
    plt.title('SVM with Polynomial Kernel (Degree 3)')
    plt.legend()
    plt.show()

def task3(path_file_PCA):
    """
    Perform PCA on the given dataset and analyze the results.
    
    Given the dataset PCA.npz that was generated by applying a linear transformation and some
    noise to a 2D dateset; apply the PCA algorithm to find the two-dimensional factors underlying
    the data!
    The final orientation should be apparent from visualising (a scatterplot of) the 2D data2. What
    was the original transformation matrix used to create the transformed dataset?
    """
    def load_data(file_path):
        """Load the dataset from the given file path."""
        data = np.load(file_path)
        X = data['X']
        return X

    def pca(X):
        """Perform Principal Component Analysis (PCA) on the given dataset."""
        # Center the data
        X_centered = X - np.mean(X, axis=0)
        
        # Compute the covariance matrix
        covariance_matrix = np.cov(X_centered.T)
        
        # Perform eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        
        # Sort eigenvalues and eigenvectors
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Project the data onto the principal components
        X_pca = np.dot(X_centered, eigenvectors)
        
        return X_pca, eigenvectors

    def visualize_data(X):
        """Visualize the transformed 2D data."""
        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], X[:, 1], c='blue', alpha=0.5)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('Transformed 2D Data')
        plt.grid(True)
        plt.show()

    def original_transformation_matrix(X, X_pca):
        """Find the original transformation matrix used to create the transformed dataset."""
        # Recover the original transformation matrix
        transformation_matrix = np.linalg.lstsq(X_pca, X, rcond=None)[0]
        return transformation_matrix
    
    print("\nTask 3: Performing Principal Component Analysis (PCA)...")

    # Step 1: Load the dataset
    X = load_data(path_file_PCA)
    
    # Step 2: Perform PCA
    X_pca, eigenvectors = pca(X)
    
    # Step 3: Visualize the transformed 2D data
    visualize_data(X_pca)
    
    # Step 4: Find the original transformation matrix
    transformation_matrix = original_transformation_matrix(X, X_pca)
    print("Original Transformation Matrix:")
    print(transformation_matrix)

task1('./Machine_Learning/lab/lab07_SVM_and_PCA/data/svm_linear.npz')
task2('./Machine_Learning/lab/lab03_non-linear_regression/data/rings.npz')
task3('./Machine_Learning/lab/lab07_SVM_and_PCA/data/PCA.npz')

"""
MY OUTPUT :

Task 1a: Classifying the dataset...
Number of support vectors (C=1): 7
Number of support vectors (C=10): 5
Number of support vectors (C=100): 4

Task 1b: Counting nonzero elements in the alpha vector...
Number of support vectors (C=1): 7
Number of support vectors (C=10): 5
Number of support vectors (C=100): 4

Task 1c: Plotting the data and the separator line...

Task 1d: Printing alpha values for support vectors...
Alpha values for support vectors (C=1): [0.46025675 1.         1.         0.5423279  0.08207115 1.
 1.        ]
Alpha values for support vectors (C=10): [ 5.11461895  0.82177917 10.          5.93639812 10.        ]
Alpha values for support vectors (C=100): [ 26.22407443 100.          88.71719256  37.50688187]

Task 1e: Comparing SVM with other classifiers...
SVM accuracy (C=1): 0.94
SVM accuracy (C=10): 0.94
SVM accuracy (C=100): 0.94
Perceptron accuracy: 0.9375
Logistic Regression accuracy: 0.9375

Task 1e: Comparing SVM, Perceptron, and Logistic Regression (using only NumPy)...
SVM accuracy (C=1): 0.62
SVM accuracy (C=10): 0.62
SVM accuracy (C=100): 0.62
PLA accuracy: 0.75
Logistic Regression accuracy: 0.625

Task 2: Classifying the dataset using a polynomial kernel...
C:\Users\user\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\sklearn\utils\validation.py:1300: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
Test accuracy: 0.71

Task 3: Performing Principal Component Analysis (PCA)...
Original Transformation Matrix:
[[ 0.45674238 -0.44456908 -0.0347302  -0.10614052  0.23814211 -0.23891002
   0.03956564  0.07595089 -0.67278467 -0.08665956]
 [-0.10485652 -0.64274632  0.00163205  0.04862701 -0.07644996  0.57584556
   0.13142353 -0.44207135  0.05361343  0.14317368]
 [ 0.18329703  0.14503893 -0.04626461 -0.56237858 -0.20272822  0.46290877
  -0.24886664  0.06300707 -0.05307004 -0.55031958]
 [-0.18574516 -0.13489616  0.83801741 -0.31020987 -0.13508019 -0.0160555
   0.0061422   0.27649933 -0.07029136  0.22105089]
 [ 0.11455545 -0.09630386 -0.20181836 -0.03277705  0.44436294  0.37525216
  -0.26459646  0.58848436  0.18235974  0.38463595]
 [ 0.27156291  0.47102337  0.32503813  0.04492633  0.52547719  0.32156599
   0.32347043 -0.31651017 -0.10151785  0.04553876]
 [-0.67459817  0.008989   -0.01951081 -0.08178474  0.46896722 -0.03821845
  -0.4063748  -0.19922092 -0.29719368 -0.15238981]
 [-0.29601744  0.28101845 -0.30063075 -0.10666798 -0.30446245  0.24694545
   0.35097949  0.17166764 -0.55405727  0.34399144]
 [ 0.23331599  0.18072859  0.17371732  0.44547297 -0.30301521  0.17176835
  -0.64119235 -0.14934723 -0.29046632  0.21248874]
 [ 0.16815978  0.08455644 -0.1644989  -0.59520089 -0.00823426 -0.24620247
  -0.21234634 -0.42476668  0.11616322  0.53249738]]
"""