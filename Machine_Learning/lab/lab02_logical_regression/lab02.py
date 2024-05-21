from LogisticRegression import LogisticRegression
import numpy as np

def task1():
    """
    Questions :
        a. For the following values of x, please calculate the values of the logistic sigmoid: 0, -1, 1, -10, 10, -1000, 1000
        b. Create a sigmoid function implementation that works for all the values above.
        c. (Optional) Your implementation should avoid any overflows during calculation.

    def sigmoid_test1(h):
        sigma1, sigma2 = [], []
        for i in range (len(h)):
            sigma1.append(1/(1+np.exp(-h[i])))
            sigma2.append(np.exp(h[i])/(1+np.exp(h[i])))
        return sigma1, sigma2

    Output :
        RuntimeWarning: overflow encountered in exp                     sigma1.append(1/(1+np.exp(-h[i])))
        RuntimeWarning: overflow encountered in exp                     sigma2.append(np.exp(h[i])/(1+np.exp(h[i])))
        RuntimeWarning: invalid value encountered in scalar divide      sigma2.append(np.exp(h[i])/(1+np.exp(h[i])))

    The error is due to a numerical overflow when calculating the exponential of very large or very negative values of `h`. 
    Specifically, for eX_train_traintremely large values of `h`, the exponential of these values exceeds the numerical range supported by the system, leading to an overflow error.
    To avoid these errors, we can utilize a numerically stable version of the sigmoid function by exploiting the mathematical identity:

    $$ \sigma(x) = \frac{1}{1 + e^{-x}} = \frac{e^x}{e^x + 1} $$

    By using this identity, we can prevent overflow errors by computing the exponential of smaller values of `h` rather than calculating the exponential of very large values. Here's what I corrected:

    def sigmoid_test2(h):
        if h >= 0.0: return 1 / (1 + np.exp(-h))
        else: return np.exp(h) / (1 + np.exp(h))

    This implementation avoids overflow errors by computing the exponential of `h` or `-h` depending on the value of `h`. This ensures that the exponential is computed for numerically smaller values, thus preventing overflow errors.
    """
    print("\nTASK 1 ----------------------------------")
    model = LogisticRegression()
    sigmoid_test_values = np.array([0., -1, 1, -10, 10, -1000, 1000])
    sigmoid_values = model.sigmoid(sigmoid_test_values)
    for i in range(len(sigmoid_test_values)):
        print(f"Result for {sigmoid_test_values[i]:0.3}: \t {sigmoid_values[i]:0.3}")

def task2():  
    """
    Description:
        Using the pla.npz and pocket.npz datasets from the previous lab, classify the samples using logistic regression.
        Note that the class labels are -1 and 1, but the range of sigmoid is [0, 1], take care of transformations where needed
    """
    print("\nTASK 2 ----------------------------------")
    
    ## a. Implement:
    #       - a prediction function for the logistic regression model, which calculates the probability that the given input belongs to the positive class,
    #       - an error function, which calculates the cross-entropy error for the given input and output
    #       - an error gradient function, which calculates the gradient with respect to the model parameters at the given input and output
    
    data = np.load("./Machine_Learning/lab/lab02_logical_regression/data/pla.npz")
    x, y, x_test, y_test =  data["X"], data["Y"], data["X_test"],data["Y_test"]
    N, d = x.shape
    w = np.zeros(d)

    model = LogisticRegression()
    train_probabilities = model.predict_proba(x, w)
    print("\nTrain probability:", train_probabilities)

    train_error = model.crossentropy_error(x, y, w)
    print("\nCross-entropy error for training data:", train_error)

    grad = model._error_gradient(x, y, w)
    print("\nGradient:", grad)

    ## b. Using the functions implemented above, write a full batch gradient descent training algorithm. 
    #     In each iteration, calculate the mean error on the training dataset, calculate its gradient and update your parameter vector using a properly chosen step size.
    #           ->  Thanks to a learning_rate_pla = 0.0002 and a learning_rate_pocket = 0.00016, the cross-entropy error decreases over iterations for both the PLA and Pocket datasets. 
    #               This indicates that the logistic regression model is learning from the training data and improving its predictions over time.
    ## c. Visualise the linear separator in each iteration

    data_pla = np.load('./Machine_Learning/lab/lab02_logical_regression/data/pla.npz')
    data_pocket = np.load('./Machine_Learning/lab/lab02_logical_regression/data/pocket.npz')

    x_pla, y_pla, x_pla_test, y_pla_test = data_pla['X'], data_pla['Y'], data_pla['X_test'], data_pla['Y_test']
    x_pocket, y_pocket, x_pocket_test, y_pocket_test = data_pocket['X'], data_pocket['Y'], data_pocket['X_test'], data_pocket['Y_test']

    learning_rate_pla = 0.002
    learning_rate_pocket = 0.002
    num_iterations = 3000

    print("\nbatch gradient descent pla")
    weights_pla = model.batch_gradient_descent(x_pla, y_pla, learning_rate_pla, num_iterations)
    print("\nbatch gradient descent pocket")
    weights_pocket = model.batch_gradient_descent(x_pocket, y_pocket, learning_rate_pocket, num_iterations)

    # d. Add stochastic gradient descent functionality to your code with a batch size parameter, where batch_size = None means full batch descent. After each couple of iterations, visualise the linear separator (as before).
    #   - Batch gradient descent utilizes the entire dataset at each iteration to update the model weights. It shows a decrease in the error (loss) over the iterations, but the change in the separating line is not clearly visible at each iteration in the visualization.
    #   - Stochastic gradient descent uses a random sample at each iteration to update the model weights. It also shows a decrease in the error, but it is more likely to fluctuate due to the use of smaller data batches. The visualization of the separating line shows more frequent changes and may appear more unstable compared to batch gradient descent.
    #   Batch gradient descent is more stable as it utilizes the entire dataset at each iteration, while stochastic gradient descent is faster to converge to a solution but may be more unstable due to the variability introduced by using small data batches. The choice between the two methods depends on computational constraints, dataset size, and the need for fast convergence.
    # e. In addition, in a separate subplot plot each of the training samples transformed into the score space (ie. just before the sigmoid) and a logistic sigmoid function as reference

    batch_size = None  # Specify the batch size
    print("\nstochastic gradient descent pla")
    weights_pla_sgd = model.stochastic_gradient_descent(x_pla, y_pla, learning_rate_pla, num_iterations, batch_size)
    weights_pla_sgd = model.stochastic_gradient_descent2(x_pla, y_pla, learning_rate_pla, num_iterations)

task1()
task2()
