import numpy as np
import matplotlib.pyplot as plt

def extract_data(path, debug=False):
    data = np.load(path)
    if debug: print(data)
    X_train, X_test, Y_train, Y_test = data['X'], data['X_test'], data['Y'], data['Y_test']
    if debug:
        print("\nX_train:", X_train,
              "\nX_test:", X_test,
              "\nY_train:", Y_train,
              "\nY_test:", Y_test)
    return X_train, X_test, Y_train, Y_test

def polynomial_features(X, degree):
    return np.column_stack([X**i for i in range(degree + 1)])

def linear_regression(X, y):
    theta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
    return theta

def plot_polynomials(X, y, X_test, y_test, degrees, thetas):
    plt.scatter(X, y, label='Training Data')
    plt.scatter(X_test, y_test, color='red', label='Test Data')
    
    x_values = np.linspace(-5, 5, 100)
    for degree, theta in zip(degrees, thetas):
        X_poly = polynomial_features(x_values, degree)
        y_values = X_poly.dot(theta)
        plt.plot(x_values, y_values, label=f'Degree {degree} Polynomial')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Polynomial Regression')
    plt.legend()
    plt.grid(True)
    plt.show()

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def calculate_mse_analytically(poly_coefficients, true_function_coefficients, interval):
    start, end = interval
    x_values = np.linspace(start, end, 1000)                        # Generate x values within the interval
    poly_values = np.polyval(poly_coefficients, x_values)           # Evaluate polynomial function
    true_values = np.polyval(true_function_coefficients, x_values)  # Evaluate true function
    mse = np.mean((poly_values - true_values)**2)
    return mse

def generate_test_data(num_points):
    x_values = np.linspace(-5 * np.pi, 5 * np.pi, num_points)
    y_values = np.sin(x_values)
    return x_values, y_values

def calculate_errors_in_out_sample(x_train, y_train, x_test, y_test, degree):
    # Randomly shuffle
    indices = np.random.choice(len(x_train), len(x_train), replace=False)
    x_train_shuffled = x_train[indices]
    y_train_shuffled = y_train[indices]
    
    in_sample_errors = []
    out_of_sample_errors = []

    for k in range(degree + 1, len(x_train) + 1):
        x_train_subset = x_train_shuffled[:k]
        y_train_subset = y_train_shuffled[:k]
        poly_coefficients = np.polyfit(x_train_subset, y_train_subset, degree)
        y_pred_train = np.polyval(poly_coefficients, x_train_subset)
        y_pred_test = np.polyval(poly_coefficients, x_test)
        in_sample_error = np.mean((y_train_subset - y_pred_train) ** 2)
        out_of_sample_error = np.mean((y_test - y_pred_test) ** 2)
        in_sample_errors.append(in_sample_error)
        out_of_sample_errors.append(out_of_sample_error)
    
    return in_sample_errors, out_of_sample_errors

def calculate_errors_in_out_sample2(x_train, y_train, x_test, y_test, degree):
    # Randomly shuffle the training data
    indices = np.random.choice(len(x_train), len(x_train), replace=False)
    x_train_shuffled = x_train[indices]
    y_train_shuffled = y_train[indices]
    
    in_sample_errors = []
    out_of_sample_errors = []
    min_dataset_size = float('inf')  # Initialize with a very large value
    
    # Calculate errors for different sizes of training data
    for k in range(degree + 1, len(x_train) + 1):
        x_train_subset = x_train_shuffled[:k]
        y_train_subset = y_train_shuffled[:k]
        
        # Fit polynomial regression model
        poly_coefficients = np.polyfit(x_train_subset, y_train_subset, degree)
        y_pred_train = np.polyval(poly_coefficients, x_train_subset)
        y_pred_test = np.polyval(poly_coefficients, x_test)
        
        # Calculate errors
        in_sample_error = np.mean((y_train_subset - y_pred_train) ** 2)
        out_of_sample_error = np.mean((y_test - y_pred_test) ** 2)
        
        in_sample_errors.append(in_sample_error)
        out_of_sample_errors.append(out_of_sample_error)
        
        # Update minimal dataset size if necessary
        if k < min_dataset_size:
            min_dataset_size = k
    
    return in_sample_errors, out_of_sample_errors, min_dataset_size

def plot_learning_curve(in_sample_errors, out_of_sample_errors):
    dataset_sizes = range(len(in_sample_errors))
    
    plt.figure(figsize=(10, 6))
    plt.plot(dataset_sizes, in_sample_errors, label='In-sample Error')
    plt.plot(dataset_sizes, out_of_sample_errors, label='Out-of-sample Error')
    plt.xlabel('Dataset Size')
    plt.ylabel('Mean Squared Error')
    plt.ylim(-0.5,1.5)
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

# --------------------------------------------------------------------

def task1(file, degrees):
    """
    Implement polynomial regression, ie. linear regression with polynomial features. Use polyno-
    mials of degree 0, 1, 2 and 5.
    """
    def task1ab(X, Y, X_test, Y_test, degrees):
        """
        a. Find the parameters that generated the dataset in poly_s.npz. Print training and test
        MSEs for each polynomial. What do you think, what was the degree of the original
        polynomial (the target function), and why (visually and based on the test error)?1
        b. Find the parameters that generated the dataset in poly_d.npz. Print training and test
        MSEs for each polynomial. What do you think, what was the degree of the original
        polynomial, and why (visually and based on the test error)?
        """
        thetas = []
        for degree in degrees:
            X_poly = polynomial_features(X, degree)
            theta = linear_regression(X_poly, Y)
            thetas.append(theta)
            Y_pred_train = X_poly.dot(theta)
            mse_train = mean_squared_error(Y, Y_pred_train)
            X_poly_test = polynomial_features(X_test, degree)
            Y_pred_test = X_poly_test.dot(theta)
            mse_test = mean_squared_error(Y_test, Y_pred_test)
            
            print(f"Degree {degree}:")
            print(f"  Training Mean Squared Error: {mse_train}")
            print(f"  Test Mean Squared Error: {mse_test}")
        return thetas
    
    def task1c(X, Y, X_test, Y_test, degrees, thetas):
        """
        c. Plot the polynomials defined by the parameters found in a) and b) along with the training
        and test samples. To plot a polynomial, simply use np.linspace on the [-5, 5] interval,
        calculate the values for each point, then use plt.plot with a continuous linestyle (which
        is the default, by the way).
        """
        plot_polynomials(X, Y, X_test, Y_test, degrees, thetas)

        """
        POLY_S
        ----------------------
        Based on the provided results:

        - **Polynomial degree: 0**
        - Both the training and test mean squared errors (MSEs) are quite high, indicating poor model performance. This suggests that a constant (zero-degree polynomial) model is too simple to capture the underlying patterns in the data.

        - **Polynomial degree: 1**
        - The training and test MSEs are lower compared to degree 0, indicating better performance. However, the MSEs are still relatively high, suggesting that a linear model (first-degree polynomial) may not be sufficient to capture the complexities of the data.

        - **Polynomial degree: 2**
        - The training and test MSEs are significantly lower compared to lower-degree polynomials. This indicates that a quadratic model (second-degree polynomial) fits the data much better, capturing more of the underlying patterns.

        - **Polynomial degree: 5**
        - The training MSE is extremely low, even close to zero, indicating that the model perfectly fits the training data. However, the test MSE is higher compared to the second-degree polynomial, suggesting that the model might be overfitting the training data. Despite this, the test MSE is still relatively low, indicating good generalization performance.

        Based on the MSE values, the second-degree polynomial (degree 2) seems to be the most appropriate choice as it achieves a low test MSE while avoiding potential overfitting seen with higher-degree polynomials. Therefore, it's likely that the original polynomial function generating the data was of second-degree.

        POLY_D
        ---------------------
        Based on these observations, it's likely that the original polynomial degree generating the data is 2. 

        The problem observed with the degree 5 polynomial regression is again overfitting. Overfitting occurs when a model learns the training data too well, capturing noise and random fluctuations rather than the underlying pattern. 
        As a result, the model performs well on the training data but fails to generalize to unseen data, leading to poor performance on the test data.
        """

    def task1d():
        """
        d. Calculate the Eout analytically, assuming the inputs are uniformly distributed on [-5; 5].
        The generating polynomial
        - in poly_s is p(x) = 3 + 0.5x + 5x^2 (plus some zero-mean noise, but you don't have to deal with that here),
        - in poly_d is p(x) = 3 + 0.5x + 5x^2 - 0.0001x7 + 0.00004x^8 (no noise).
        """
        poly_s_coefficients = [3, 0.5, 5]                                   # Polynomial coefficients for poly_s: p(x) = 3 + 0.5x + 5x^2
        poly_d_coefficients = [3, 0.5, 5, 0, 0, 0, 0, -0.0001, 0.00004]     # Polynomial coefficients for poly_d: p(x) = 3 + 0.5x + 5x^2 - 0.0001x^7 + 0.00004x^8
        interval = (-5, 5)

        mse_poly_s = calculate_mse_analytically(poly_s_coefficients, [0, 0, 0], interval)
        mse_poly_d = calculate_mse_analytically(poly_d_coefficients, [0, 0, 0, 0, 0, 0, 0, 0, 0], interval)
        print("Mean Squared Error (MSE) for poly_s:", mse_poly_s)
        print("Mean Squared Error (MSE) for poly_d:", mse_poly_d)

    X, X_test, Y, Y_test = extract_data(file)
    X = X.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)
    thetas = task1ab(X, Y, X_test, Y_test, degrees)
    task1c(X, Y, X_test, Y_test, degrees, thetas)
    task1d()

def task2():
    """
    Plot the learning curve (here it means Ein and Eout versus the training dataset size) of polynomial regression when approximating a sine function. 
    For this, generate random datasets of sizes ranging to 100 on the input interval [-5𝜋, +5𝜋], train a polynomial model (similarly to Exercise 1) on this dataset, and measure the in sample error. 
    Then estimate the out of sample error by taking 500 evenly spaced points on the interval, and measuring the difference between the output of your model and sin(x) on these data points. 
    Average your measurements across 50 experiments (with new training data points sampled randomly from the interval). Finally, plot Ein and Eout versus the size of the dataset.

    a. So your task is basically:
    * take 500 evenly spaced points in the above interval (np.linspace) and apply the
    sine function; this will be your test dataset (for the whole exercise)
    * pick a polynomial degree d
    * for all training dataset sizes k ranging from d + 1 to 100, do the following:
        * sample k amount of points (uniformly randomly) from the interval (not from the test set), apply the sine function; this will be your training dataset for this iteration
        * fit your polynomial, measure Ein, Eout
        * do this 50 times, averaging the errors
    * plot Ein and Eout versus the dataset size (learning curves)

    b. Try this for polynomials of different degree (eg. 0, 1, 2, 5, 8). Note the minimal dataset
    size for each degree! Based on your experiments, which polynomial (degree) would you
    use if you had 20 points in your training dataset?
    """
    x_test, y_test = generate_test_data(500) # take 500 evenly spaced points in the above interval (np.linspace) and apply the sine function; this will be your test dataset (for the whole exercise)
    degree = 8  # pick a polynomial degree d
    num_experiments = 50
    in_sample_errors_avg = np.zeros(100 - degree)
    out_of_sample_errors_avg = np.zeros(100 - degree)

    for _ in range(num_experiments):  # do this 50 times
        x_train = np.random.uniform(-5 * np.pi, 5 * np.pi, 100) # for all training dataset sizes 𝑘 ranging from d + 1 to 100, do the following
                                                                # sample k amount of points (uniformly randomly) from the interval (not from the test set),
        y_train = np.sin(x_train) # apply the sine function; this will be your training dataset for this iteration

        in_sample_errors, out_of_sample_errors = calculate_errors_in_out_sample(x_train, y_train, x_test, y_test, degree) # it your polynomial, measure 𝐸𝑖𝑛, 𝐸𝑜𝑢t
        in_sample_errors_avg += np.array(in_sample_errors)
        out_of_sample_errors_avg += np.array(out_of_sample_errors)
    in_sample_errors_avg /= num_experiments     #  averaging the error
    out_of_sample_errors_avg /= num_experiments #  averaging the error

    plot_learning_curve(in_sample_errors_avg, out_of_sample_errors_avg) # plot Ein and Eout versus the dataset size (learning curves)

    x_test, y_test = generate_test_data(20)
    degrees = [0, 1, 2, 5, 8]

    for degree in degrees:
        x_train = np.random.uniform(-5 * np.pi, 5 * np.pi, 100)
        y_train = np.sin(x_train)
        in_sample_errors, out_of_sample_errors, min_dataset_size = calculate_errors_in_out_sample2(x_train, y_train, x_test, y_test, degree)
        print(f"Degree {degree}: Minimal dataset size = {min_dataset_size}")
        plot_learning_curve(in_sample_errors, out_of_sample_errors) 
    """
    For a 20 point dataset size, I think the better one is maybe the degree 2 because he have the best fitting in/out sample error...
    """
        
task1("./Machine_Learning/lab/lab04_noise/data/poly_s.npz", degrees=[0, 1, 2, 5])
task1("./Machine_Learning/lab/lab04_noise/data/poly_d.npz", degrees=[0, 1, 2, 5])
task2()