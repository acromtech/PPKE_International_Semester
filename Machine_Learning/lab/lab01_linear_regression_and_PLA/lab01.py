import numpy as np
import matplotlib.pyplot as plt

from LinearRegression import LinearRegression as lr

def task1(title, file, biais=False, plot=False, addPerturbation=False):
    print("\n"+title)
    X_train, Y_train, X_test, Y_test = lr.load_dataset('./Machine_Learning/lab/lab01_linear_regression_and_PLA/data/'+file)
    if biais == False:
        w_train = lr.get_weight(X_train,Y_train)
        w_test = lr.get_weight(X_test,Y_test)
        Y_train_find = lr.find_output(X_train, w_train)
        Y_test_find = lr.find_output(X_test, w_test)
        err_train = lr.calculate_mean_square_error(Y_train, Y_train_find)
        err_test = lr.calculate_mean_square_error(Y_test, Y_test_find)
        lr.display_data(w_train, w_test, err_train, err_test)
        if plot==True : lr.plot_data(X_train, Y_train_find, X_test, Y_test_find)
    else:
        X_train_with_biais = lr.get_biais(X_train)
        X_test_with_biais = lr.get_biais(X_test)
        w_train = lr.get_weight(X_train_with_biais,Y_train)
        w_test = lr.get_weight(X_test_with_biais,Y_test)

        if addPerturbation==False :
            Y_train_find = lr.find_output(X_train_with_biais, w_train)
            Y_test_find = lr.find_output(X_test_with_biais, w_test)
            err_train = lr.calculate_mean_square_error(Y_train, Y_train_find)
            err_test = lr.calculate_mean_square_error(Y_test, Y_test_find)
            lr.display_data(w_train, w_test, err_train, err_test)
        else :
            # Generate perturbations
            epsilon = 0.01  # Small perturbation
            perturbations = np.random.normal(0, epsilon, size=w_train.shape)
            bias_perturbation = np.random.normal(0, epsilon)

            # Add perturbations
            w_train_perturbed = w_train + perturbations
            w_train_perturbed[-1] += bias_perturbation

            Y_train_find = lr.find_output(X_train_with_biais, w_train_perturbed)
            Y_test_find = lr.find_output(X_test_with_biais, w_test)
            err_train = lr.calculate_mean_square_error(Y_train, Y_train_find)
            err_test = lr.calculate_mean_square_error(Y_test, Y_test_find)
            lr.display_data(w_train_perturbed, w_test, err_train, err_test)

        if plot==True : lr.plot_data(X_train_with_biais, Y_train_find, X_test_with_biais, Y_test_find)

#--------------------------------------------------------
task1("a. linear.npz: no bias term", "linear.npz", 
      plot=True)
task1("b. affine1.npz: bias term is nonzero", "affine1.npz", 
      plot=True, biais=True)
task1("c. affine2.npz: there is noise in the dataset.", "affine2.npz", 
      plot=True, biais=True, addPerturbation=True) # Add some small (one-dimensional) perturbations to the weight vector (and bias) you found, and compare the new MSE.
task1("d. rank.npz: X may be a little bit weird", "rank.npz", 
      plot=True, biais=True)
task1("e. zero.npz", "zero.npz", 
      plot=True, biais=True)



    # Which coeﬀicients are neglibible? Why? 
    # Why is the error so large? What do you think, is there output noise in the dataset?

    # Regarding the significance of coefficients and the high error:

    # 1. Some coefficients might be negligible if their corresponding features have little influence on the output. This could happen due to either multicollinearity among features or the features being irrelevant to predicting the output.
    # 2. The error might be large due to several reasons:
    #    - The features might not have a strong linear relationship with the output, leading to poor performance of linear regression.
    #    - There could be outliers or noise in the data, making it challenging for the linear regression model to accurately capture the underlying patterns.
    #    - Overfitting or underfitting of the model may occur if the model complexity does not match the complexity of the data.

    # To determine whether there is output noise in the dataset, we need to examine the residuals (difference between actual and predicted values) and look for patterns or randomness. Additionally, analyzing the significance of coefficients and conducting hypothesis testing could provide insights into the relevance of features.

    # Lastly, the provided code also plots the outputs using a 3D scatter plot to visualize the data and predictions for both training and test datasets. This visualization helps in understanding the relationships between features and the output.

#--------------------------------------------------------

### Task 2

# **Hoeffding equation. You are testing a hypothesis on 43 samples, sampled from the data distribution. After measurement, you find your error to be 0.1. Your boss told you that she will tolerate at most an error of 0.2 in production (averaged for the long run). Can you bound the probability that the error will be larger (and she will be unhappy with you)**

# The Hoeffding Inequality is used to bound the probability that the deviation between the sample mean and the true mean of a random variable is larger than a certain value. Given a set of independent and identically distributed (i.i.d) random variables $ X_1, X_2, ..., X_N $, the Hoeffding Inequality states that:

# $$ P(|\nu - \mu| \geq \epsilon) \leq 2e^{-2N\epsilon^2} $$

# Where:
# - $ \nu $ is the sample mean,
# - $ \mu $ is the true mean,
# - $ N $ is the number of samples,
# - $ \epsilon $ is the deviation we're interested in,

# In this case, $ N = 43 $ and $ \epsilon = 0.2 $. We want to find the probability that $ |\nu - \mu| \geq 0.2 $.

# We can calculate this probability using the Hoeffding Inequality.

epsilon = 0.2
N = 43
probabilite = 2 * np.exp(-2 * epsilon**2 * N)
print("La probabilité que l'erreur dépasse 0.2 est:", probabilite)

# The output -> 0.06412937065572148 and $0.06412937065572148 < 0.2$ so the boss will be happy !

#--------------------------------------------------------

# Task 3 : PLA Algorithm
# Load the data
data = np.load('./Machine_Learning/lab/lab01_linear_regression_and_PLA/data/pla.npz')

# Run PLA
w, (ani, fig) = lr.pla(data)

# Display the animation
from IPython.display import HTML
HTML(ani.to_jshtml())