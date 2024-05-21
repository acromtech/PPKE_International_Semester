import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

ospath = "./Machine_Learning/lab/lab08_expectation_maximisation"

def print_task_results(final_means, final_covariances, final_component_probs, responsibilities):
    print("Final Component Means:")
    print(final_means)
    print("\nFinal Component Covariances:")
    print(final_covariances)
    print("\nFinal Component Probabilities:")
    print(final_component_probs)
    print("\nResponsibilities Matrix:")
    print(responsibilities)


# Task 1 - EM Algorithm for Gaussian Mixture Model with Soft Assignments

data_task1 = np.load(ospath + "/data/EM_var.npz")['X']
num_components_task1 = 6
num_samples_task1 = data_task1.shape[0]
data_dim_task1 = data_task1.shape[1]

# Initialization for Task 1
means_task1 = np.random.rand(num_components_task1, data_dim_task1) * np.max(data_task1, axis=0)
covariances_task1 = [np.eye(data_dim_task1) * np.var(data_task1, axis=0)] * num_components_task1
component_probs_task1 = np.ones(num_components_task1) / num_components_task1
max_iter_task1 = 1000
tolerance_task1 = 1e-6

# E-step for Task 1
def compute_responsibilities(data, means, covariances, component_probs):
    responsibilities = np.zeros((data.shape[0], num_components_task1))
    for i in range(num_components_task1):
        cov = covariances[i] + 1e-6 * np.eye(data_dim_task1)  # Add a small constant to ensure numerical stability
        rv = multivariate_normal(mean=means[i], cov=cov)
        responsibilities[:, i] = component_probs[i] * rv.pdf(data)
    responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)
    return responsibilities

# M-step for Task 1
def update_parameters(data, responsibilities):
    means = np.dot(responsibilities.T, data) / np.sum(responsibilities, axis=0, keepdims=True).T
    covariances = []
    for i in range(num_components_task1):
        weighted_data = (data - means[i]) * np.sqrt(responsibilities[:, i][:, np.newaxis])
        cov = np.dot(weighted_data.T, weighted_data) / np.sum(responsibilities[:, i])
        covariances.append(cov)
    component_probs = np.mean(responsibilities, axis=0)
    return means, covariances, component_probs

# EM Algorithm for Task 1
def expectation_maximization(data, means, covariances, component_probs, max_iter, tolerance):
    for i in range(max_iter):
        responsibilities = compute_responsibilities(data, means, covariances, component_probs)
        new_means, new_covariances, new_component_probs = update_parameters(data, responsibilities)
        
        # Difference between old and new parameters
        mean_diff = np.linalg.norm(new_means - means)
        cov_diff = sum([np.linalg.norm(new_covariances[k] - covariances[k]) for k in range(num_components_task1)])
        prob_diff = np.linalg.norm(new_component_probs - component_probs)
        
        # Termination conditions
        if mean_diff < tolerance and cov_diff < tolerance and prob_diff < tolerance:
            print(f"Convergence reached after {i+1} iterations.")
            break
        
        # Update parameters for the next iteration
        means, covariances, component_probs = new_means, new_covariances, new_component_probs
    
    return means, covariances, component_probs, responsibilities

def task1():
    """
    CONCLUSION TASK 1

    The Expectation-Maximization (EM) algorithm was used to fit a Gaussian Mixture Model (GMM) to data generated by a Gaussian mixture model with 6 components. 

    Here's what we observed:

    - The EM algorithm converges after a certain number of iterations. 
    In our case, convergence was achieved after a reasonable number of iterations.

    - The final component means (represented by red crosses) are computed by the EM algorithm. 
    These means serve to represent the centers of each component of the fitted Gaussian mixture.

    - The data is displayed using a scatter plot, where each point is colored according to its predicted cluster. 
    Colors represent different components of the Gaussian mixture model. 
    This allows us to visualize how the data is distributed across the different components.

    - By looking at the visualization, we can see how the clusters are separated in feature space. 
    The centers of the final components represent the centers of the clusters estimated by the EM algorithm.
    """
    print("-----------------------------------------------------------------")
    print("\nTask 1 - EM Algorithm for Gaussian Mixture Model with Soft Assignments\n")
    final_means, final_covariances, final_component_probs, responsibilities = expectation_maximization(data_task1, means_task1, covariances_task1, component_probs_task1, max_iter_task1, tolerance_task1)
    print_task_results(final_means, final_covariances, final_component_probs, responsibilities)
    plt.figure(figsize=(10, 8))
    plt.scatter(data_task1[:, 0], data_task1[:, 1], c=np.argmax(responsibilities, axis=1), cmap='viridis', alpha=0.6)
    plt.scatter(final_means[:, 0], final_means[:, 1], c='red', marker='x', label='Mean of Components')
    plt.title('Gaussian Mixture Model - EM Algorithm')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.colorbar(label='Component')
    plt.show()

task1()


# Task 2 - EM Algorithm for Gaussian Mixture Model with Hard Assignments

data_task2 = np.load(ospath + "/data/EM_equal.npz")['X']
num_components_task2 = 4
num_samples_task2 = data_task2.shape[0]
data_dim_task2 = data_task2.shape[1]

# Initialization for Task 2
means_task2 = np.random.rand(num_components_task2, data_dim_task2) * np.max(data_task2, axis=0)
covariances_task2 = [np.eye(data_dim_task2) * np.var(data_task2, axis=0)] * num_components_task2
component_probs_task2 = np.ones(num_components_task2) / num_components_task2
max_iter_task2 = 1000
tolerance_task2 = 1e-6

# E-step for Task 2 (with hard assignments)
def compute_responsibilities_hard(data, means, covariances, component_probs):
    responsibilities = np.zeros((data.shape[0], num_components_task2))
    for i in range(num_components_task2):
        rv = multivariate_normal(mean=means[i], cov=covariances[i] + 1e-6 * np.eye(data_dim_task2), allow_singular=True)  
        responsibilities[:, i] = component_probs[i] * rv.pdf(data)
    
    # Hard assignments
    max_responsibility_idx = np.argmax(responsibilities, axis=1)
    hard_responsibilities = np.zeros_like(responsibilities)
    for i, idx in enumerate(max_responsibility_idx):
        hard_responsibilities[i, idx] = 1
    
    return hard_responsibilities

# M-step for Task 2
def update_parameters_hard(data, responsibilities):
    denominator = np.sum(responsibilities, axis=0, keepdims=True) + 1e-6  
    means = np.dot(responsibilities.T, data) / denominator.T
    covariances = []
    for i in range(num_components_task2):
        weighted_data = (data - means[i]) * np.sqrt(responsibilities[:, i][:, np.newaxis])
        cov = np.dot(weighted_data.T, weighted_data) / denominator[0, i]
        covariances.append(cov)
    component_probs = np.mean(responsibilities, axis=0)
    return means, covariances, component_probs

# EM Algorithm for Task 2 (with hard assignments)
def expectation_maximization_hard(data, means, covariances, component_probs, max_iter, tolerance):
    for i in range(max_iter):
        responsibilities = compute_responsibilities_hard(data, means, covariances, component_probs)
        new_means, new_covariances, new_component_probs = update_parameters_hard(data, responsibilities)
        
        # Difference between old and new parameters
        mean_diff = np.linalg.norm(new_means - means)
        cov_diff = sum([np.linalg.norm(new_covariances[k] - covariances[k]) for k in range(num_components_task2)])
        prob_diff = np.linalg.norm(new_component_probs - component_probs)
        
        # Termination conditions
        if mean_diff < tolerance and cov_diff < tolerance and prob_diff < tolerance:
            print(f"Convergence reached after {i+1} iterations.")
            break
        
        # Update parameters for the next iteration
        means, covariances, component_probs = new_means, new_covariances, new_component_probs
    
    return means, covariances, component_probs, responsibilities

def task2():
    """
    CONCLUSION TASK 2

    In Task 2, the EM algorithm with hard assignments was applied to fit a Gaussian Mixture Model (GMM) to data generated by a Gaussian mixture model with 4 components. 

    Here's a summary of the findings:

        - The EM algorithm converges after a certain number of iterations. 
        In our case, convergence was achieved after a reasonable number of iterations.

        - The final component means (represented by red crosses) are computed by the EM algorithm. 
        These means represent the centers of each component of the fitted Gaussian mixture.

        - The data is displayed using a scatter plot, where each point is colored according to its predicted cluster. 
        Colors represent different components of the Gaussian mixture model. 
        This visualization helps understand how the data is distributed across the different components.

        - By examining the visualization, we can observe how the clusters are separated in feature space. 
        The centers of the final components represent the centers of the clusters estimated by the EM algorithm.

    Question c. : 
        The termination conditions for the EM algorithm involve reaching parameter convergence, maximum iterations, or log-likelihood convergence. 
        These are similar to those in K-means, which focuses on centroid stability or cluster assignment convergence. 
        Both algorithms aim to optimize their respective objective functions, but EM uses soft assignments while K-means uses hard assignments.
    """
    print("-----------------------------------------------------------------")
    print("\nTask 2 - EM Algorithm for Gaussian Mixture Model with Hard Assignments\n")
    final_means, final_covariances, final_component_probs, responsibilities = expectation_maximization_hard(data_task2, means_task2, covariances_task2, component_probs_task2, max_iter_task2, tolerance_task2)
    print_task_results(final_means, final_covariances, final_component_probs, responsibilities)
    plt.figure(figsize=(10, 8))
    plt.scatter(data_task2[:, 0], data_task2[:, 1], c=np.argmax(responsibilities, axis=1), cmap='viridis', alpha=0.6)
    plt.scatter(final_means[:, 0], final_means[:, 1], c='red', marker='x', label='Mean of Components')
    plt.title('Gaussian Mixture Model - EM Algorithm with Hard Assignments')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.colorbar(label='Component')
    plt.show()

task2()


# Task 3 - General EM Algorithm with Soft Assignments and Comparison

def compute_responsibilities_soft(data, means, covariances, component_probs):
    responsibilities = np.zeros((data.shape[0], num_components_task2))
    for i in range(num_components_task2):
        rv = multivariate_normal(mean=means[i], cov=covariances[i])
        responsibilities[:, i] = component_probs[i] * rv.pdf(data)
    responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)
    return responsibilities

def update_parameters_soft(data, responsibilities):
    means = np.dot(responsibilities.T, data) / np.sum(responsibilities, axis=0, keepdims=True).T
    covariances = []
    for i in range(num_components_task2):
        weighted_data = (data - means[i]) * np.sqrt(responsibilities[:, i][:, np.newaxis])
        cov = np.dot(weighted_data.T, weighted_data) / np.sum(responsibilities[:, i])
        covariances.append(cov)
    component_probs = np.mean(responsibilities, axis=0)
    return means, covariances, component_probs

def expectation_maximization_soft(data, means, covariances, component_probs, max_iter, tolerance):
    for i in range(max_iter):
        responsibilities = compute_responsibilities_soft(data, means, covariances, component_probs)
        new_means, new_covariances, new_component_probs = update_parameters_soft(data, responsibilities)
        mean_diff = np.linalg.norm(new_means - means)
        cov_diff = sum([np.linalg.norm(new_covariances[k] - covariances[k]) for k in range(num_components_task2)])
        prob_diff = np.linalg.norm(new_component_probs - component_probs)
        if mean_diff < tolerance and cov_diff < tolerance and prob_diff < tolerance:
            print(f"Convergence reached after {i+1} iterations.")
            break
        means, covariances, component_probs = new_means, new_covariances, new_component_probs
    return means, covariances, component_probs, responsibilities

def compare_EM_algorithms(data, num_components, max_iter, tolerance):
    # Hard assignment
    means_hard, covariances_hard, component_probs_hard, responsibilities_hard = expectation_maximization_hard(data, means_task2, covariances_task2, component_probs_task2, max_iter_task2, tolerance_task2)
    # Soft assignment
    means_soft, covariances_soft, component_probs_soft, responsibilities_soft = expectation_maximization_soft(data, means_task2, covariances_task2, component_probs_task2, max_iter_task2, tolerance_task2)
    
    return (means_hard, covariances_hard, component_probs_hard, responsibilities_hard), (means_soft, covariances_soft, component_probs_soft, responsibilities_soft)

def task3():
    """
    CONCLUSION TASK 3

    In Task 3, we implemented a general EM algorithm with soft assignments and compared it with the EM algorithm with hard assignments.

    Here's what we observed:

        - Both the soft assignment and hard assignment EM algorithms converge after a certain number of iterations.
        - We compared the final component means, covariances, and probabilities obtained from both algorithms.
        - Additionally, we observed how the data points were assigned to clusters in both cases.

    """
    print("-----------------------------------------------------------------")
    print("\nTask 3 - General EM Algorithm with Soft Assignments and Comparison\n")
    result_hard, result_soft = compare_EM_algorithms(data_task2, num_components_task2, max_iter_task2, tolerance_task2)
    print("Results for EM Algorithm with Hard Assignments:")
    print_task_results(*result_hard)
    print("\nResults for EM Algorithm with Soft Assignments:")
    print_task_results(*result_soft)

task3()