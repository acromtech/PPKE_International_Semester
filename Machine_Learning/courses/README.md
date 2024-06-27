# Concepts

## Introduction
* [Introduction to Machine Learning (ML)](./introduction.md)
* [Learning types](./learning_types.md)

## Perceptron Learning Algorithm (PLA)
PLA is a supervised learning algorithm used for binary classification. It updates the weights of the model iteratively to minimize the classification error on the training set. The algorithm converges if the data is linearly separable.
* [Perceptron](./perceptron.md)
* [PLA algorithm](./pla.md)

## Pocket Algorithm
An extension of the PLA that handles non-linearly separable data. It keeps a "pocket" of the best solution found so far (in terms of classification accuracy) and updates it as the algorithm runs.
* [Pocket algorithm](./pocket.md)

## Hoeffding’s Inequality / Probably Approximately Correct (PAC)
Hoeffding's Inequality provides a bound on the probability that the sum of random variables deviates from its expected value. PAC learning is a framework that defines the conditions under which a learning algorithm can produce a hypothesis that is probably approximately correct.
* [Hoeffding Inequality - Simple hypothesis](./hoeffding_inequality_simple_hypothesis.md)
* [Hoeffding Inequality - Multiple hypothesis](./hoeffding_inequality_multiple_hypothesis.md)
* [Feasability of learning - Brief summary](./feasability_of_learning.md)

## BAD Sample and BAD Data
These concepts refer to the quality of data. BAD samples are those that do not represent the underlying distribution well, leading to poor model performance. BAD data can arise from noise, outliers, or errors in data collection.

## Growth Function / Convex Set
The growth function measures the complexity of a hypothesis class by counting the maximum number of distinct labelings it can produce on any set of points. A convex set is a set where any line segment between two points in the set lies entirely within the set.

## Break Point of H
The break point is the smallest number of points that cannot be shattered by the hypothesis set H. It helps in determining the complexity of H.

## Restriction of Break Point
This concept involves limiting the growth function by using the break point to bound the number of labelings that H can produce.

## Bounding Function: Basic Cases and Inductive Cases
Bounding functions are used to bound the growth function of H. Basic cases refer to initial bounds, while inductive cases involve using induction to generalize the bounds.

## A Pictorial Proof
A visual demonstration or illustration to prove a concept or theorem in machine learning.

## Definition of VC Dimension
The VC (Vapnik-Chervonenkis) dimension of a hypothesis set is the largest number of points that can be shattered by the set. It measures the capacity of the hypothesis space.

## VC Dimension of Perceptrons
The VC dimension of a perceptron (linear classifier) is d+1, where d is the number of input features. This reflects the perceptron's ability to shatter d+1 points in general position.

## Physical Intuition of VC Dimension
VC dimension provides an intuition about the capacity of a model to fit various data sets. A higher VC dimension indicates a more complex model with greater capacity to fit data.

## Interpreting VC Dimension
VC dimension helps in understanding the trade-off between model complexity and the ability to generalize to new data. It is used to prevent overfitting.

## Noise and Error
Noise refers to random variations in data that cannot be captured by the model. Error consists of bias (error due to wrong assumptions) and variance (error due to sensitivity to fluctuations in the training set).

## Probabilistic Data Generation
A process where data points are generated according to some probability distribution. It is used in various probabilistic models and algorithms.

## Probabilistic Target Errors
Errors that occur when the target variable is generated probabilistically, leading to inherent uncertainty in predictions.

## Error Measure with Mini-Targets
Using small, specific targets (mini-targets) for evaluating and minimizing errors in a model.

## Minimizing Models for Error Measures
Optimizing models to minimize specific error measures, such as mean squared error, cross-entropy loss, etc.

## Nonlinear via Nonlinear Feature Transform plus Linear with Price of Model Complexity
Using a nonlinear feature transformation to map input data into a higher-dimensional space where a linear model can be applied. This increases model complexity.

## What is Overfitting?
Overfitting occurs when a model learns the training data too well, capturing noise and outliers, resulting in poor generalization to new data.

## The Role of Noise and Data Size
Noise can lead to overfitting if not handled properly. Larger data sets help mitigate the impact of noise and improve the model's generalization.

## Deterministic Noise
Errors that arise from the limitations of the model's hypothesis space, which cannot capture the underlying data distribution perfectly.

## Dealing with Overfitting
Techniques to prevent overfitting include regularization, cross-validation, pruning, and using simpler models.

## Regularization
A technique to add a penalty term to the loss function to discourage complex models, thus preventing overfitting.

## Regularized Hypothesis Set
A hypothesis set that incorporates regularization to limit model complexity and improve generalization.

## Original H + Constraint
Combining the original hypothesis set H with additional constraints to form a regularized hypothesis set.

## Weight Decay Regularization
A form of regularization where a penalty proportional to the square of the weights is added to the loss function, encouraging smaller weights.

## Add λNwTw in Eaug
Adding a regularization term (λNwTw) to the augmented error function (Eaug) to penalize large weights.

## Regularization and VC Theory
Regularization reduces the effective VC dimension of a model, lowering its capacity and reducing the risk of overfitting.

## General Regularizers
Various regularization techniques that can be target-dependent, plausible, or friendly, depending on the specific problem and model.

## Model Selection Problem
The challenge of selecting the best model from a set of candidates based on performance on a validation set.

## Validation
A technique to evaluate model performance on a separate data set not used for training, to ensure it generalizes well.

## Leave-One-Out Cross Validation
A cross-validation method where one data point is left out as the validation set, and the model is trained on the remaining data. This is repeated for each data point.

## V-Fold Cross Validation
A cross-validation technique where the data is divided into V equal parts, and the model is trained V times, each time using a different part as the validation set and the remaining parts as the training set.

## Motivation
The reason for using various machine learning techniques, often to improve model performance, interpretability, or generalization.

## Neural Network Hypothesis
The hypothesis set defined by a neural network model, characterized by layers of neurons and their connections.

## Neural Network Learning
The process of training a neural network using algorithms like backpropagation to minimize the loss function.

## Optimization and Regularization
Techniques used to improve neural network training, including gradient descent for optimization and regularization methods to prevent overfitting.

## Deep Neural Network
A neural network with many layers (deep architecture), capable of learning complex representations from data.

## Autoencoder
A type of neural network used for unsupervised learning that aims to encode input data into a lower-dimensional representation and then decode it back to the original input.

## Denoising Autoencoder
An extension of autoencoders that aims to remove noise from the input data by learning to reconstruct the original, noise-free data from corrupted inputs.

## Principal Component Analysis (PCA)
A dimensionality reduction technique that transforms data into a set of orthogonal components (principal components) ordered by the amount of variance they capture.

## Reinforcement Learning (RL) / Q-Learning / Policy Gradient
RL is a learning paradigm where an agent learns to make decisions by interacting with an environment. Q-Learning is an RL algorithm that learns the value of actions in states. Policy Gradient methods learn the policy directly, optimizing the expected reward.