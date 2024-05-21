# import os
# os.system("pip3 install torch torchvision")
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

class MnistClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 5, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(5, 1, kernel_size=3, stride=2)
        self.fc = nn.Linear(1 * 5 * 5, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class MnistTrainer:
    def __init__(self, model, train_loader, test_loader, criterion, optimizer, num_epochs=10):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.train_losses = []
        self.test_losses = []

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            for inputs, labels in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            self.train_losses.append(running_loss / len(self.train_loader.dataset))
            test_loss = self.evaluate()
            self.test_losses.append(test_loss)
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {self.train_losses[-1]:.4f}, Test Loss: {test_loss:.4f}")

    def evaluate(self):
        # self.model.eval()
        # test_loss = 0.0
        # correct = 0
        # total = 0
        # with torch.no_grad():
        #     for inputs, labels in self.test_loader:
        #         outputs = self.model(inputs)
        #         test_loss += self.criterion(outputs, labels).item() * inputs.size(0)
        #         _, predicted = torch.max(outputs, 1)
        #         total += labels.size(0)
        #         correct += (predicted == labels).sum().item()
        # test_loss /= len(self.test_loader.dataset)
        # print(f"Accuracy on Test Set: {correct / total:.2%}")
        # return test_loss

        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total

    def plot_learning_curves(self):
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Learning Curves')
        plt.legend()
        plt.show()

def task1():
    """
    1. Using automatic differentiation (AD), calculate the derivative of the following function: f(x) = σ2(x), where σ(x) is the logistic sigmoid function.
    """

    def task1a():
        """
        a. Apply AD on the interval [−3,3] and plot your result.
        """
        xx = torch.linspace(-3, 3, 50)

        # Plot the sigmoid function
        yy = torch.sigmoid(xx)**2
        plt.plot(xx, yy, label='sigmoid')
        plt.title("Sigmoid")
        plt.legend()
        plt.grid(True)
        plt.show()
        return xx, yy

    def task1b(xx, yy):
        """
        b. Check the result by comparing against the derivative calculated using the analytically derived formula.
        """
        # Plot the derivative of the sigmoid function calculated analytically
        derivative_analytic = (2*np.exp(-xx))/((np.exp(-xx)+1)**3)
        plt.plot(xx, derivative_analytic, label="derivative of sigmoid^2")
        plt.title("Sigmoid and Its Derivative")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Calculate the derivative using automatic differentiation (AD)
        N = 50
        gradient = torch.zeros(N)
        for i, x_value in enumerate(np.linspace(-3, 3, N)):
            x = torch.tensor(x_value, requires_grad=True)
            y = torch.sigmoid(x)**2
            y.backward()
            gradient[i] = x.grad
        
        # Check if AD gradients equal to analytical gradients
        print('AD and analytic gradients equal:', torch.allclose(derivative_analytic, gradient))

        # Plot the calculated gradient and the analytic derivative
        plt.plot(xx, gradient, label="gradient")
        plt.scatter(xx, derivative_analytic, label="derivative_analytic")
        plt.title("Gradient & derivative analytic")
        plt.legend()
        plt.grid(True)
        plt.show()

    print("Task 1: Calculating derivatives of the sigmoid function...")
    xx, yy = task1a()
    task1b(xx, yy)

def task2():
    """
    2. Find a minimum point of the Himmelblau function: f(x,y) = (x^2+ y -11)^2+ (x + y^2-7)^2
    """

    def f(x):
        x = torch.atleast_2d(x)
        assert x.shape[1] == 2, 'Second dimension should contain the input features'
        # Himmelblau function
        x1 = x[:, 0]
        x2 = x[:, 1]
        return ((x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2)

    def plot_contour():
        N = 500
        xx = torch.linspace(-6, 6, N)
        x1, x2 = torch.meshgrid(xx, xx, indexing='ij')
        x = torch.stack((x1.ravel(), x2.ravel()), axis=1)
        y = f(x).reshape(N, N)
        fig, (ax, ax_log) = plt.subplots(
            1, 2, figsize=(10, 5), layout='constrained')
        ax.contourf(x1, x2, y, levels=20)
        ax.grid()
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_title('Linear scale')
        ax_log.contourf(x1, x2, torch.log(y), levels=20)
        ax_log.grid()
        ax_log.set_xlabel('$x_1$')
        ax_log.set_ylabel('$x_2$')
        ax_log.set_title('Log scale')
        return fig, (ax, ax_log)
    
    def task2a():
        """
        a. Plot the 2D or 3D contour of the function on the [-6,6]^2 interval.
        """
        print("\nTask 2a: Plotting the contour of the function...")
        # Plot the contour
        plot_contour()
        plt.show()

    def task2b(learning_rate):
        """
        b. Using automatic differentiation (but without using an optimiser), write a loop that minimises the function
        """
        print("Task 2b: Minimising the function using manual differentiation...")
        # initial guess
        x = torch.tensor([0., 0.], requires_grad=True)
        # we need to clone and detach so as not to copy the meta information, only the
        # values
        trajectory = [x.clone().detach()]
        for i in range(10):
            # zero out the gradients (because we reuse the tensor)
            x.grad = None
            # forward pass
            y = f(x)
            # calculate the gradient
            y.backward()
            # step in the direction of the negative gradient
            # we need to temporarily disable autograd for this:
            with torch.no_grad():
                x.add_(x.grad * learning_rate)
            # save trajectory
            trajectory.append(x.clone().detach())
        trajectory_manual = torch.stack(trajectory)
        print("Trajectory manual: ", trajectory_manual)
        return trajectory_manual

    def task2c(learning_rate):
        """
        c. Try different built-in optimisers (at least 3) to minimise the same function.
        Tune their hyperparameters (especially the learning rate) by trying out a few sensible values.
        """
        print(f"Task 2c: Minimising the function using optimisers with learning rate={learning_rate}...")
        # initial guesses for each optimiser
        initial_guesses = [
            [0., 0.],  # SGD
            [0., 0.],  # Adam
            [0., 0.],  # RMSprop
        ]

        trajectories = []

        for i, initial_guess in enumerate(initial_guesses):
            x = torch.tensor(initial_guess, requires_grad=True)
            optimiser = [
                optim.SGD(params=[x], lr=learning_rate),
                optim.Adam(params=[x], lr=learning_rate),
                optim.RMSprop(params=[x], lr=learning_rate),
            ][i]
            trajectory = [x.clone().detach()]
            for _ in range(10):
                optimiser.zero_grad()
                y = f(x)
                y.backward()
                optimiser.step()
                trajectory.append(x.clone().detach())
            trajectories.append(torch.stack(trajectory))
        print("Trajectories: ", trajectories)
        return trajectories

    def task2d(learning_rate):
        """
        d. Save the trajectories, and plot each trajectory on top of the contour plot,
        and compare the behaviour of the different optimisers.
        """
        print(f"\nTask 2d: Plotting trajectories for different optimisers with learning rate={learning_rate}...")

        # Plot the contour of the function
        fig, (ax, ax_log) = plot_contour()

        # Set x and y limits to [-6, 6]
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax_log.set_xlim(-6, 6)
        ax_log.set_ylim(-6, 6)

        # Plot the trajectory from task2b
        traj_manual = task2b(learning_rate)
        ax.plot(traj_manual[:, 0], traj_manual[:, 1], '-o', label='Manual', color='black')
        ax_log.plot(traj_manual[:, 0], traj_manual[:, 1], '-o', label='Manual', color='black')

        # Run optimisers and save trajectories
        trajectories = task2c(learning_rate)

        # Plot each trajectory on top of the contour plot
        labels = ['SGD', 'Adam', 'RMSprop']  # Add more labels if needed
        for traj, label in zip(trajectories, labels):
            ax.plot(traj[:, 0], traj[:, 1], '-o', label=label)
            ax_log.plot(traj[:, 0], traj[:, 1], '-o', label=label)

        # Add legend and display the plot
        ax.legend()
        ax_log.legend()
        plt.show()

    task2d(learning_rate=0.1)
    task2d(learning_rate=0.01)
    task2d(learning_rate=0.001)

def task3():
    """
    Classify the MNIST dataset using a deep learning model. MNIST is a classification dataset
    consisting of 28x28 grayscale images of handwritten digits. There are 10 classes in the dataset
    (corresponding to each digit).
    a. Download and load into memory the training and test datasets.
    b. Create a suitable convolutional neural network model, and specify an appropriate loss
        function (categorical cross entropy) and an optimiser (it may turn out that for this task,
        the optimisers perform differently than in Task 2)
    c. Train the model, using minibatches of suitable size (e.g., depending on your hardware),
        evaluate on the test set. Plot some of the misclassified samples from the test set.
    d. Create a learning curve that plots the loss measured on the training and the test dataset
        for each epoch (one pass through the whole training dataset).
    """
    batch_size = 32
    transform = transforms.ToTensor()

    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = MnistClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    trainer = MnistTrainer(model, train_loader, test_loader, criterion, optimizer)
    trainer.train()
    trainer.plot_learning_curves()

def task4():
    """
    4. The task is to reproduce one of the original double descent experiments from [1], Appendix C3.
    This is also an MNIST classification task, so modify and use the code from Task 3.
    a. Create a new model type that is a fully connected neural network with one hidden layer.
    This layer should have a configurable width.
    b. From layer sizes 3 to 800, train the model on 4000 randomly selected training samples,
    then evaluate on the test dataset.
    c. Plot the training and test accuracies (or zero-one errors) versus the layer sizes. Observe
    any similarities, differences with those of the original paper.
    """
    import torch.nn as nn
    
    class FullyConnectedNN(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            x = x.view(x.size(0), -1)  # Flatten the input images
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    input_size = 28 * 28  # Size of each MNIST image
    num_classes = 10
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor()),
        batch_size=1000, shuffle=False)

    hidden_sizes = list(range(3, 10))
    train_accuracies = []
    test_accuracies = []

    for hidden_size in hidden_sizes:
        model = FullyConnectedNN(input_size, hidden_size, num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                                transform=transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=4000, shuffle=True)

        trainer = MnistTrainer(model, train_loader, test_loader, criterion, optimizer, num_epochs=5)
        trainer.train()
        train_accuracies.append(trainer.evaluate())
        test_accuracies.append(trainer.evaluate())


    plt.plot(hidden_sizes, train_accuracies, label='Train Accuracy')
    plt.plot(hidden_sizes, test_accuracies, label='Test Accuracy')
    plt.xlabel('Hidden Layer Size')
    plt.ylabel('Accuracy')
    plt.title('Double Descent Phenomenon')
    plt.legend()
    plt.show()

task1()
task2()
task3()
task4()
