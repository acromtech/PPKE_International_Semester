import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Optional
from visualise import visualise

class LinearRegression:
    def load_dataset(file):
        data = np.load(file)
        return data['X'], data['Y'], data['X_test'], data['Y_test']

    def get_weight(X, Y):  
        return np.linalg.inv(X.T @ X) @ X.T @ Y

    def find_output(X, w):
        Y = X @ w
        return Y

    def calculate_mean_square_error(Y, Y_find):
        return np.mean((Y - Y_find) ** 2)

    def display_data(w_train, w_test, err_train, err_test):
        print("Estimated weight vector:", w_train)
        print("Reference weight vector:", w_test)
        print("Mean squared error (train):", err_train)
        print("Mean squared error (test):", err_test)

    def get_biais(X):
        # Avant de calculer les poids pour les données qui nécessitent un terme de biais (biais différent de zéro), 
        # la fonction get_biais ajoute une colonne de 1 à la matrice des caractéristiques. 
        # Cela équivaut à ajouter un terme constant (x_0=1) à chaque exemple, permettant au modèle de capturer une translation dans les données.

        # X :   matrice des caractéristiques d'origine, qui contient les valeurs des caractéristiques pour chaque exemple de données.
        #       généralement de dimension (m,n), où m est le nombre d'exemples de données et n est le nombre de caractéristiques.
        # crée une matrice de dimension (m,1) remplie de 1.
        matrix_all_one = np.ones((X.shape[0], 1))
        # concatène les deux matrices données le long de l'axe spécifié. 
        # L'argument axis=1 signifie que la concaténation doit se faire le long de l'axe des colonnes, 
        # c'est-à-dire que les matrices sont empilées côte à côte.
        X_biais = np.concatenate((X, matrix_all_one), axis=1)
        return X_biais

    def plot_data(X_train, Y_train, X_test, Y_test):
        fig, (ax1) = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
        ax1.scatter(X_train[:, 0], X_train[:, 1], Y_train, c='blue', label='train')
        ax1.scatter(X_test[:, 0], X_test[:, 1], Y_test, c='green', label='test')
        ax1.set_xlabel(f'$X_{0}$')
        ax1.set_ylabel(f'$X_{1}$')
        ax1.set_zlabel('$Y$')
        ax1.legend()
        plt.show()

    def pla(data,
            plot_every: Optional[int] = 1,
            max_iteration: int = 100) -> tuple[np.ndarray, float] | tuple[np.ndarray, float, list]:
        
        x, y = data['X'], data['Y']
        N, d = x.shape
        
        w = np.random.rand(d + 1)  # Initialize weight vector randomly
        if plot_every:
            fig, ax = plt.subplots()
            artists = []
        for i in range(max_iteration):
            acc = np.sum(np.sign(x @ w[1:] + w[0]) == y) / N
            if acc == 1: break  # If all points are classified correctly, stop iterating
            
            # Choose a misclassified sample
            misclassified_idx = np.where(np.sign(x @ w[1:] + w[0]) != y)[0][0]
            x_misclassified = np.concatenate(([1], x[misclassified_idx]))

            # Update the weight vector
            w += y[misclassified_idx] * x_misclassified
            if plot_every and i % plot_every == 0:
                _fig, _ax, artists_ = visualise(w, x, y, ax=ax, animated=True, title=f'Training, accuracy: {acc:0.3}')
                artists.append(artists_)
        if plot_every:
            ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=1000)
            return w, (ani, fig)
        else:
            return w, acc

#-----------------------------------------------------

    def accuracy_score(y_true, y_pred):
        correct = 0
        for true, pred in zip(y_true, y_pred):
            if true == pred:
                correct += 1
        return correct / len(y_true)

    # Exemple d'utilisation avec des prédictions et des étiquettes réelles
    # predictions = [0, 1, 1, 0]
    # labels = [0, 1, 0, 1]
    # accuracy = accuracy_score(labels, predictions)
    # print("Accuracy:", accuracy)

    def mean_squared_error(y_true, y_pred):
        mse = sum((true - pred) ** 2 for true, pred in zip(y_true, y_pred)) / len(y_true)
        return mse

    # Exemple d'utilisation avec des prédictions et des valeurs réelles
    # predictions = [2.5, 3.0, 4.0, 4.5]
    # labels = [3, 2.5, 4.5, 5]
    # mse = mean_squared_error(labels, predictions)
    # print("Mean Squared Error:", mse)

    def mean_absolute_error(y_true, y_pred):
        mae = sum(abs(true - pred) for true, pred in zip(y_true, y_pred)) / len(y_true)
        return mae

    # Exemple d'utilisation avec des prédictions et des valeurs réelles
    # predictions = [2.5, 3.0, 4.0, 4.5]
    # labels = [3, 2.5, 4.5, 5]
    # mae = mean_absolute_error(labels, predictions)
    # print("Mean Absolute Error:", mae)

    def cross_entropy_loss(y_true, y_pred):
        ce = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        return ce

    # Exemple d'utilisation avec des probabilités prédites et des étiquettes réelles (classes codées en one-hot)
    # predictions = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2]])
    # labels = np.array([[1, 0], [0, 1], [1, 0]])
    # cross_entropy = cross_entropy_loss(labels, predictions)
    # print("Cross Entropy Loss:", cross_entropy)
