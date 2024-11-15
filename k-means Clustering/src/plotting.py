import matplotlib.pyplot as plt
import numpy as np

def plot_kmeans_iterations(data, centroids_history, assignments_history):
    """
    Plots K-Means iterations showing centroids and data point assignments.

    Parameters:
    data (np.array of shape (n_samples,2)): Data points.
    centroids_history (list of (k,2) shape np.arrays): List of the centroids at each iteration.
    assignments_history (list of (k) shape np.arrays): List of data point assignments at each iteration.
    """
    n_iters = len(centroids_history)
    plt.figure(figsize=(15, 5 * n_iters))

    for i in range(n_iters):
        plt.subplot(n_iters, 1, i + 1)
        plt.scatter(data[:, 0], data[:, 1], c=assignments_history[i], alpha=0.5, marker='.')
        plt.scatter(centroids_history[i][:, 0], centroids_history[i][:, 1], c='red', marker='X')
        plt.title(f'Iteration {i+1}')
        plt.xlabel('X')
        plt.ylabel('Y')

    plt.tight_layout()
    plt.show()

def plot_pulsar_clusters_vs_labels(X, y, y_pred):
    """
    Plots the results of k-means clustering against actual labels.

    This function visualizes the comparison between k-means clustering predictions 
    (y_pred) and actual labels (y) for each pair of features in the dataset. 
    It uses different colors to indicate true positives, true negatives, false negatives,
    and false positives.

    Parameters:
    X (numpy.ndarray): The feature data set. Each row represents a sample, 
                       and each column represents a feature.
    y (numpy.ndarray): The actual labels for each sample. This array is 1-dimensional.
    y_pred (numpy.ndarray): The predicted labels from k-means clustering. 
                            This array is 1-dimensional.

    The function creates scatter plots for each pair of features in the data set.
    """
    
    feature_names = ['Mean of the integrated profile',
                     'Standard deviation of the integrated profile',
                     'Excess kurtosis of the integrated profile',
                     'Skewness of the integrated profile',
                     'Mean of the DM-SNR curve',
                     'Standard deviation of the DM-SNR curve',
                     'Excess kurtosis of the DM-SNR curve',
                     'Skewness of the DM-SNR curve']

    # making a coloring array for scatterplots
    lightred = np.array([0.9, 0, 0])
    red = np.array([0.4, 0, 0])
    yellow = np.array([0.9, 0.9, 0])
    blue = np.array([0, 0, 0.9])

    data_coloring = np.zeros((y.size, 3))
    data_coloring[(y_pred == 1) & (y == 1)] = yellow  # True positive
    data_coloring[(y_pred == 0) & (y == 0)] = blue    # True negative
    data_coloring[(y_pred == 0) & (y == 1)] = red     # False negative
    data_coloring[(y_pred == 1) & (y == 0)] = lightred # False positive

    for i in range(4):
        dims = [2 * i, 2 * i + 1]
        plt.figure(figsize=(8, 6))
        plt.title('k-means prediction vs. label')
        plt.scatter(X[:, dims[0]], X[:, dims[1]], c=data_coloring, marker='.', alpha=0.6)
        plt.xlabel(feature_names[dims[0]])
        plt.ylabel(feature_names[dims[1]])

        # Adding legend
        plt.scatter([], [], color=yellow, label='True Positive')
        plt.scatter([], [], color=blue, label='True Negative')
        plt.scatter([], [], color=red, label='False Negative')
        plt.scatter([], [], color=lightred, label='False Positive')
        plt.legend()

        plt.show()
