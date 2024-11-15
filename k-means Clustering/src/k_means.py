import numpy as np
import matplotlib.pyplot as plt
import sklearn

def generate_gaussian_mixture_data(k, samples_per_component, d, mean_range, std_dev):
    """
    Generates data from a Gaussian mixture model.

    Parameters:
    k (int): Number of components.
    samples_per_component (int): Number of samples from each of the k clusters to generate.
    mean_range (tuple): The range of means for the Gaussian components.
    std_dev (float): (Uniform) standard deviation for each component.

    Returns:
    numpy.ndarray: Generated dataset with shape (k*samples_per_component, d).
    """
    data = []
    for _ in range(k):
        mean = np.random.uniform(mean_range[0], mean_range[1], d)
        cov = np.eye(d) * std_dev
        samples = np.random.multivariate_normal(mean, cov, samples_per_component)
        data.append(samples)
        
    data = np.vstack(data) # here we go from a list to np.array
    np.random.shuffle(data) # in-place.
    return data

def kmeans(data, k, max_iters=100, tol=1e-3):
    """
    Implements the K-Means clustering algorithm and tracks the history of centroids and assignments.

    Parameters:
    data (numpy.ndarray): Data points for clustering.
    k (int): Number of clusters.
    max_iters (int): Maximum number of iterations.
    tol (float): Iteration stops if all centroids have moved less than tol

    Returns:
    tuple: (final_centroids, final_cluster_assignments, centroids_history, assignments_history)
    """
    def assignments(data, centroids):
        """ Returns np.array of shape (n_samples) with the centroid index for each data point. """
        distances = np.linalg.norm(data - centroids[:, np.newaxis,:], axis=2)  # data.shape is (n,2) and centroids[:, np.newaxis,:] has shape (k,1,2) -> broadcast to (k,n,2)
        cluster_assignments = np.argmin(distances, axis=0)
        return cluster_assignments
    

    # Randomly initialize centroids
    centroids = data[np.random.choice(data.shape[0], k, replace=False), :]  # choose k random data points as centroid inits
    centroids_history = [centroids]
    assignments_history = []

    for _ in range(max_iters):
        # Assign data points to the nearest centroid
        cluster_assignments = assignments(data, centroids)
        assignments_history.append(cluster_assignments)

        # Update centroids
        centroids = np.array([data[cluster_assignments == j].mean(axis=0) for j in range(k)])
        centroids_history.append(centroids)
        
        # Check for convergence (if all centroids do not change more than tol)
        if np.all( np.linalg.norm(centroids_history[-1] - centroids_history[-2], axis=1) < tol):
            print(f'K-means converged to tol {tol} after {len(centroids_history) - 1} iterations.')
            break
    
    # Assign data points to final centroids
    cluster_assignments = assignments(data, centroids)
    assignments_history.append(cluster_assignments)
    
    return centroids, cluster_assignments, centroids_history, assignments_history