# tests/test_main.py
import sys
sys.path.append("..")

import unittest
from src.k_means import generate_gaussian_mixture_data, kmeans
from src.plotting import plot_kmeans_iterations
import matplotlib.pyplot as plt

class TestGMM(unittest.TestCase):
    def test_now(self):
        k = 5   
        n_per_cluster = 300
        mean_range = (-5, 5)
        std_dev = 1.0
        d=2
        X = generate_gaussian_mixture_data(k, n_per_cluster, d, mean_range, std_dev)
        plt.scatter(X[:,0], X[:,1], c='b', marker='.')

        # apply kmeans() to the toy-data and plot the iterations.
        cf, af, centr, assigns = kmeans(X, k, max_iters=100, tol=1e-5)
        plot_kmeans_iterations(X, centr, assigns)

if __name__ == "__main__":
    unittest.main()


