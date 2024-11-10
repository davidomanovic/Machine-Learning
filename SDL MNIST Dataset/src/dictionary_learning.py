# src/dictionary_learning.py

import numpy as np
import matplotlib.pyplot as plt

class DictionaryLearningModel:
    def __init__(self):
        pass

    def perform_svd(self, matrix):
        U, Sigma, Vt = np.linalg.svd(matrix, full_matrices=False)
        return U, Sigma, Vt

    def reconstruct_matrix(self, U, Sigma, Vt):
        return U @ np.diag(Sigma) @ Vt

    def visualize_matrix(self, matrix, title="Matrix Visualization"):
        plt.imshow(matrix, cmap='viridis')
        plt.colorbar()
        plt.title(title)
        plt.show()
