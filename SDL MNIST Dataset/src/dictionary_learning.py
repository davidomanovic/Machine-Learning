# src/dictionary_learning.py

import numpy as np
import matplotlib.pyplot as plt

class DictionaryLearningModel:
    def __init__(self):
        pass

    def perform_svd(self, matrix):
        U, Sigma, Vt = np.linalg.svd(matrix, full_matrices=False)
        return U, Sigma, Vt
        
    def truncSVD(self, U, Sigma, Vt, d):
        """
        Remove redundant basis vectors after the 'd'th column vector
    
        Parameters
        ----------
        U: unitary array (obtained from SVD) array_like
        Σ: rectangular diagonal matrix with
        non-negative real singular values (obtained from SVD)
        Vh: conjugate transpose of V (unitary matrix from SVD)
        d: factor to remove the (m-d) last columns of the SVD.
    
        Returns
        --------
        Low-rank matrix approximation of SVD matrices from an matrix A.
        Σ_d: truncated singular value matrix by size d
        W: truncated U (m x d) matrix
        H: the weights/latent variables
        """
        W = U[:, :d]  # Dictionary with orthogonal columns
        Sigma_d = Sigma[:d, :d]  # Reduced singular value matrix
        Vt_d = Vt[:d, :]  # Reduced unitary matrix
        H = Sigma_d @ Vt_d  # The weights
        return Sigma_d, W, H

    def orthproj(self, W, B):
        return W @ (W.T @ B) # Orthogonal projection of B onto dictionary
        
    def columndistance(self, P, B):
        return np.linalg.norm(B - P, axis = 0, ord = 2) # Euclidian distance per column

    def nnproj(self, W, A, maxiter = 50, delta = 10**(-10)):
        """
        Takes in a dictionary W and matrix A and calculates the
        non negative projection of A onto W.
    
        Parameters
        -----------
        W: non-negative dictionary
        A: Matrix for ENMF approach
        maxiter: max iterations for finding the weight
        delta: safe-division factor 
    
        Returns
        -----------
        nnP_W: Non-negative projection
        H_p: Non-negative weights/latent variables
        """
    
        # Initial constants to make calculations late more effcient
        WTA = W.T @ A
        WTW = W.T @ W
        H_p = np.random.uniform(0, 1, np.shape(WTA)) # Initial non-negative estimate for H_+

        for k in range(maxiter):
            H_p *= WTA / (WTW @ H_p + delta)
    
        nnP_W = W @ H_p # Calculate the non-negativ projection
        return nnP_W, H_p # Return the nn-projection and nn-weights/latent variables

    def decomposeTruncateData(self, A, d):
        # Calculate the SVD
        U, Sigma, Vh = np.linalg.svd(A, full_matrices=False)
        Sigma = np.diag(Sigma)
        # Returns Σ_d, W, H from the function truncSVD
        return truncSVD(U, Sigma, Vh, d)

    def reconstruct_matrix(self, U, Sigma, Vt):
        return U @ np.diag(Sigma) @ Vt

    def construct_ENMF(self, A, b, d):
        # Chooses d amount of random columnss of A
        indexes = np.random.choice(A.shape[1], d, replace=False)
    
        # Create the basis of the random columnss
        W_plus = A[:, indexes]
    
        # Calculates the projection and weights of the matrix in the basis of A
        P_plus, H_plus = nnproj(W_plus, b)
    
        # Returns P_plus, H_plus, W_plus
        return P_plus, H_plus, W_plus


