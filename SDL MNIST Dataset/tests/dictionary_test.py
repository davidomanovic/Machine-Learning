from src.dictionary_learning import DictionaryLearningModel
import numpy as np

# Define example matrices for testing
A1 = np.array([[1000, 1], [0, 1], [0, 0]], dtype=np.double)
A2 = np.array([[1, 0, 0], [1, 0, 0], [0, 0, 1]], dtype=np.double)

# Instantiate the model
model = DictionaryLearningModel()

# Test with matrix A1
U, Sigma, Vt = model.perform_svd(A1)
reconstructed_A1 = model.reconstruct_matrix(U, Sigma, Vt)

# Print results for A1
print("Original Matrix A1:")
print(A1)
print("\nSVD Decomposition of A1:")
print("U:", U)
print("Sigma:", Sigma)
print("Vt:", Vt)
print("\nReconstructed Matrix A1:")
print(reconstructed_A1)

# Visualize reconstructed A1
model.visualize_matrix(reconstructed_A1, title="Reconstructed A1")
