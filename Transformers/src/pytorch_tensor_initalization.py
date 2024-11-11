import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Initializing tensors
#Initializing a scalar torch.tensor object
scalar = torch.tensor(7) 
print("A scalar tensor has dimension: ", scalar.ndim)
scalar.item() # turn tensor into Python int


# Initializing a vector torch.tensor object
vector = torch.tensor([7, 7]) 
print("A vector tensor has dimension: ", vector.ndim)


# Initializing a matrix torch.tensor object
MATRIX = torch.tensor([[7,7],
                      [7,7]])
print("A matrix tensor has dimension: ", MATRIX.ndim)


# Initializing a TENSOR torch.tensor object
TENSOR = torch.tensor([[[7, 7, 7],
                      [7, 7, 7],
                      [7, 7, 7]]])
print("A tensor tensor has dimension: ", MATRIX.ndim)

# Initialize a random tensor of size (2, 3)
random_tensor = torch.rand(2, 3)
print(random_tensor)

# Initialize tensor with similar shape to image (in pixels)
random_image_tensor = torch.rand(size=(224, 224, 3)) # height, width, color channels (RGB)

# Tensor from some range
first_ten_integers = torch.arange(start=0, end= 10, step=1)
print(first_ten_integers)

# Zeros
ten_zeros = torch.zeros_like(input=first_ten_integers)
print(ten_zeros)

# Float 32 tensor
float_32_tensor = torch.tensor([1.1, 2.2, 3.3],
                               dtype=None,
                               device=None,
                               requires_grad=False)
