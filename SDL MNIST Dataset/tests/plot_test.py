import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dictionary_learning import DictionaryLearningModel
from src.visualize_MNIST import Plotter

import numpy as np

# Instantiate the model
model = DictionaryLearningModel()
plot = Plotter()

# Load the data and rescale
train = np.load('data/train.npy')/255.0
test = np.load('data/test.npy')/255.0

# Plot first 16 images of the nine integer
plot.plotimgs(test[:, 9, :], nplot=4)