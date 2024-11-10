import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Paths relative to file's location
train_path = os.path.join(os.path.dirname(__file__), '../data/train.npy')
test_path = os.path.join(os.path.dirname(__file__), '../data/test.npy')

from src.dictionary_learning import DictionaryLearningModel
from src.visualize_MNIST import Plotter

import numpy as np

# Instantiate the model
model = DictionaryLearningModel()
plot = Plotter()

# Load the data and rescale
train = np.load(train_path)/255.0
test = np.load(test_path)/255.0

# Plot first 16 images of the nine integer
plot.plotimgs(test[:, 9, :], nplot=4)