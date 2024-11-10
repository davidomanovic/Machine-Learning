import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dictionary_learning import DictionaryLearningModel
from src.accuracy import Accuracy

import numpy as np

# Instantiate the model
dict = DictionaryLearningModel()
acc = Accuracy()

# Generate a test set containing 3 different digits
digits = np.array([6,7,8])
train_path = os.path.join(os.path.dirname(__file__), '../data/train.npy')
test_path = os.path.join(os.path.dirname(__file__), '../data/test.npy')

def generate_test(test, digits = np.array([0,1,2]), N = 800):
        assert N <= test.shape[2] , "N needs to be smaller than or equal to the total amount of available test data for each class"
        assert len(digits)<= 10, "List of digits can only contain up to 10 digits"

        # Arrays to store test set and labels
        test_sub = np.zeros((test.shape[0], len(digits)*N))
        test_labels = np.zeros(len(digits)*N)

        # Iterate over all digit classes and store test data and labels
        for i, digit in enumerate(digits):
            test_sub[:, i*N:(i+1)*N] = test[:,digit,:]
            test_labels[i*N:(i+1)*N] = digit

        # Indexes to be shuffled 
        ids = np.arange(0,len(digits)*N)

        # Shuffle indexes
        np.random.shuffle(ids)

        # Return shuffled data 
        return test_sub[:,ids], test_labels[ids]

# Load the data and rescale
train = np.load(train_path)/255.0
test = np.load(test_path)/255.0

# Runs the generate_test function for the test data
B, B_labels = generate_test(test, digits, N = 800)

print("SVD: \n")

# Dose the calculation for the test with the SVD method
distances = acc.distance(B, train, "SVD", digits, d = 32)
accuracy, guesses, min_values = acc.guessImageAndCalculateAccuracies(digits, distances, B_labels)
recalls = acc.findClassRecalls(guesses, B_labels)

# Prints the result of the test
print("Accuracy: ", accuracy*100, "%")
print("Our guesses: ", guesses)
print("Lowest values of norm for each image: ", min_values)

acc.prettyTable(recalls, digits)


print("\nENMF")

# Dose the calculation for the test with the ENMF method
distances = acc.distance(B, train, "ENMF", digits, d = 32)
accuracy, guesses, min_values = acc.guessImageAndCalculateAccuracies(digits, distances, B_labels)
recalls = acc.findClassRecalls(guesses, B_labels)

# Prints the result of the test
print("Accuracy: ", accuracy*100, "%")
print("Our guesses: ", guesses)
print("Lowest values of norm for each image: ", min_values)

print("-------- Recalls --------")
acc.prettyTable(recalls, digits)
