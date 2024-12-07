import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dictionary_learning import DictionaryLearningModel as dict
from src.visualize_MNIST import Plotter as plot

class Accuracy:
    def __init__(self):
        pass

    def distance(self, B, imagebase, approach, digits, d = 32):
        """ 
        Takes in data in the form of images, and applies the EMNF approach on the data
        and and calculates the distances for the images to the basis of the images
        it is trained to. It does this using the method specified in the "approach" parameter.
        
        Parameters
        -----------
        B: Test dataset of images, stored columnsvise
        imagebase: base of all images that is used for training
        approach: type of approach to slolution (SVN, ENMF)
        digits: python list. Contains desired integers
        d: amount of data that is being removed
        

        Returns
        distances: the length values for the matrix B onto each digit basis.
        -----------
        """
        # Sets n as the shape of the collums of B
        n = B.shape[1]

        # Create empty arrays that have the size (len(digits), collums of B)
        distances = np.zeros((len(digits), n))

        # Turnes digits in to an numpy array so it can be used later
        digits = np.array(digits)

        # Decides method type
        if approach == "SVD":

            # For loop that goes trough the indexes and values of digits
            for (i, dig) in enumerate(digits):
                    
                    # Calculates the lengths to eatch image
                    distances[i] = plot.distances_SVD(imagebase[:, dig, :], d, B, onlyComp = True)
        
        elif approach == "ENMF":
            
            # For loop that goes trough the indexes and values of digits
            for (i, dig) in enumerate(digits):
                    
                    # Calculates the lengths to eatch image
                    distances[i] = dict.distances_ENMF(imagebase[:, dig, :], d, B, onlyComp = True)
    
        # Returns the distances
        return distances

    def guessImageAndCalculateAccuracies(self, digits, distances, correctIndicies):
        """ 
        Takes in some digits, distance sets and correctIndices and guesses which images each distance set corresponds 
        to and determines its own correctness and accuracy in doing so.

        Parameters
        -----------
        digits: which digits to guess from
        distances: a list containing sets of distance information of images for each digit.
        
        Returns
        accuracy: the total average accuracy
        gusses: the guesses made for each digit
        min_values: the lowest distance for each digit.
        -----------
        """
        # Guesses the indexes
        indices = np.argmin(distances, axis=0)

        # Converts index guesses to the correct value of the guess
        guesses = digits[indices] 

        # Finds the lowest distance for each image
        min_values = np.minimum(distances[0], distances[1], distances[2])

        # Calculate accuracy
        correctamount = np.sum(np.where(guesses == correctIndicies, 1, 0))
        accuracy = correctamount / len(indices)

        # Returns the accuracy, guesses and min_values
        return accuracy, guesses, min_values

    def findClassRecalls(self, guesses, correctIndicies):
        """ 
        Takes in some digits, distance sets and correctIndices and guesses which images each distance set corresponds 
        to and determines its own correctness and accuracy in doing so.

        Parameters
        -----------
        guesses: which digits to guess from
        correctIndicies: a list containing what the correct image each guess corresponds to

        Returns
        recalls: a dictionary containing the accuracy of guesses for each digit.
        -----------
        """
        # Creates an empty python dictonary
        recalls = {}

        # Create the keys for all of the potential integers
        class_counts = {digit: {'tp': 0, 'fn': 0} for digit in range(10)}

        # For loop that goes trough all of the guesses
        for (predicted, actual) in zip(guesses, correctIndicies):

            # Counts up all of the correct guesses, and wrong guesses
            if predicted == actual:
                class_counts[actual]['tp'] += 1
            else:
                class_counts[actual]['fn'] += 1

        # For loop that goes trough all of the potetial integers
        for digit in range(10):

            # Create varibles for the amount of rigth and wrong guesses
            tp = class_counts[digit]['tp']
            fn = class_counts[digit]['fn']

            # Calculates the accuracy of the guesses
            if tp + fn == 0:
                recalls[digit] = 0.0
            else:
                recalls[digit] = tp / (tp + fn)
            
        # Returns the recalls
        return recalls

    def prettyTable(self, dict, digits):
        """
        Print a pretty table from a dictionary of recalls. Is used to print the
        result of the test.
        """

        # Prints start of table
        print("\n---- Recalls ----")
        print("+-----+--------+")
        print("|Class|Accuracy|")
        print("+-----+--------+")

        # Prints all of the results in the table
        for key, value in dict.items():
            if key in digits:
                print(f"|  {key}  | {value:.4f} |")
        
        # Prints the end of the table
        print("+-----+--------+")
