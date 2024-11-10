import numpy as np

class accuracy:
  def __init__(self):
    pass

  def distance(B, imagebase, approach, digits, d = 32):
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
                distances[i] = distances_SVD(imagebase[:, dig, :], d, B, onlyComp = True)
    
    elif approach == "ENMF":
         
         # For loop that goes trough the indexes and values of digits
         for (i, dig) in enumerate(digits):
                
                # Calculates the lengths to eatch image
                distances[i] = distances_ENMF(imagebase[:, dig, :], d, B, onlyComp = True)
    
    # Returns the distances
    return distances

def guessImageAndCalculateAccuracies(digits, distances, correctIndicies):
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

def findClassRecalls(guesses, correctIndicies):
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

  def generate_test(test, digits = np.array([0,1,2]), N = 800):

    """
    Randomly generates test set.
    input:
        test: numpy array. Should be the test data loaded from file
        digits: python list. Contains desired integers
        N: int. Amount of test data for each class
    output:
        test_sub: (784,len(digits)*N) numpy array. Contains len(digits)*N images
        test_labels: (len(digits)*N) numpy array. Contains labels corresponding to the images of test_sub
    """

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

def prettyTable(dict, digits):
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
