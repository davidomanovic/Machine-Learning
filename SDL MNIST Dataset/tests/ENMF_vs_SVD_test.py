from src.dictionary_learning import dictionary_learning, accuracy
import numpy as np

# Generate a test set containing 3 different digits
digits = np.array([6,7,8])

# Load the data and rescale
train = np.load('train.npy')/255.0
test = np.load('test.npy')/255.0

# Runs the generate_test function for the test data
B, B_labels = generate_test(test, digits, N = 800)


print("SVD: \n")

# Dose the calculation for the test with the SVD method
distances = distance(B, train, "SVD", digits, d = 32)
accuracy, guesses, min_values = guessImageAndCalculateAccuracies(digits, distances, B_labels)
recalls = findClassRecalls(guesses, B_labels)

# Prints the result of the test
print("Accuracy: ", accuracy*100, "%")
print("Our guesses: ", guesses)
print("Lowest values of norm for each image: ", min_values)

prettyTable(recalls, digits)


print("\nENMF")

# Dose the calculation for the test with the ENMF method
distances = distance(B, train, "ENMF", digits, d = 32)
accuracy, guesses, min_values = guessImageAndCalculateAccuracies(digits, distances, B_labels)
recalls = findClassRecalls(guesses, B_labels)

# Prints the result of the test
print("Accuracy: ", accuracy*100, "%")
print("Our guesses: ", guesses)
print("Lowest values of norm for each image: ", min_values)

print("-------- Recalls --------")
prettyTable(recalls, digits)
