import sys
import os

# Get the absolute path to the src directory
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))

# Debug print to check the paths
print(f"Adding to sys.path: {src_path}")

# Add src to sys.path
sys.path.insert(0, src_path)

# Print the sys.path to verify
print(f"sys.path after insert: {sys.path}")

# Try importing from the src package
try:
    from grammar_checker import GrammarCheckerLSTM, load_model_and_vocab, preprocess_sentence, predict
except ModuleNotFoundError as e:
    print(f"Error: {e}")
    sys.exit(1)

# Function to load test sentences from the file
def load_test_sentences(file_path='data/test_sentences.txt'):
    sentences = []
    with open(file_path, 'r') as file:
        for line in file:
            sentences.append(line.strip())  # Strip any leading/trailing whitespace
    return sentences

def load_and_test_model(model_path='models/grammar_checker_model.pth'):
    # Load the model and the vocabulary
    model, word_to_idx = load_model_and_vocab(model_path)

    # Define idx_to_label for prediction output
    idx_to_label = {0: 'incorrect', 1: 'correct'}

    # Load test sentences from the file
    test_sentences = load_test_sentences('data/test_sentences.txt')

    # Test the model with the sentences
    for sentence in test_sentences:
        print(f"Sentence: {sentence}")
        prediction = predict(model, sentence, word_to_idx, idx_to_label)
        print(f"Prediction: {prediction}")
        print()



if __name__ == "__main__":
    load_and_test_model('models/grammar_checker_model.pth')
