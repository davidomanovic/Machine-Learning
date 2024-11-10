import sys
import os


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Paths relative to the train.py file's location
sentences_file = os.path.join(os.path.dirname(__file__), '../data/test_sentences.txt')
labels_file = os.path.join(os.path.dirname(__file__), '../data/labels.txt')
models_path = os.path.join(os.path.dirname(__file__), '../models/')
model_path = os.path.join(models_path, 'grammar_checker_model.pth')  # Full model path with filename

# Try importing from the src package
try:
    from src.grammar_checker import GrammarCheckerLSTM, load_model_and_vocab, preprocess_sentence, predict
except ModuleNotFoundError as e:
    print(f"Error: {e}")
    sys.exit(1)

# Function to load test sentences from the file
def load_test_sentences(file_path=sentences_file):
    sentences = []
    with open(file_path, 'r') as file:
        for line in file:
            sentences.append(line.strip())  # Strip any leading/trailing whitespace
    return sentences

def load_and_test_model(model_path=model_path):
    # Load the model and the vocabulary
    model, word_to_idx = load_model_and_vocab(model_path)

    # Define idx_to_label for prediction output
    idx_to_label = {0: 'incorrect', 1: 'correct'}

    # Load test sentences from the file
    test_sentences = load_test_sentences(sentences_file)

    # Test the model with the sentences
    for sentence in test_sentences:
        print(f"Sentence: {sentence}")
        prediction = predict(model, sentence, word_to_idx, idx_to_label)
        print(f"Prediction: {prediction}")
        print()



if __name__ == "__main__":
    load_and_test_model(model_path)
