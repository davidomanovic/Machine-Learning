# tests/test_grammar_checker.py
import torch
from src.grammar_checker import GrammarCheckerLSTM, load_model, preprocess_sentence, predict

# Define vocabulary and label mappings
word_to_idx = {"<PAD>": 0, "She": 1, "goes": 2, "to": 3, "the": 4, "park": 5, "He": 6, "go": 7, "store": 8}
label_to_idx = {1: 0, 0: 1}  # 1 = correct, 0 = incorrect
idx_to_label = {0: 'correct', 1: 'incorrect'}

# Load the trained model
def load_and_test_model(model_path='grammar_checker_model.pth'):
    input_dim = len(word_to_idx)
    hidden_dim = 64
    output_dim = 2  # Correct or Incorrect

    # Initialize model
    model = GrammarCheckerLSTM(input_dim, hidden_dim, output_dim)
    load_model(model, model_path)

    # Test sentences
    sentences = [
        "She goes to the park.",
        "He go to the store.",
        "I am reading a book.",
        "She not like pizza."
    ]

    for sentence in sentences:
        print(f"Sentence: {sentence}")
        predictions = predict(model, sentence, word_to_idx)
        result = [idx_to_label[label] for label in predictions]
        print(f"Prediction: {result}")
        print()

if __name__ == "__main__":
    load_and_test_model('grammar_checker_model.pth')
