# src/train.py
from src.grammar_checker import train_model

# Example data (sentences and labels)
sentences = [
    ["She", "goes", "to", "the", "park"],
    ["He", "go", "to", "the", "store"],
    ["I", "am", "reading", "a", "book"],
    ["She", "not", "like", "pizza"]
]

# Labels (1 = correct, 0 = incorrect)
labels = [
    [1, 1, 1, 1, 1],
    [1, 0, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 0, 1, 1, 1]
]

# Vocabulary (index for each word)
word_to_idx = {"<PAD>": 0, "She": 1, "goes": 2, "to": 3, "the": 4, "park": 5, "He": 6, "go": 7, "store": 8}

# Train the model
train_model(sentences, labels, word_to_idx)
