# src/train.py

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.grammar_checker import train_model, save_model_and_vocab

# Define the file paths to your sentences and labels
sentences_file = 'data/sentences.txt'
labels_file = 'data/labels.txt'

def load_sentences_from_file(file_path):
    sentences = []
    with open(file_path, 'r') as file:
        for line in file:
            # Split each line into words and strip any excess whitespace
            sentences.append(line.strip().split())
    return sentences


def load_labels_from_file(file_path):
    labels = []
    with open(file_path, 'r') as file:
        for line in file:
            # Convert the space-separated label values into a list of integers
            labels.append(list(map(int, line.strip().split())))
    return labels

# Load sentences and labels from files
sentences = load_sentences_from_file(sentences_file)
labels = load_labels_from_file(labels_file)

# Generate word_to_idx dynamically from the sentences
def build_word_to_idx(sentences):
    word_to_idx = {"<PAD>": 0}  # We reserve 0 for padding token
    index = 1  # Start from index 1 for the words

    # Iterate through each sentence and each word in the sentence
    for sentence in sentences:
        for word in sentence:
            if word not in word_to_idx:
                word_to_idx[word] = index
                index += 1
    return word_to_idx

word_to_idx = build_word_to_idx(sentences)

# Train the model and save both the model and vocabulary
embedding_dim = 8
model_path = 'models/grammar_checker_model.pth'


# Train the model and save the model along with word_to_idx
train_model(sentences, labels, word_to_idx, embedding_dim=embedding_dim, model_path=model_path)
