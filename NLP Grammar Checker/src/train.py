# src/train.py

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.grammar_checker import train_model, save_model_and_vocab

# Paths relative to the train.py file's location
sentences_file = os.path.join(os.path.dirname(__file__), '../data/sentences.txt')
labels_file = os.path.join(os.path.dirname(__file__), '../data/labels.txt')
models_path = os.path.join(os.path.dirname(__file__), '../models/')
model_path = os.path.join(models_path, 'grammar_checker_model.pth')  # Full model path with filename

def load_sentences_from_file(file_path):
    sentences = []
    with open(file_path, 'r') as file:
        for line in file:
            sentences.append(line.strip().split())
    return sentences

def load_labels_from_file(file_path):
    labels = []
    with open(file_path, 'r') as file:
        for line in file:
            labels.append(list(map(int, line.strip().split())))
    return labels

# Load sentences and labels from files
sentences = load_sentences_from_file(sentences_file)
labels = load_labels_from_file(labels_file)

def build_word_to_idx(sentences):
    word_to_idx = {"<PAD>": 0}
    index = 1
    for sentence in sentences:
        for word in sentence:
            if word not in word_to_idx:
                word_to_idx[word] = index
                index += 1
    return word_to_idx

word_to_idx = build_word_to_idx(sentences)
embedding_dim = 8

os.makedirs(models_path, exist_ok=True)  # Ensure models directory exists

# Train the model and save it along with word_to_idx
train_model(sentences, labels, word_to_idx, embedding_dim=embedding_dim, model_path=model_path)
