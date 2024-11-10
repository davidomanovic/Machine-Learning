import os
import tensorflow as tf
from datasets import load_dataset
from model.bert_lm import BertLM

# Load dataset
def load_text_data(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return lines

def create_dataset(text_data):
    # Use Hugging Face datasets library to create a simple dataset
    dataset = load_dataset('text', data_files={'train': text_data})
    return dataset

def main():
    # Initialize model
    model = BertLM()

    # Prepare data
    input_file = 'data/input_text.txt'
    text_data = load_text_data(input_file)
    dataset = create_dataset(text_data)

    # Train the model
    model.train(dataset['train'], epochs=3, batch_size=8)

    # Save the fine-tuned model
    model.save_model('saved_model/bert_lm')

if __name__ == "__main__":
    main()

