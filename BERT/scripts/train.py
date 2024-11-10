import os
import sys
import tensorflow as tf
from datasets import load_dataset
# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.bert_lm import BertLM

# Load dataset
def load_text_data(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return lines

def create_dataset(file_path):
    # Use Hugging Face datasets library to create a simple dataset
    dataset = load_dataset('text', data_files={'train': file_path})
    return dataset

def main():
    # Initialize model
    model = BertLM()

    # Prepare data
    input_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'input_text.txt')
    dataset = create_dataset(input_file) 
    
    # Tokenize and prepare the dataset for training
    def tokenize_function(examples):
        return model.encode_input(examples['text'])  # Use the encode_input method to tokenize

    # Map the tokenize function to the dataset
    dataset = dataset.map(tokenize_function, batched=True)

    # Train the model
    model.train(dataset['train'], epochs=5, batch_size=8)

    save_path = os.path.join(os.path.dirname(__file__), '..','trained_model')
    # Save the fine-tuned model
    model.save_model(save_path)

if __name__ == "__main__":
    main()
