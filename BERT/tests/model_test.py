import os
from transformers import T5Tokenizer, TFT5ForConditionalGeneration

# Define the workspace path dynamically
workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
load_path = os.path.join(workspace_dir, 'trained_model')

# Load the tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = TFT5ForConditionalGeneration.from_pretrained(load_path)

# Testing the model with each sentence, adding "translate English to English:" to clarify language expectation
test_sentences = [
    "Answer the following in English only: What is the color of a red apple?"
]

# Function to prepare input, generate prediction, and decode
def generate_text(input_text):
    # Tokenize the input text
    input_ids = tokenizer(input_text, return_tensors="tf").input_ids
    
    # Generate the output using the model
    output = model.generate(input_ids, max_length=100, num_beams=5, early_stopping=False)
    
    # Decode the output to readable text
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return decoded_output

# Testing the model with each sentence
for sentence in test_sentences:
    print("Input:", sentence)
    print("Generated Output:", generate_text(sentence))
    print("-" * 50)
