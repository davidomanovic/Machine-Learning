import os
from transformers import T5Tokenizer, TFT5ForConditionalGeneration
import tensorflow as tf

# Define the workspace path dynamically
workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
model_path = os.path.join(workspace_dir, 'trained_model', 'tf_model.h5')

# Load the tokenizer and model (only the tokenizer needs to be loaded from Hugging Face)
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Load the model directly with TensorFlow (skip using Hugging Face's from_pretrained method)
model = TFT5ForConditionalGeneration.from_pretrained(model_path)

# Test the model with an example
input_text = "question answering: What is the role of attention in transformer models?"
input_ids = tokenizer(input_text, return_tensors="tf").input_ids
output = model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True)
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
print(decoded_output)
