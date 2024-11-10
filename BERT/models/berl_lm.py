import os
import tensorflow as tf
from transformers import TFBertForMaskedLM, BertTokenizer, AdamW, get_linear_schedule_with_warmup

class BertLM(tf.keras.Model):
    def __init__(self, model_name='bert-base-uncased'):
        # Load the BERT tokenizer and model for masked language modeling
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = TFBertForMaskedLM.from_pretrained(model_name)
    
    def encode_input(self, text, max_length=512):
        """Encode the text to the format BERT understands."""
        return self.tokenizer(text, return_tensors='tf', padding='max_length', truncation=True, max_length=max_length)

    def train(self, dataset, epochs=3, batch_size=8):
        """Fine-tune the model on a dataset."""
        # Prepare the dataset
        train_dataset = dataset.map(lambda x: self.encode_input(x['text']), batched=True)
        train_dataset = train_dataset.shuffle(1000).batch(batch_size)

        # Prepare the optimizer and learning rate schedule
        num_train_steps = len(train_dataset) * epochs
        optimizer = AdamW(self.model.parameters(), lr=5e-5, weight_decay=0.01)
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps=0, 
                                                    num_training_steps=num_train_steps)

        # Compile the model
        self.model.compile(optimizer=optimizer, loss=self.model.compute_loss)

        # Train the model
        self.model.fit(train_dataset, epochs=epochs, verbose=1)

    def save_model(self, save_path):
        """Save the model to the specified directory."""
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    def load_model(self, load_path):
        """Load a fine-tuned model."""
        self.model = TFBertForMaskedLM.from_pretrained(load_path)
        self.tokenizer = BertTokenizer.from_pretrained(load_path)
