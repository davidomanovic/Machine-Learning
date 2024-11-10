import tensorflow as tf
from transformers import TFBertForMaskedLM, BertTokenizer

class BertLM:
    def __init__(self, model_name='bert-base-uncased'):
        # Load the BERT tokenizer and model for masked language modeling
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = TFBertForMaskedLM.from_pretrained(model_name)
    
    def encode_input(self, text):
        """Encode the text to the format BERT understands."""
        return self.tokenizer(text, return_tensors='tf', padding=True, truncation=True)

    def train(self, dataset, epochs=3, batch_size=8):
        """Fine-tune the model on a dataset."""
        # Prepare the dataset
        train_dataset = dataset.map(lambda x: self.encode_input(x['text']), batched=True)
        train_dataset = train_dataset.shuffle(1000).batch(batch_size)
        
        # Compile the model
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5), 
                           loss=self.model.compute_loss)
        
        # Train the model
        self.model.fit(train_dataset, epochs=epochs)

    def save_model(self, save_path):
        """Save the model to the specified directory."""
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
    
    def load_model(self, load_path):
        """Load a fine-tuned model."""
        self.model = TFBertForMaskedLM.from_pretrained(load_path)
        self.tokenizer = BertTokenizer.from_pretrained(load_path)

