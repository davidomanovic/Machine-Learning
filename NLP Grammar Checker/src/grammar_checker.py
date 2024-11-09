# src/grammar_checker.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import numpy as np

# Define the LSTM model for grammar checking
class GrammarCheckerLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout_prob=0.5):
        super(GrammarCheckerLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_prob)  # Dropout layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # Adjust for bidirectional LSTM

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        x = self.dropout(lstm_out)
        output = self.fc(x)
        return output
        

# Dataset for grammar checking
class GrammarDataset(Dataset):
    def __init__(self, sentences, labels, word_to_idx, max_len=None):
        self.sentences = sentences
        self.labels = labels
        self.word_to_idx = word_to_idx
        # Dynamically determine max_len if not provided
        self.max_len = max_len or max(len(sentence) for sentence in sentences)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]
        sentence_idx = [self.word_to_idx.get(word, 0) for word in sentence]
        padded_sentence = sentence_idx + [0] * (self.max_len - len(sentence_idx))
        return torch.tensor(padded_sentence), torch.tensor(label + [0] * (self.max_len - len(label)))

# Preprocess the input sentence
def preprocess_sentence(sentence, word_to_idx, max_len=5):
    words = sentence.split()
    sentence_idx = [word_to_idx.get(word, 0) for word in words]  # Use 0 (PAD) for unknown words
    padded_sentence = sentence_idx + [0] * (max_len - len(sentence_idx))  # Pad sentence to max_len
    return torch.tensor([padded_sentence])  # Return as a batch (shape: [1, max_len])

# Save the trained model and vocabulary
def save_model_and_vocab(model, path, word_to_idx):
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': word_to_idx
    }, path)

# Load the trained model and vocabulary
def load_model_and_vocab(path):
    checkpoint = torch.load(path)
    model = GrammarCheckerLSTM(vocab_size=len(checkpoint['vocab']), embedding_dim=8, hidden_dim=64, output_dim=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    word_to_idx = checkpoint['vocab']
    return model, word_to_idx

# Train the model with early stopping
def train_model(sentences, labels, word_to_idx, model_path='models/grammar_checker_model.pth', epochs=50, batch_size=2, embedding_dim=8, patience=3):
    vocab_size = len(word_to_idx)
    hidden_dim = 64
    output_dim = 2  # Correct or Incorrect

    dataset = GrammarDataset(sentences, labels, word_to_idx)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model with embedding layer
    model = GrammarCheckerLSTM(vocab_size, embedding_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    best_loss = float('inf')  # Initialize the best loss as a large number
    epochs_since_improvement = 0  # Count the number of epochs since the last improvement

    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        epoch_loss = 0  # Track the total loss for this epoch

        for batch in dataloader:
            sentences, labels = batch
            optimizer.zero_grad()

            # Forward pass
            outputs = model(sentences)  # Output shape: (batch_size, max_len, output_dim)
            outputs = outputs.view(-1, output_dim)  # Flatten the output to match the label shape
            labels = labels.view(-1)  # Flatten the labels

            # Compute loss
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()  # Add the loss for this batch to the epoch total

        avg_epoch_loss = epoch_loss / len(dataloader)  # Average loss for the epoch
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_epoch_loss:.4f}")

        # Check if the current loss is better than the best observed loss
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            epochs_since_improvement = 0  # Reset the counter
            # Save model if it improves
            save_model_and_vocab(model, model_path, word_to_idx)
        else:
            epochs_since_improvement += 1

        # Stop early if the model hasn't improved for 'patience' epochs
        if epochs_since_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

# Prediction function to get grammar correction suggestions
def predict(model, sentence, word_to_idx, idx_to_label, max_len=5):
    # Ensure model is in evaluation mode
    model.eval()  # This is important to disable dropout and batch normalization

    # Convert the sentence into a tensor using word_to_idx
    sentence_tensor = preprocess_sentence(sentence, word_to_idx, max_len)
    
    # Make the prediction using the trained model
    with torch.no_grad():
        output = model(sentence_tensor.long())  # Pass the sentence through the model
    
    # Squeeze the output to remove the batch dimension
    output = output.squeeze(0)  # [seq_len, 2] (removes batch dimension)
    
    predictions = []
    for i in range(output.shape[0]):
        word_logits = output[i]  # Get the logits for each word
        predicted_label = word_logits.argmax(dim=-1).item()  # Get the index of the max value (either 0 or 1)
        result = idx_to_label[predicted_label]  # Map index to label (0 -> 'incorrect', 1 -> 'correct')
        predictions.append(result)
    
    # Return the predictions for each word in the sentence
    return predictions

# Debugging predictions to show word-wise predictions
def debug_predictions(model, sentence, word_to_idx, idx_to_label, max_len=5):
    model.eval()  # Set the model to evaluation mode
    sentence_tensor = preprocess_sentence(sentence, word_to_idx, max_len)
    
    with torch.no_grad():
        output = model(sentence_tensor.long())
    
    output = output.squeeze(0)  # Remove batch dimension
    for i in range(output.shape[0]):
        word_logits = output[i]
        predicted_label = word_logits.argmax(dim=-1).item()  # Predict the label (0 or 1)
        print(f"Word: {sentence.split()[i]}, Prediction: {idx_to_label[predicted_label]}")
