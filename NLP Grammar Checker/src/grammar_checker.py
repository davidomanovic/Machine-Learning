# src/grammar_checker.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import numpy as np

# Define the LSTM model for grammar checking
class GrammarCheckerLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GrammarCheckerLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)
        return output

# Dataset for grammar checking
class GrammarDataset(Dataset):
    def __init__(self, sentences, labels, word_to_idx, max_len=5):
        self.sentences = sentences
        self.labels = labels
        self.word_to_idx = word_to_idx
        self.max_len = max_len

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
    sentence_idx = [word_to_idx.get(word, 0) for word in words]
    padded_sentence = sentence_idx + [0] * (max_len - len(sentence_idx))
    return torch.tensor([padded_sentence])  # Return as a batch

# Save the trained model
def save_model(model, path):
    torch.save(model.state_dict(), path)

# Load the trained model
def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()

# Train the model
def train_model(sentences, labels, word_to_idx, model_path='grammar_checker_model.pth', epochs=10, batch_size=2):
    input_dim = len(word_to_idx)  # Size of vocabulary
    hidden_dim = 64
    output_dim = 2  # Correct or Incorrect

    # Prepare dataset and dataloader
    dataset = GrammarDataset(sentences, labels, word_to_idx)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = GrammarCheckerLSTM(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(epochs):
        for batch in dataloader:
            sentences, labels = batch
            optimizer.zero_grad()
            outputs = model(sentences.float())
            loss = criterion(outputs.view(-1, 2), labels.view(-1))
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    # Save the model after training
    save_model(model, model_path)

# Predict whether words are grammatically correct
def predict(model, sentence, word_to_idx, max_len=5):
    sentence_tensor = preprocess_sentence(sentence, word_to_idx, max_len)
    with torch.no_grad():
        output = model(sentence_tensor.float())

    predicted_labels = output.argmax(dim=-1).squeeze().numpy()
    return predicted_labels
