import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Example dataset class
class GrammarDataset(Dataset):
    def __init__(self, sentences, labels):
        self.sentences = sentences
        self.labels = labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.labels[idx]

# Example sentences and labels (word-level labels)
X = [["She", "goes", "to", "the", "park"], ["He", "go", "to", "the", "store"]]
y = [[1, 1, 1, 1, 1], [1, 0, 1, 1, 1]]  # 1 = correct, 0 = incorrect

# Vocabulary and label mapping
word_to_idx = {"<PAD>": 0, "She": 1, "goes": 2, "to": 3, "the": 4, "park": 5, "He": 6, "go": 7, "store": 8}
label_to_idx = {1: 0, 0: 1}  # 1 = correct, 0 = incorrect

# Padding sentences
max_len = max(len(sentence) for sentence in X)
X_padded = [[word_to_idx[word] for word in sentence] + [0] * (max_len - len(sentence)) for sentence in X]
y_padded = [label + [0] * (max_len - len(label)) for label in y]

# Convert to torch tensors
X_tensor = torch.tensor(X_padded)
y_tensor = torch.tensor(y_padded)

# DataLoader
dataset = GrammarDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# LSTM Model for Sequence Labeling
class GrammarCheckerLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GrammarCheckerLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)
        return output

# Model initialization
input_dim = len(word_to_idx)  # Size of vocabulary
hidden_dim = 64
output_dim = 2  # Correct or Incorrect
model = GrammarCheckerLSTM(input_dim, hidden_dim, output_dim)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
for epoch in range(epochs):
    for batch in dataloader:
        sentences, labels = batch
        optimizer.zero_grad()
        outputs = model(sentences.float())
        loss = criterion(outputs.view(-1, 2), labels.view(-1))
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
