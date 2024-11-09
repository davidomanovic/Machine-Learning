# Grammar Checker with LSTM

This project is a simple implementation of a Grammar Checker using a Long Short-Term Memory (LSTM) neural network model. It uses PyTorch for deep learning, and the goal is to classify whether the individual words in a sentence are grammatically correct or incorrect.

### Project Structure

- **src/**: Contains the core logic including the LSTM model, data preprocessing, training, and saving the model.
- **tests/**: Contains a testing script that loads the trained model, runs predictions, and prints the results.

### Requirements

This project requires Python 3 and the following dependencies:

- `torch`: PyTorch library for deep learning.
- `numpy`: Library for numerical operations.

You can install the required dependencies by running:

```bash
pip install -r requirements.txt
