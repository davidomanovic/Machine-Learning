# Grammar Checker with LSTM

This project is a simple implementation of a Grammar Checker using a Long Short-Term Memory (LSTM) neural network model. It uses PyTorch for deep learning, and the goal is to classify whether the individual words in a sentence are grammatically correct or incorrect. This is compatible with VSCode, or you can directly run in terminal :)

### Project Structure

- **data/**: Contains all training data and example sentences. Play around and edit them to your liking (just make sure they are reasonably similar to the vocabulary in the test set)!
- **src/**: Contains the core logic including the LSTM model, data preprocessing, training, and saving the model. If you want to train the model, you have to run the `train.py`
- **tests/**: Contains a testing script that loads the trained model, runs predictions, and prints the results.
- **models/**: Contains the model of the Grammar Checker using the LSTM model

### Requirements

1. This project requires Python 3. You can install the required dependencies by running:

```bash
pip install -r requirements.txt
```

2. Prepare your text data in the `data/sentences.txt` file. I have prefilled 30 sentences, and also test sentences.

3. To train the model, run:
```bash
python src/train.py
```

4. The model will be saved to the `models/grammar_checker_model.pth` directory.

5. In `tests/` you can simply run the code after training the model and generating the aforementioned file.
