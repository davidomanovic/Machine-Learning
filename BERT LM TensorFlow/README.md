# BERT Language Model (LM) with TensorFlow

This project demonstrates fine-tuning a BERT model for language modeling using TensorFlow. 
Bidirectional encoder representations from transformers (BERT) is a language model introduced in October 2018 by researchers at Google. It learns to represent text as a sequence of vectors using self-supervised learning. It uses the encoder-only transformer architecture.

## Prerequisites

- TensorFlow 2.x
- Hugging Face Transformers library
- Datasets library from Hugging Face

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare your text data in the `data/input_text.txt` file.

3. To train the model, run:
```bash
python scripts/train.py
```


4. The model will be saved to the `saved_model/bert_lm` directory.

## Notes

This is a simple setup for fine-tuning BERT. You can modify it for larger datasets or more advanced features such as early stopping, distributed training, etc.
