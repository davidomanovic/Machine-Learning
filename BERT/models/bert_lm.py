from transformers import TFBertForMaskedLM, BertTokenizer
import tensorflow as tf
import numpy as np

class BertLM:
    def __init__(self):
        # Load pre-trained BERT model and tokenizer
        self.model = TFBertForMaskedLM.from_pretrained('bert-base-uncased',
                                                      hidden_dropout_prob=0.3)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def encode_input(self, texts, max_length=512):
        return self.tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='np'
        )

    def prepare_dataset(self, dataset, batch_size):
        input_ids = []
        attention_masks = []
        for example in dataset:
            encoded = self.encode_input(example['text'])
            input_ids.append(encoded['input_ids'])
            attention_masks.append(encoded['attention_mask'])

        input_ids = np.array(input_ids)
        attention_masks = np.array(attention_masks)

        input_ids = np.squeeze(input_ids, axis=1)  # Remove the extra dimension
        attention_masks = np.squeeze(attention_masks, axis=1)

        dataset = tf.data.Dataset.from_tensor_slices((input_ids, attention_masks))
        return dataset.batch(batch_size)

    def train(self, dataset, epochs, batch_size):
        train_dataset = self.prepare_dataset(dataset, batch_size)

        # Implement learning rate warm-up and decay
        lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=5e-6,  # Start with a much smaller learning rate
            first_decay_steps=1000,
            t_mul=2.0,
            m_mul=1.0,
            alpha=0.0
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, weight_decay=1e-5)

        # Compile the model with Adam optimizer and sparse categorical cross entropy
        self.model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')

        # Gradient clipping
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipvalue=1.0)

        # Train the model with early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)

        self.model.fit(train_dataset, epochs=epochs, callbacks=[early_stopping])

    def save_model(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
