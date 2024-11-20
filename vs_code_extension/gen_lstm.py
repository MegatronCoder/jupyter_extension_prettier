import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

class SentimentAnalyzer:
    def __init__(self, text_column='text', label_column='sentiment', max_length=100, max_words=10000):
        self.text_column = text_column
        self.label_column = label_column
        self.max_length = max_length
        self.max_words = max_words
        self.tokenizer = None
        self.label_encoder = None
        self.model = None
        
    def prepare_data(self, df):
        # Clean data
        df = df.dropna(subset=[self.text_column, self.label_column])
        df[self.text_column] = df[self.text_column].astype(str)
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        labels = self.label_encoder.fit_transform(df[self.label_column])
        
        # Tokenize texts
        self.tokenizer = Tokenizer(num_words=self.max_words)
        self.tokenizer.fit_on_texts(df[self.text_column])
        sequences = self.tokenizer.texts_to_sequences(df[self.text_column])
        padded_sequences = pad_sequences(sequences, maxlen=self.max_length, padding='post')
        
        return train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)
    
    def create_model(self):
        vocab_size = min(len(self.tokenizer.word_index) + 1, self.max_words)
        num_classes = len(self.label_encoder.classes_)
        
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, 128, input_length=self.max_length),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def train(self, df, epochs=1, batch_size=32):

        X_train, X_test, y_train, y_test = self.prepare_data(df)
        
        # Create and train model
        self.model = self.create_model()
        
        # Add early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping]
        )
        return history
    
    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]
            
        # Tokenize and pad
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_length, padding='post')
        
        # Get predictions
        predictions = self.model.predict(padded_sequences)
        predicted_labels = self.label_encoder.inverse_transform(predictions.argmax(axis=1))
        
        # Get confidence scores
        confidence_scores = predictions.max(axis=1)
        
        return list(zip(predicted_labels, confidence_scores))



# lets g0!

df = pd.read_csv(r'C:\Users\Manthan\Desktop\CV_DL_Practicals\sentiment_analysis.csv')
analyzer = SentimentAnalyzer(
    text_column='text',
    label_column='sentiment',
    max_length=1000,  
    max_words=10000  
)

# Train
history = analyzer.train(df, epochs=1)

# Make predictions
texts = [
    "This game is amazing!",
    "The service was terrible",
    "It's okay, nothing special"
]
predictions = analyzer.predict(texts)
for text, (sentiment, confidence) in zip(texts, predictions):
    print(f"Text: {text}")
    print(f"Sentiment: {sentiment} (Confidence: {confidence:.2f})\n")

