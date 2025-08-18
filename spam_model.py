import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, SpatialDropout1D, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re
import json
import os

tf.random.set_seed(42)
np.random.seed(42)

class SpamDetectionAnalyzer:
    def __init__(self, max_features=5000, max_length=50):
        self.max_features = max_features
        self.max_length = max_length
        self.tokenizer = None
        self.label_encoder = None
        self.model = None
        
    def preprocess_text(self, text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        # Remove URLs, phone numbers, and special characters common in spam
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'\b\d{4,}\b', '', text)  # Remove long numbers (phone numbers, codes)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return re.sub(r'\s+', ' ', text).strip()
    
    def prepare_data(self, df):
        df['cleaned_text'] = df['message'].apply(self.preprocess_text)
        self.label_encoder = LabelEncoder()
        df['label_encoded'] = self.label_encoder.fit_transform(df['label'])
        self.tokenizer = Tokenizer(num_words=self.max_features, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(df['cleaned_text'])
        X = pad_sequences(self.tokenizer.texts_to_sequences(df['cleaned_text']), maxlen=self.max_length)
        y = tf.keras.utils.to_categorical(df['label_encoded'])
        return X, y
    
    def build_model(self, num_classes):
        # TFLite-friendly architecture optimized for Android compatibility
        self.model = Sequential([
            Embedding(self.max_features, 128, input_length=self.max_length),
            SpatialDropout1D(0.3),
            # Use multiple Conv1D layers instead of complex operations
            tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same'),
            tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same'),
            tf.keras.layers.GlobalMaxPooling1D(),
            # Simpler dense layers for better TFLite conversion
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
        
        # Use a slightly lower learning rate for better stability
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.model.compile(
            optimizer=optimizer, 
            loss='categorical_crossentropy', 
            metrics=['accuracy']
        )
    
    def train(self, df, epochs=25, batch_size=32):
        X, y = self.prepare_data(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, 
                                                            stratify=np.argmax(y, axis=1))
        
        self.build_model(len(self.label_encoder.classes_))
        
        # Enhanced callbacks for better training
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=7,
                restore_best_weights=True,
                verbose=1,
                mode='max'
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=0.00001,
                verbose=1
            )
        ]
        
        print(f"Training on {len(X_train)} samples, validating on {len(X_test)} samples")
        print(f"Classes: {self.label_encoder.classes_}")
        
        # Train with enhanced parameters
        history = self.model.fit(
            X_train, y_train, 
            batch_size=batch_size, 
            epochs=epochs, 
            validation_data=(X_test, y_test), 
            callbacks=callbacks,
            verbose=1
        )
        
        # Comprehensive evaluation
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\n=== Final Model Performance ===")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Show per-class performance
        test_predictions = self.model.predict(X_test, verbose=0)
        test_pred_classes = np.argmax(test_predictions, axis=1)
        test_true_classes = np.argmax(y_test, axis=1)
        
        print(f"\n=== Per-Class Performance ===")
        for i, class_name in enumerate(self.label_encoder.classes_):
            true_count = np.sum(test_true_classes == i)
            pred_count = np.sum(test_pred_classes == i)
            correct_count = np.sum((test_true_classes == i) & (test_pred_classes == i))
            
            precision = correct_count / pred_count if pred_count > 0 else 0
            recall = correct_count / true_count if true_count > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"{class_name:8}: Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        
        return history
    
    def predict(self, text):
        cleaned = self.preprocess_text(text)
        X = pad_sequences(self.tokenizer.texts_to_sequences([cleaned]), maxlen=self.max_length)
        pred = self.model.predict(X, verbose=0)[0]
        return {
            'label': self.label_encoder.inverse_transform([np.argmax(pred)])[0],
            'confidence': float(np.max(pred))
        }
    
    def save_assets(self, output_dir='spam_model_assets'):
        os.makedirs(output_dir, exist_ok=True)
        self.model.save(f'{output_dir}/model.h5')
        with open(f'{output_dir}/tokenizer.json', 'w') as f:
            json.dump({
                'word_index': self.tokenizer.word_index,
                'num_words': self.tokenizer.num_words,
                'max_length': self.max_length
            }, f)
        
        with open(f'{output_dir}/labels.json', 'w') as f:
            json.dump({'classes': self.label_encoder.classes_.tolist()}, f)
    
    def convert_to_tflite(self, output_dir='spam_model_assets'):
        os.makedirs(output_dir, exist_ok=True)
        
        def rep_data():
            # More diverse representative dataset for better calibration
            texts = [
                "free money now", "hey how are you", "win cash prizes", "call me", "urgent offer",
                "congratulations winner", "hello friend", "meeting tomorrow", "limited time offer",
                "click here now", "good morning", "how was your day", "special discount today",
                "urgent action required", "thanks for yesterday", "see you later", "amazing deal",
                "your account", "normal conversation", "spam message example"
            ]
            for text in texts:
                X = pad_sequences(self.tokenizer.texts_to_sequences([self.preprocess_text(text)]), 
                                maxlen=self.max_length)
                yield [X.astype(np.float32)]
        
        # Multiple conversion strategies - prioritizing accuracy and Android compatibility
        conversion_strategies = [
            {
                'name': 'android_optimized',
                'description': 'Full float32 model - best accuracy, larger size',
                'config': lambda c: None  # No quantization at all
            },
            {
                'name': 'dynamic_range',
                'description': 'Dynamic range quantization - good balance',
                'config': lambda c: setattr(c, 'optimizations', [tf.lite.Optimize.DEFAULT])
            },
            {
                'name': 'float16_accurate',
                'description': 'Float16 quantization - smaller with good accuracy',
                'config': lambda c: (
                    setattr(c, 'optimizations', [tf.lite.Optimize.DEFAULT]),
                    setattr(c, 'target_spec', tf.lite.TargetSpec(supported_types=[tf.float16]))
                )[-1]
            },
            {
                'name': 'hybrid_quantized',
                'description': 'Hybrid quantization - weights quantized, activations float',
                'config': lambda c: (
                    setattr(c, 'optimizations', [tf.lite.Optimize.DEFAULT]),
                    setattr(c, 'representative_dataset', rep_data)
                )[-1]
            }
        ]
        
        successful_conversions = []
        
        print("=== TFLite Conversion with Multiple Strategies ===")
        
        for strategy in conversion_strategies:
            try:
                print(f"\nTrying {strategy['name']}: {strategy['description']}")
                
                converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
                
                # Apply strategy-specific configuration
                if strategy['config']:
                    strategy['config'](converter)
                
                # Android-specific optimizations
                converter.allow_custom_ops = False  # Ensure only built-in ops
                converter.experimental_new_converter = True
                
                # Convert model
                tflite_model = converter.convert()
                
                # Save model
                if strategy['name'] == 'android_optimized':
                    filename = 'model.tflite'  # Main model
                else:
                    filename = f"model_{strategy['name']}.tflite"
                
                path = f'{output_dir}/{filename}'
                with open(path, 'wb') as f:
                    f.write(tflite_model)
                
                size_kb = len(tflite_model) / 1024
                print(f"✓ {strategy['name']} model saved: {size_kb:.1f} KB")
                
                # Test the model
                if self._validate_tflite_model(path, strategy['name']):
                    successful_conversions.append({
                        'path': path,
                        'name': strategy['name'],
                        'size': size_kb,
                        'description': strategy['description']
                    })
                    print(f"✓ {strategy['name']} model validation passed")
                else:
                    print(f"⚠️ {strategy['name']} model validation failed")
                
            except Exception as e:
                print(f"✗ {strategy['name']} conversion failed: {e}")
                continue
        
        if successful_conversions:
            print(f"\n=== Conversion Summary ===")
            for model in successful_conversions:
                print(f"✓ {model['name']}: {model['size']:.1f} KB - {model['description']}")
            
            # Return the best model for Android (prioritize android_optimized)
            best_model = next((m for m in successful_conversions if m['name'] == 'android_optimized'), 
                            successful_conversions[0])
            print(f"\n🚀 Recommended for Android: {best_model['name']} ({best_model['size']:.1f} KB)")
            return best_model['path']
        else:
            raise Exception("All TFLite conversion strategies failed!")
    
    def _validate_tflite_model(self, model_path, strategy_name):
        """Validate TFLite model accuracy against Keras model"""
        try:
            # Load TFLite model
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Test with sample texts
            test_texts = ["free money now", "how are you", "urgent offer"]
            matches = 0
            
            for text in test_texts:
                # Keras prediction
                keras_result = self.predict(text)
                
                # TFLite prediction
                cleaned = self.preprocess_text(text)
                X = pad_sequences(self.tokenizer.texts_to_sequences([cleaned]), maxlen=self.max_length)
                
                # Set input tensor
                interpreter.set_tensor(input_details[0]['index'], X.astype(np.float32))
                interpreter.invoke()
                
                # Get output
                tflite_output = interpreter.get_tensor(output_details[0]['index'])[0]
                tflite_prediction = self.label_encoder.inverse_transform([np.argmax(tflite_output)])[0]
                
                if keras_result['label'] == tflite_prediction:
                    matches += 1
            
            accuracy = matches / len(test_texts)
            print(f"    Validation accuracy: {accuracy*100:.1f}% ({matches}/{len(test_texts)})")
            
            return accuracy >= 0.67  # Accept if at least 2/3 predictions match
            
        except Exception as e:
            print(f"    Validation error: {e}")
            return False

def main():
    # Load and prepare the spam dataset
    df = pd.read_csv('spam_sms.csv', names=['label', 'message'])
    df = df.dropna(subset=['message'])
    
    # Clean the data - keep only valid spam/ham labels
    print("Original dataset info:")
    print(f"Total messages: {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    # Filter out invalid labels - keep only 'ham' and 'spam'
    valid_labels = ['ham', 'spam']
    df = df[df['label'].isin(valid_labels)]
    
    print(f"\nAfter cleaning (keeping only {valid_labels}):")
    print(f"Total messages: {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    print(f"Classes: {sorted(df['label'].unique())}")
    print()
    
    # Create and train the analyzer
    analyzer = SpamDetectionAnalyzer()
    analyzer.train(df, epochs=5)
    analyzer.save_assets()
    analyzer.convert_to_tflite()
    
    # Test with sample messages
    test_texts = [
        "Congratulations! You've won $1000! Call now!",
        "FREE entry to win prizes! Text NOW!",
        "Thanks for the meeting today",
        "URGENT! Your account will be suspended!"
    ]
    
    print("\n=== Spam Detection Results ===")
    for text in test_texts:
        result = analyzer.predict(text)
        print(f"'{text}' -> {result['label']} ({result['confidence']:.3f})")

if __name__ == "__main__":
    main()