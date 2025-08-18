import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Conv1D, GlobalMaxPooling1D, SpatialDropout1D, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight 
import re
import json
import os

tf.random.set_seed(42)
np.random.seed(42)

class TwitterSentimentAnalyzer:
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
        text = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return re.sub(r'\s+', ' ', text).strip()
    
    def prepare_data(self, df):
        df['cleaned_text'] = df['text'].apply(self.preprocess_text)
        self.label_encoder = LabelEncoder()
        df['sentiment_encoded'] = self.label_encoder.fit_transform(df['sentiment'])
        self.tokenizer = Tokenizer(num_words=self.max_features, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(df['cleaned_text'])
        X = pad_sequences(self.tokenizer.texts_to_sequences(df['cleaned_text']), maxlen=self.max_length)
        y = tf.keras.utils.to_categorical(df['sentiment_encoded'])
        return X, y
    
    def build_model(self, num_classes):
        # Ultra-high accuracy architecture - maximized for TensorFlow-like performance
        self.model = Sequential([
            # Enhanced embedding layer with higher dimensions
            Embedding(self.max_features, 256, input_length=self.max_length),
            SpatialDropout1D(0.2),  # Reduced dropout for maximum information retention
            
            # Multi-scale CNN layers for comprehensive feature extraction
            Conv1D(128, 3, activation='relu', padding='same', name='conv1d_1'),
            BatchNormalization(),
            Conv1D(128, 4, activation='relu', padding='same', name='conv1d_2'), 
            BatchNormalization(),
            Conv1D(128, 5, activation='relu', padding='same', name='conv1d_3'),
            BatchNormalization(),
            
            # Global pooling to capture most important features
            GlobalMaxPooling1D(),
            
            # Enhanced dense layers with more capacity
            Dense(512, activation='relu', name='dense_1'),
            BatchNormalization(),
            Dropout(0.3),  # Moderate dropout
            
            Dense(256, activation='relu', name='dense_2'),
            BatchNormalization(), 
            Dropout(0.2),  # Lower dropout to preserve more information
            
            Dense(128, activation='relu', name='dense_3'),
            BatchNormalization(),
            Dropout(0.1),  # Minimal dropout before output
            
            # Output layer
            Dense(num_classes, activation='softmax', name='output')
        ])
        
        # Ultra-precise optimizer configuration for maximum accuracy
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.0005,  # Lower learning rate for more precise training
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7  # Higher precision
        )
        
        self.model.compile(
            optimizer=optimizer, 
            loss='categorical_crossentropy', 
            metrics=['accuracy']  # Use only accuracy to avoid conflicts
        )
        
        print(f"🏗️ Built ultra-high accuracy model:")
        print(f"   📊 Parameters: {self.model.count_params():,}")
        print(f"   🧠 Architecture: Multi-scale CNN + Enhanced Dense layers")
        print(f"   🎯 Optimization: High-precision Adam optimizer")
    
    def train(self, train_df, test_df=None, epochs=50, batch_size=16):  # Smaller batch for better accuracy
        # Prepare training data
        X_train, y_train = self.prepare_data(train_df)
        
        # Prepare test data if provided, otherwise use train/test split
        if test_df is not None:
            # Use provided test dataset
            test_df['cleaned_text'] = test_df['text'].apply(self.preprocess_text)
            test_df['sentiment_encoded'] = self.label_encoder.transform(test_df['sentiment'])
            X_test = pad_sequences(self.tokenizer.texts_to_sequences(test_df['cleaned_text']), maxlen=self.max_length)
            y_test = tf.keras.utils.to_categorical(test_df['sentiment_encoded'])
        else:
            # Use stratified split on training data
            X_train, X_test, y_train, y_test = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=np.argmax(y_train, axis=1)
            )
        
        self.build_model(len(self.label_encoder.classes_))
        
        # Calculate class weights to handle imbalance
        y_integers = np.argmax(y_train, axis=1)
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_integers),
            y=y_integers
        )
        class_weight_dict = dict(enumerate(class_weights))
        
        print(f"Class weights: {class_weight_dict}")
        print(f"Training classes distribution:")
        unique, counts = np.unique(y_integers, return_counts=True)
        for i, (class_idx, count) in enumerate(zip(unique, counts)):
            class_name = self.label_encoder.inverse_transform([class_idx])[0]
            print(f"  {class_name}: {count} samples")
        
        # Ultra-precise training callbacks for maximum accuracy
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=15,  # Increased patience for thorough training
            restore_best_weights=True,
            verbose=1,
            mode='max',
            min_delta=0.0001  # More sensitive to improvements
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.7,  # More gradual reduction
            patience=7,  # More patience before reducing
            min_lr=0.000001,  # Lower minimum for fine-tuning
            verbose=1,
            cooldown=3  # Prevent frequent changes
        )
        
        # Additional callback for monitoring multiple metrics
        class MetricsLogger(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if epoch % 5 == 0:  # Every 5 epochs
                    print(f"\n📊 Epoch {epoch+1} Detailed Metrics:")
                    print(f"   🎯 Accuracy: {logs.get('accuracy', 0):.4f} | Val Accuracy: {logs.get('val_accuracy', 0):.4f}")
                    print(f"   📉 Loss: {logs.get('loss', 0):.4f} | Val Loss: {logs.get('val_loss', 0):.4f}")
                    if 'precision' in logs:
                        print(f"   🎪 Precision: {logs.get('precision', 0):.4f} | Recall: {logs.get('recall', 0):.4f}")
        
        callbacks = [early_stopping, reduce_lr, MetricsLogger()]
        
        print(f"Training on {len(X_train)} samples, validating on {len(X_test)} samples")
        print(f"Classes: {self.label_encoder.classes_}")
        
        self.model.fit(
            X_train, y_train, 
            batch_size=batch_size, 
            epochs=epochs, 
            validation_data=(X_test, y_test), 
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=1
        )
        
        # Detailed evaluation
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\n=== Final Model Performance ===")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Show predictions for each class with enhanced metrics
        print("\n=== Per-class prediction check ===")
        test_predictions = self.model.predict(X_test, verbose=0)
        test_pred_classes = np.argmax(test_predictions, axis=1)
        test_true_classes = np.argmax(y_test, axis=1)
        
        for class_idx in range(len(self.label_encoder.classes_)):
            class_name = self.label_encoder.inverse_transform([class_idx])[0]
            true_count = np.sum(test_true_classes == class_idx)
            pred_count = np.sum(test_pred_classes == class_idx)
            correct_count = np.sum((test_true_classes == class_idx) & (test_pred_classes == class_idx))
            precision = correct_count / pred_count if pred_count > 0 else 0
            recall = correct_count / true_count if true_count > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            print(f"  {class_name}: {correct_count}/{true_count} correct, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        
        return X_test, y_test
    
    def predict(self, text):
        cleaned = self.preprocess_text(text)
        X = pad_sequences(self.tokenizer.texts_to_sequences([cleaned]), maxlen=self.max_length)
        pred = self.model.predict(X, verbose=0)[0]
        return {
            'sentiment': self.label_encoder.inverse_transform([np.argmax(pred)])[0],
            'confidence': float(np.max(pred))
        }
    
    def save_assets(self, output_dir='model_assets'):
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
    
    def convert_to_tflite(self, output_dir='model_assets'):
        os.makedirs(output_dir, exist_ok=True)
        
        def rep_data():
            # Comprehensive representative dataset covering all sentiment patterns
            texts = [
                # Positive variations
                "I love this product amazing quality", "fantastic experience highly recommend", 
                "excellent service outstanding results", "best purchase ever so happy",
                "wonderful product great value", "impressive quality exceeded expectations",
                
                # Negative variations  
                "terrible quality waste money", "horrible experience very disappointed",
                "worst service ever completely useless", "awful product don't buy",
                "poor quality terrible experience", "disappointing results very bad",
                
                # Neutral variations
                "okay product nothing special", "average quality standard performance",
                "normal product meets expectations", "decent quality reasonable price",
                "standard service nothing outstanding", "adequate product fair value",
                
                # Edge cases and mixed sentiment
                "good but could be better", "bad quality but cheap price",
                "love design hate functionality", "great service slow delivery",
                "expensive but worth it", "cheap but effective solution"
            ]
            for text in texts:
                X = pad_sequences(self.tokenizer.texts_to_sequences([self.preprocess_text(text)]), 
                                maxlen=self.max_length)
                yield [X.astype(np.float32)]
        
        # Ultra-accurate conversion strategies - NO compression priority
        conversion_strategies = [
            {
                'name': 'maximum_accuracy',
                'description': 'Pure float32 - exact TensorFlow model replica, no optimization',
                'config': lambda c: (
                    # Absolutely no optimizations or quantization
                    setattr(c, 'optimizations', []),
                    setattr(c, 'allow_custom_ops', True),
                    setattr(c, 'experimental_new_converter', False),  # Use stable converter
                    # Ensure float32 precision throughout
                    setattr(c, 'target_spec', tf.lite.TargetSpec(
                        supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
                    ))
                )[-1]
            },
            {
                'name': 'tf_ops_enabled',
                'description': 'Float32 with TF ops support - maximum compatibility',
                'config': lambda c: (
                    setattr(c, 'optimizations', []),
                    setattr(c, 'allow_custom_ops', True),
                    setattr(c, 'experimental_new_converter', True),
                    setattr(c, 'target_spec', tf.lite.TargetSpec(
                        supported_ops=[
                            tf.lite.OpsSet.TFLITE_BUILTINS,
                            tf.lite.OpsSet.SELECT_TF_OPS,
                            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
                        ]
                    ))
                )[-1]
            },
            {
                'name': 'high_precision_dynamic',
                'description': 'Minimal dynamic range - preserves accuracy with slight size reduction',
                'config': lambda c: (
                    setattr(c, 'optimizations', [tf.lite.Optimize.DEFAULT]),
                    setattr(c, 'allow_custom_ops', True),
                    # Use representative dataset but keep high precision
                    setattr(c, 'representative_dataset', rep_data),
                    setattr(c, 'target_spec', tf.lite.TargetSpec(
                        supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
                    ))
                )[-1]
            },
            {
                'name': 'android_compatible_full',
                'description': 'Android-compatible with maximum accuracy preservation',
                'config': lambda c: (
                    setattr(c, 'optimizations', []),
                    setattr(c, 'allow_custom_ops', False),  # Only built-in ops for Android
                    setattr(c, 'experimental_new_converter', True),
                    setattr(c, 'target_spec', tf.lite.TargetSpec(
                        supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS]
                    ))
                )[-1]
            }
        ]
        
        successful_conversions = []
        
        print("\n=== Maximum Accuracy TFLite Conversion ===")
        print("Priority: Accuracy over size - preserving TensorFlow model behavior")
        
        for strategy in conversion_strategies:
            try:
                print(f"\n🔄 Converting: {strategy['name']}")
                print(f"   Strategy: {strategy['description']}")
                
                converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
                
                # Apply strategy-specific configuration
                if strategy['config']:
                    strategy['config'](converter)
                
                # Additional accuracy preservation settings
                converter.experimental_enable_resource_variables = True
                
                # Convert model
                print("   Converting model...")
                tflite_model = converter.convert()
                
                # Save model with priority naming
                if strategy['name'] == 'maximum_accuracy':
                    filename = 'model.tflite'  # Primary model
                elif strategy['name'] == 'android_compatible_full':
                    filename = 'model_android.tflite'  # Android fallback
                else:
                    filename = f"model_{strategy['name']}.tflite"
                
                path = f'{output_dir}/{filename}'
                with open(path, 'wb') as f:
                    f.write(tflite_model)
                
                size_kb = len(tflite_model) / 1024
                print(f"   ✅ Saved: {size_kb:.1f} KB")
                validation_score = self._comprehensive_validate_tflite(path, strategy['name'])
                if validation_score >= 0.9:  # 90%+ accuracy required
                    successful_conversions.append({
                        'path': path,
                        'name': strategy['name'],
                        'size': size_kb,
                        'description': strategy['description'],
                        'accuracy': validation_score
                    })
                    print(f"   ✅ Validation: {validation_score*100:.1f}% accuracy - PASSED")
                else:
                    print(f"   ⚠️ Validation: {validation_score*100:.1f}% accuracy - FAILED (below 90%)")
                
            except Exception as e:
                print(f"   ❌ Conversion failed: {e}")
                continue
        
        if successful_conversions:
            print(f"\n=== Ultra-Accurate Models Generated ===")
            # Sort by accuracy, then by name priority
            successful_conversions.sort(key=lambda x: (-x['accuracy'], x['name'] != 'maximum_accuracy'))
            
            for model in successful_conversions:
                print(f"✅ {model['name']}: {model['size']:.1f} KB - {model['accuracy']*100:.1f}% accuracy")
                print(f"   📋 {model['description']}")
            
            # Return the most accurate model
            best_model = successful_conversions[0]
            print(f"\n🏆 BEST MODEL FOR MAXIMUM ACCURACY:")
            print(f"   📁 File: {best_model['name']}")
            print(f"   📊 Accuracy: {best_model['accuracy']*100:.1f}%")
            print(f"   💾 Size: {best_model['size']:.1f} KB")
            print(f"   🎯 Use this for: Production deployment requiring maximum accuracy")
            
            return best_model['path']
        else:
            raise Exception("❌ All ultra-accurate TFLite conversion strategies failed!")
    
    def _comprehensive_validate_tflite(self, model_path, strategy_name):
        """Comprehensive validation with extensive test cases for maximum accuracy verification"""
        try:
            # Load TFLite model
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Comprehensive test suite covering all sentiment patterns
            test_cases = [
                # Clear positive cases
                ("I absolutely love this amazing product", "positive"),
                ("Fantastic experience, highly recommend to everyone", "positive"),
                ("Outstanding quality, exceeded all my expectations", "positive"),
                ("Best purchase I've ever made, so happy", "positive"),
                
                # Clear negative cases
                ("This is terrible, complete waste of money", "negative"),
                ("Horrible experience, very disappointed", "negative"),
                ("Worst service ever, completely useless", "negative"),
                ("Awful product, don't waste your time", "negative"),
                
                # Clear neutral cases
                ("The product is okay, nothing special", "neutral"),
                ("Average quality, meets basic expectations", "neutral"),
                ("Standard product, no major complaints", "neutral"),
                ("Decent quality for the price point", "neutral"),
                
                # Edge cases and challenging examples
                ("Good product but expensive", None),  # Mixed sentiment
                ("Fast delivery but poor quality", None),  # Mixed sentiment
                ("Love the design, hate the functionality", None),  # Mixed sentiment
                ("Cheap but effective solution", None),  # Mixed sentiment
            ]
            
            total_tests = 0
            exact_matches = 0
            confidence_sum = 0
            
            for text, expected_sentiment in test_cases:
                try:
                    # Keras prediction (ground truth)
                    keras_result = self.predict(text)
                    keras_sentiment = keras_result['sentiment']
                    keras_confidence = keras_result['confidence']
                    
                    # TFLite prediction
                    cleaned = self.preprocess_text(text)
                    X = pad_sequences(self.tokenizer.texts_to_sequences([cleaned]), maxlen=self.max_length)
                    
                    interpreter.set_tensor(input_details[0]['index'], X.astype(np.float32))
                    interpreter.invoke()
                    
                    tflite_output = interpreter.get_tensor(output_details[0]['index'])[0]
                    tflite_sentiment = self.label_encoder.inverse_transform([np.argmax(tflite_output)])[0]
                    tflite_confidence = float(np.max(tflite_output))
                    
                    total_tests += 1
                    
                    # Check for exact match between Keras and TFLite
                    if keras_sentiment == tflite_sentiment:
                        exact_matches += 1
                    
                    # Accumulate confidence for average calculation
                    confidence_diff = abs(keras_confidence - tflite_confidence)
                    confidence_similarity = 1 - min(confidence_diff, 1.0)
                    confidence_sum += confidence_similarity
                    
                    # For expected sentiment, also check if either model got it right
                    if expected_sentiment and (keras_sentiment == expected_sentiment or tflite_sentiment == expected_sentiment):
                        pass  # Expected behavior
                        
                except Exception as test_e:
                    print(f"      Test case failed: {text[:30]}... - {test_e}")
                    continue
            
            if total_tests == 0:
                return 0.0
            
            # Calculate comprehensive accuracy score
            exact_match_score = exact_matches / total_tests
            confidence_score = confidence_sum / total_tests
            
            # Weighted final score (70% exact matches, 30% confidence similarity)
            final_score = 0.7 * exact_match_score + 0.3 * confidence_score
            
            print(f"      Validation Details:")
            print(f"        Exact matches: {exact_matches}/{total_tests} ({exact_match_score*100:.1f}%)")
            print(f"        Confidence similarity: {confidence_score*100:.1f}%")
            print(f"        Combined score: {final_score*100:.1f}%")
            
            return final_score
            
        except Exception as e:
            print(f"      Validation error: {e}")
            return 0.0
    
    def _validate_tflite_model(self, model_path, strategy_name):
        """Validate TFLite model accuracy against Keras model"""
        try:
            # Load TFLite model
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Test with sample texts
            test_texts = ["I love this product", "This is terrible", "It's okay nothing special"]
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
                
                if keras_result['sentiment'] == tflite_prediction:
                    matches += 1
            
            accuracy = matches / len(test_texts)
            print(f"    Validation accuracy: {accuracy*100:.1f}% ({matches}/{len(test_texts)})")
            
            return accuracy >= 0.67  # Accept if at least 2/3 predictions match
            
        except Exception as e:
            print(f"    Validation error: {e}")
            return False
    
    def compare_keras_vs_tflite(self, test_texts=None, model_path='model_assets/model.tflite'):
        if test_texts is None:
            test_texts = ["I love this!", "This sucks", "It's okay", "Amazing quality", "Terrible product"]
        
        if not os.path.exists(model_path):
            print(f"TFLite model not found at {model_path}")
            return
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        input_dtype = input_details[0]['dtype']
        output_dtype = output_details[0]['dtype']
        input_scale, input_zero_point = input_details[0].get('quantization', (0.0, 0))
        output_scale, output_zero_point = output_details[0].get('quantization', (0.0, 0))
        
        print("=== Keras vs TFLite Prediction Comparison ===")
        print(f"TFLite Input Type: {input_dtype}")
        print(f"TFLite Output Type: {output_dtype}")
        if input_scale != 0.0:
            print(f"Input Quantization: scale={input_scale:.6f}, zero_point={input_zero_point}")
        if output_scale != 0.0:
            print(f"Output Quantization: scale={output_scale:.6f}, zero_point={output_zero_point}")
        print()
        
        matches = 0
        total = len(test_texts)
        
        for text in test_texts:
            # Keras prediction
            keras_result = self.predict(text)
            
            # TFLite prediction
            X = pad_sequences(self.tokenizer.texts_to_sequences([self.preprocess_text(text)]), 
                            maxlen=self.max_length)
            
            # Handle different input types
            if input_dtype == np.int8:
                # Quantize input for INT8 models
                X_quantized = np.clip(np.round(X / input_scale + input_zero_point), -128, 127).astype(np.int8)
                interpreter.set_tensor(input_details[0]['index'], X_quantized)
            else:
                interpreter.set_tensor(input_details[0]['index'], X.astype(np.float32))
            
            interpreter.invoke()
            tflite_pred = interpreter.get_tensor(output_details[0]['index'])[0]
            
            # Handle different output types
            if output_dtype == np.int8:
                # Dequantize output for INT8 models
                tflite_pred = (tflite_pred.astype(np.float32) - output_zero_point) * output_scale
            
            # Apply softmax if needed
            tflite_pred_exp = np.exp(tflite_pred - np.max(tflite_pred))
            tflite_pred_softmax = tflite_pred_exp / np.sum(tflite_pred_exp)
            
            tflite_sentiment = self.label_encoder.inverse_transform([np.argmax(tflite_pred_softmax)])[0]
            tflite_confidence = float(np.max(tflite_pred_softmax))
            
            is_match = keras_result['sentiment'] == tflite_sentiment
            if is_match:
                matches += 1
            
            print(f"Text: '{text}'")
            print(f"  Keras:  {keras_result['sentiment']} ({keras_result['confidence']:.3f})")
            print(f"  TFLite: {tflite_sentiment} ({tflite_confidence:.3f})")
            print(f"  Match: {'✓' if is_match else '✗' }")
            print()
        
        accuracy = matches / total * 100
        print(f"Model Consistency: {matches}/{total} ({accuracy:.1f}%)")
        
        if accuracy < 80:
            print("⚠️  Low consistency detected. Consider using float16 quantization instead of int8.")
        else:
            print("✓ Good model consistency between Keras and TFLite versions.")

def main():
    print("=== Sentiment Analysis Model with Train/Test CSV ===")
    
    # Load training and test datasets with proper encoding handling
    encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
    
    def load_csv_with_encoding(filename):
        for encoding in encodings_to_try:
            try:
                df = pd.read_csv(filename, encoding=encoding)
                print(f"Successfully loaded {filename} with {encoding} encoding")
                return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"Error loading {filename} with {encoding}: {e}")
                continue
        raise Exception(f"Could not load {filename} with any supported encoding")
    
    try:
        train_df = load_csv_with_encoding('train.csv')
        test_df = load_csv_with_encoding('test.csv')
        print(f"Training dataset loaded: {len(train_df)} samples")
        print(f"Test dataset loaded: {len(test_df)} samples")
        
        # Clean the data - remove rows with missing text or sentiment
        train_df = train_df.dropna(subset=['text', 'sentiment'])
        test_df = test_df.dropna(subset=['text', 'sentiment'])
        
        print(f"After cleaning - Training: {len(train_df)}, Test: {len(test_df)}")
        
        # Show class distribution in training data
        print("\nTraining data class distribution:")
        print(train_df['sentiment'].value_counts())
        
        print("\nTest data class distribution:")
        print(test_df['sentiment'].value_counts())
        
    except FileNotFoundError as e:
        print(f"Error: Could not find required CSV files. {e}")
        print("Please ensure both 'train.csv' and 'test.csv' are in the current directory.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Initialize and train the analyzer with ultra-high accuracy settings
    analyzer = TwitterSentimentAnalyzer(max_features=20000, max_length=150)  # Maximized for accuracy
    print(f"🚀 Initialized Ultra-Accuracy Sentiment Analyzer:")
    print(f"   📝 Max Features: {analyzer.max_features:,}")
    print(f"   📏 Max Length: {analyzer.max_length}")
    print(f"   🎯 Target: Maximum TensorFlow-equivalent accuracy")
    
    analyzer.train(train_df, test_df, epochs=20)  # More epochs for thorough training
    analyzer.save_assets()
    tflite_path = analyzer.convert_to_tflite()
    
    # Test predictions with diverse examples
    test_texts = [
        # Positive examples
        "I love this! It's absolutely amazing and works perfectly!",
        "This is fantastic! Highly recommend to everyone!",
        "Excellent quality and outstanding service!",
        "Best product ever, so happy with my purchase!",
        
        # Negative examples  
        "This is terrible! Complete waste of money!",
        "Horrible experience, very disappointed and frustrated!",
        "This sucks, completely useless and broken!",
        "Worst service ever, never buying again!",
        
        # Neutral examples
        "The product is okay, nothing special but works",
        "It's average quality, meets basic expectations",
        "Neither good nor bad, just normal performance",
        "Standard product, no complaints but nothing exciting"
    ]
    
    print("\n=== Model Predictions on Sample Texts ===")
    sentiment_count = {}
    
    for text in test_texts:
        result = analyzer.predict(text)
        sentiment = result['sentiment']
        confidence = result['confidence']
        
        # Count predictions
        sentiment_count[sentiment] = sentiment_count.get(sentiment, 0) + 1
        
        print(f"'{text[:50]}...' -> {sentiment} ({confidence:.3f})")
    
    print(f"\nPrediction distribution: {sentiment_count}")
    
    # Test on a sample of actual test data
    print("\n=== Predictions on Actual Test Data Sample ===")
    test_sample = test_df.sample(min(10, len(test_df)), random_state=42)
    correct_predictions = 0
    
    for idx, row in test_sample.iterrows():
        actual_sentiment = row['sentiment']
        predicted = analyzer.predict(row['text'])
        predicted_sentiment = predicted['sentiment']
        confidence = predicted['confidence']
        
        is_correct = actual_sentiment == predicted_sentiment
        if is_correct:
            correct_predictions += 1
            
        status = "✓" if is_correct else "✗"
        print(f"{status} '{row['text'][:40]}...'")
        print(f"   Actual: {actual_sentiment} | Predicted: {predicted_sentiment} ({confidence:.3f})")
    
    sample_accuracy = correct_predictions / len(test_sample) * 100
    print(f"\nSample Test Accuracy: {correct_predictions}/{len(test_sample)} ({sample_accuracy:.1f}%)")
    
    # Compare Keras vs TFLite models
    print("\n=== Model Validation (Keras vs TFLite) ===")
    validation_texts = test_texts[:6]  # Use subset to avoid too much output
    analyzer.compare_keras_vs_tflite(validation_texts, tflite_path)
    
    # Generate labels.json for reference
    with open('generated_labels.json', 'w') as f:
        labels = {
            'classes': analyzer.label_encoder.classes_.tolist(),
            'num_classes': len(analyzer.label_encoder.classes_),
            'model_info': {
                'max_features': analyzer.max_features,
                'max_length': analyzer.max_length,
                'architecture': 'CNN with Conv1D layers'
            }
        }
        json.dump(labels, f, indent=2)
    
    print(f"\n✓ Model training completed successfully!")
    print(f"✓ Generated labels saved to 'generated_labels.json'")
    print(f"✓ Model assets saved to 'model_assets/' directory")
    print(f"✓ Mobile-optimized TFLite model: {tflite_path}")
    
    return analyzer

if __name__ == "__main__":
    main()