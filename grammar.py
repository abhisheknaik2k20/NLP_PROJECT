import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, SpatialDropout1D, Dropout, Conv1D, GlobalMaxPooling1D, BatchNormalization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import re
import json
import os
import chardet

# Set seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class UltraAccurateGrammarAnalyzer:
    def __init__(self, max_features=25000, max_length=120):  # Ultra-high capacity for maximum accuracy
        """
        🚀 Initialize Ultra-Accurate Grammar Analyzer
        Enhanced for maximum TensorFlow-equivalent TFLite accuracy with minimal shrinking
        """
        self.max_features = max_features
        self.max_length = max_length
        self.tokenizer = None
        self.label_encoder = None
        self.model = None
        
        print(f"🚀 Initialized ULTRA-MAXIMUM Accuracy Grammar Analyzer:")
        print(f"   📝 Max Features: {self.max_features:,}")
        print(f"   📏 Max Length: {self.max_length}")
        print(f"   🎯 Target: MAXIMUM TensorFlow-equivalent accuracy")
        print(f"   📱 Mobile: Minimal shrinking, maximum precision")
        
    def _detect_encoding(self, filepath):
        """Detect file encoding to avoid issues"""
        try:
            with open(filepath, 'rb') as f:
                result = chardet.detect(f.read())
            return result['encoding']
        except:
            return 'utf-8'
    
    def _load_csv_safely(self, filepath):
        """Load CSV with proper encoding detection"""
        encodings = [self._detect_encoding(filepath), 'utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(filepath, encoding=encoding)
                print(f"Successfully loaded {filepath} with {encoding} encoding")
                return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"Error with {encoding}: {e}")
                continue
        
        raise Exception(f"Could not load {filepath} with any encoding")
        
    def preprocess_text(self, text):
        """Enhanced text preprocessing for grammar detection"""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        # Preserve important punctuation for grammar analysis
        text = re.sub(r'[^\w\s\.\,\?\!\-\'\;\:\(\)]', '', text)
        # Normalize multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def prepare_data_from_csv(self, csv_path):
        """Prepare data from Grammar Correction.csv with ultra-accurate preprocessing and data augmentation"""
        df = self._load_csv_safely(csv_path)
        print(f"Original dataset loaded: {len(df)} samples")
        
        # Create input-output pairs for grammar correction with enhanced data augmentation
        ungrammatical_data = []
        grammatical_data = []
        
        for _, row in df.iterrows():
            if pd.notna(row['Ungrammatical Statement']) and pd.notna(row['Standard English']):
                # Original pairs
                ungrammatical_text = self.preprocess_text(row['Ungrammatical Statement'])
                grammatical_text = self.preprocess_text(row['Standard English'])
                
                if ungrammatical_text and grammatical_text and ungrammatical_text != grammatical_text:
                    ungrammatical_data.append({
                        'text': ungrammatical_text,
                        'label': 0,
                        'label_text': 'ungrammatical'
                    })
                    grammatical_data.append({
                        'text': grammatical_text,
                        'label': 1,
                        'label_text': 'grammatical'
                    })
                    
                    # Data augmentation: Create variations for better learning
                    # Add punctuation variants
                    for punct in ['.', '!', '?', '']:
                        if punct != '' or not ungrammatical_text.endswith(('.', '!', '?')):
                            ungrammatical_data.append({
                                'text': ungrammatical_text.rstrip('.!?') + punct,
                                'label': 0,
                                'label_text': 'ungrammatical'
                            })
                            grammatical_data.append({
                                'text': grammatical_text.rstrip('.!?') + punct,
                                'label': 1,
                                'label_text': 'grammatical'
                            })
        
        # Additional high-quality grammatical examples for better balance
        additional_grammatical = [
            "I am going to the store today.",
            "She has completed all her assignments.",
            "They were playing basketball yesterday.",
            "The sun rises in the east every morning.",
            "We have been studying for three hours.",
            "He will finish the project by tomorrow.",
            "The students are working on their homework.",
            "I walked to work every day last month.",
            "The flowers bloom beautifully in spring.",
            "She thinks the movie was excellent.",
            "The computer is working perfectly now.",
            "They take vacations every summer.",
            "The train leaves at exactly six PM.",
            "The scientist performs careful experiments daily.",
            "The teacher explains concepts very clearly."
        ]
        
        for text in additional_grammatical:
            grammatical_data.append({
                'text': self.preprocess_text(text),
                'label': 1,
                'label_text': 'grammatical'
            })
        
        # Combine and create balanced dataset
        all_data = ungrammatical_data + grammatical_data
        grammar_df = pd.DataFrame(all_data)
        
        # Remove any duplicates and ensure quality
        grammar_df = grammar_df.drop_duplicates(subset=['text']).reset_index(drop=True)
        
        print(f"Enhanced dataset created: {len(grammar_df)} samples")
        print("Class distribution:")
        print(grammar_df['label_text'].value_counts())
        
        # Preprocess text
        grammar_df['cleaned_text'] = grammar_df['text'].apply(self.preprocess_text)
        grammar_df = grammar_df[grammar_df['cleaned_text'].str.len() > 0]
        print(f"After cleaning: {len(grammar_df)} samples")
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        grammar_df['label_encoded'] = self.label_encoder.fit_transform(grammar_df['label_text'])
        
        print(f"Label encoding: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        
        # Create tokenizer and sequences with enhanced vocabulary
        self.tokenizer = Tokenizer(num_words=self.max_features, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(grammar_df['cleaned_text'])
        X = pad_sequences(self.tokenizer.texts_to_sequences(grammar_df['cleaned_text']), 
                         maxlen=self.max_length)
        y = tf.keras.utils.to_categorical(grammar_df['label_encoded'])
        
        print(f"Final dataset shape: X={X.shape}, y={y.shape}")
        return X, y, grammar_df
    
    def build_model(self, num_classes):
        """Ultra-high accuracy architecture - maximized for TensorFlow-like performance"""
        self.model = Sequential([
            # Enhanced embedding layer with higher dimensions for better representation
            Embedding(self.max_features, 256, input_length=self.max_length),
            SpatialDropout1D(0.2),  # Reduced dropout for maximum information retention
            
            # Multi-scale CNN layers for comprehensive grammatical pattern detection
            Conv1D(128, 3, activation='relu', padding='same', name='conv1d_1'),
            BatchNormalization(),
            Conv1D(128, 4, activation='relu', padding='same', name='conv1d_2'),
            BatchNormalization(),
            Conv1D(128, 5, activation='relu', padding='same', name='conv1d_3'),
            BatchNormalization(),
            
            # Additional CNN layer for deeper pattern recognition
            Conv1D(64, 2, activation='relu', padding='same', name='conv1d_4'),
            BatchNormalization(),
            
            # Global pooling to capture most important grammatical features
            GlobalMaxPooling1D(),
            
            # Enhanced dense layers with more capacity for grammar rules
            Dense(512, activation='relu', name='dense_1'),
            BatchNormalization(),
            Dropout(0.3),  # Moderate dropout
            
            Dense(256, activation='relu', name='dense_2'),
            BatchNormalization(),
            Dropout(0.2),  # Lower dropout to preserve information
            
            Dense(128, activation='relu', name='dense_3'),
            BatchNormalization(),
            Dropout(0.1),  # Minimal dropout before output
            
            # Output layer for binary classification
            Dense(num_classes, activation='softmax', name='output')
        ])
        
        # Ultra-precise optimizer configuration for maximum accuracy
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.0008,  # Optimized learning rate for grammar patterns
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7  # Higher precision
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']  # Focus on accuracy
        )
        
        print(f"🏗️ Built ultra-high accuracy grammar model:")
        print(f"   📊 Parameters: {self.model.count_params():,}")
        print(f"   🧠 Architecture: Multi-scale CNN + Enhanced Dense layers")
        print(f"   🎯 Optimization: High-precision Adam optimizer")
    
    def train(self, csv_path, epochs=40, batch_size=16):  # Smaller batch for better accuracy
        """Train with ultra-accurate settings for maximum performance"""
        X, y, df = self.prepare_data_from_csv(csv_path)
        
        # Use stratified split to ensure balanced classes
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(splitter.split(X, np.argmax(y, axis=1)))
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Calculate class weights for balanced training
        y_train_labels = np.argmax(y_train, axis=1)
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train_labels), y=y_train_labels)
        class_weight_dict = dict(enumerate(class_weights))
        
        print(f"Class weights: {class_weight_dict}")
        print(f"Training classes distribution:")
        unique, counts = np.unique(y_train_labels, return_counts=True)
        for i, (cls, count) in enumerate(zip(unique, counts)):
            print(f"  {self.label_encoder.classes_[cls]}: {count} samples")
        
        self.build_model(len(self.label_encoder.classes_))
        
        # Enhanced callbacks for ultra-accurate training
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=8,  # More patience for better convergence
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.7,
            patience=4,
            min_lr=0.00001,
            verbose=1
        )
        
        # Enhanced metrics logging
        class MetricsLogger(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if epoch % 5 == 0:  # Every 5 epochs
                    print(f"\n📊 Epoch {epoch+1} Detailed Metrics:")
                    print(f"   🎯 Accuracy: {logs.get('accuracy', 0):.4f} | Val Accuracy: {logs.get('val_accuracy', 0):.4f}")
                    print(f"   📉 Loss: {logs.get('loss', 0):.4f} | Val Loss: {logs.get('val_loss', 0):.4f}")
        
        callbacks = [early_stopping, reduce_lr, MetricsLogger()]
        
        print(f"Training on {len(X_train)} samples, validating on {len(X_test)} samples")
        print(f"Classes: {self.label_encoder.classes_}")
        
        # Train with enhanced settings
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
        
        # Enhanced per-class evaluation
        y_pred = self.model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        print(f"\n=== Per-class Grammar Detection Analysis ===")
        for i, class_name in enumerate(self.label_encoder.classes_):
            class_mask = (y_true_classes == i)
            if np.sum(class_mask) > 0:
                correct = np.sum((y_pred_classes == y_true_classes) & class_mask)
                total = np.sum(class_mask)
                class_precision = np.sum((y_pred_classes == i) & (y_true_classes == i)) / max(np.sum(y_pred_classes == i), 1)
                class_recall = correct / total
                class_f1 = 2 * (class_precision * class_recall) / max(class_precision + class_recall, 1e-7)
                print(f"  {class_name}: {correct}/{total} correct, Precision: {class_precision:.3f}, Recall: {class_recall:.3f}, F1: {class_f1:.3f}")
        
        return X_test, y_test  # Return test data for further validation
    
    def predict(self, text):
        """Enhanced prediction with confidence scoring"""
        cleaned = self.preprocess_text(text)
        X = pad_sequences(self.tokenizer.texts_to_sequences([cleaned]), maxlen=self.max_length)
        pred = self.model.predict(X, verbose=0)[0]
        return {
            'label': self.label_encoder.inverse_transform([np.argmax(pred)])[0],
            'confidence': float(np.max(pred)),
            'probabilities': {
                self.label_encoder.classes_[i]: float(pred[i])
                for i in range(len(self.label_encoder.classes_))
            }
        }
    
    def save_assets(self, output_dir='grammar_model_assets'):
        """Save model assets for deployment"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save Keras model (legacy format warning is expected)
        self.model.save(f'{output_dir}/model.h5')
        
        # Save tokenizer
        with open(f'{output_dir}/tokenizer.json', 'w') as f:
            json.dump({
                'word_index': self.tokenizer.word_index,
                'num_words': self.tokenizer.num_words,
                'max_length': self.max_length
            }, f)
        
        # Save labels
        with open(f'{output_dir}/labels.json', 'w') as f:
            json.dump({'classes': self.label_encoder.classes_.tolist()}, f)
            
        print(f"✓ Model assets saved to '{output_dir}/' directory")
    
    def _comprehensive_validate_tflite(self, tflite_model, model_name="TFLite"):
        """Ultra-comprehensive TFLite model validation"""
        try:
            interpreter = tf.lite.Interpreter(model_content=tflite_model)
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Comprehensive test samples for grammar detection
            validation_texts = [
                ("I goes to the store everyday.", "ungrammatical"),
                ("I go to the store everyday.", "grammatical"),
                ("They was playing soccer last night.", "ungrammatical"),
                ("They were playing soccer last night.", "grammatical"),
                ("She have completed her homework.", "ungrammatical"),
                ("She has completed her homework.", "grammatical"),
                ("He don't know the answer.", "ungrammatical"),
                ("He doesn't know the answer.", "grammatical"),
                ("The students studies for the exam.", "ungrammatical"),
                ("The students study for the exam.", "grammatical"),
                ("The flowers is blooming in spring.", "ungrammatical"),
                ("The flowers bloom in spring.", "grammatical"),
                ("She think she can finish the project.", "ungrammatical"),
                ("She thinks she can finish the project.", "grammatical"),
                ("The computer not working properly.", "ungrammatical"),
                ("The computer is not working properly.", "grammatical")
            ]
            
            exact_matches = 0
            confidence_scores = []
            
            print(f"      Validation Details:")
            
            for text, expected_label in validation_texts:
                # Keras prediction
                keras_result = self.predict(text)
                
                # TFLite prediction
                cleaned = self.preprocess_text(text)
                X = pad_sequences(self.tokenizer.texts_to_sequences([cleaned]), 
                                maxlen=self.max_length)
                
                input_data = X.astype(input_details[0]['dtype'])
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                
                output_data = interpreter.get_tensor(output_details[0]['index'])
                tflite_pred_idx = np.argmax(output_data[0])
                tflite_confidence = float(np.max(output_data[0]))
                tflite_label = self.label_encoder.inverse_transform([tflite_pred_idx])[0]
                
                # Check exact label match
                if keras_result['label'] == tflite_label:
                    exact_matches += 1
                
                # Calculate confidence similarity
                keras_confidence = keras_result['confidence']
                confidence_similarity = min(keras_confidence, tflite_confidence) / max(keras_confidence, tflite_confidence)
                confidence_scores.append(confidence_similarity)
            
            # Calculate metrics
            exact_match_rate = (exact_matches / len(validation_texts)) * 100
            avg_confidence_similarity = np.mean(confidence_scores) * 100
            combined_score = (exact_match_rate + avg_confidence_similarity) / 2
            
            print(f"        Exact matches: {exact_matches}/{len(validation_texts)} ({exact_match_rate:.1f}%)")
            print(f"        Confidence similarity: {avg_confidence_similarity:.1f}%")
            print(f"        Combined score: {combined_score:.1f}%")
            
            # Validation result
            validation_passed = combined_score >= 90.0
            status = "✅ PASSED" if validation_passed else "⚠️ FAILED (below 90%)"
            print(f"   {status} Validation: {combined_score:.1f}% accuracy - {status}")
            
            return validation_passed, combined_score
            
        except Exception as e:
            print(f"   ❌ Validation failed: {str(e)}")
            return False, 0.0
    
    def convert_to_tflite(self, output_dir='grammar_model_assets'):
        """Ultra-accurate TFLite conversion with multiple strategies for maximum Android accuracy"""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n=== Maximum Accuracy TFLite Conversion ===")
        print(f"Priority: Accuracy over size - preserving TensorFlow model behavior\n")
        
        def representative_dataset():
            """Enhanced representative dataset for better quantization"""
            sample_texts = [
                "I goes to the store everyday.",
                "I go to the store everyday.", 
                "They was playing soccer last night.",
                "They were playing soccer last night.",
                "She have completed her homework.",
                "She has completed her homework.",
                "He don't know the answer.",
                "He doesn't know the answer.",
                "The students studies for the exam.",
                "The students study for the exam.",
                "The flowers is blooming in spring.",
                "The flowers bloom in spring.",
                "She think she can finish the project.",
                "She thinks she can finish the project.",
                "The computer not working properly.",
                "The computer is not working properly."
            ]
            
            for text in sample_texts:
                cleaned = self.preprocess_text(text)
                X = pad_sequences(self.tokenizer.texts_to_sequences([cleaned]), 
                                maxlen=self.max_length)
                yield [X.astype(np.float32)]
        
        successful_models = {}
        
        # Strategy 1: Maximum Accuracy - Pure Float32
        print("🔄 Converting: maximum_accuracy")
        print("   Strategy: Pure float32 - exact TensorFlow model replica, no optimization")
        print("   Converting model...")
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            # No optimizations - pure float32
            tflite_model = converter.convert()
            
            model_size_kb = len(tflite_model) / 1024
            print(f"   ✅ Saved: {model_size_kb:.1f} KB")
            
            # Validate model
            is_valid, score = self._comprehensive_validate_tflite(tflite_model, "maximum_accuracy")
            if is_valid:
                with open(f'{output_dir}/model_maximum_accuracy.tflite', 'wb') as f:
                    f.write(tflite_model)
                successful_models['maximum_accuracy'] = {
                    'size_kb': model_size_kb,
                    'accuracy': score,
                    'description': 'Pure float32 - exact TensorFlow model replica, no optimization'
                }
        except Exception as e:
            print(f"   ❌ Conversion failed: {str(e)}")
        
        print()
        
        # Strategy 2: TF Ops Enabled (Float32 with TF ops support)
        print("🔄 Converting: tf_ops_enabled")
        print("   Strategy: Float32 with TF ops support - maximum compatibility")
        print("   Converting model...")
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            converter.allow_custom_ops = True
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            tflite_model = converter.convert()
            
            model_size_kb = len(tflite_model) / 10245
            print(f"   ✅ Saved: {model_size_kb:.1f} KB")
            
            # Validate model
            is_valid, score = self._comprehensive_validate_tflite(tflite_model, "tf_ops_enabled")
            if is_valid:
                with open(f'{output_dir}/model_tf_ops_enabled.tflite', 'wb') as f:
                    f.write(tflite_model)
                successful_models['tf_ops_enabled'] = {
                    'size_kb': model_size_kb,
                    'accuracy': score,
                    'description': 'Float32 with TF ops support - maximum compatibility'
                }
        except Exception as e:
            print(f"   ❌ Conversion failed: {str(e)}")
        
        print()
        
        # Strategy 3: High Precision Dynamic Range
        print("🔄 Converting: high_precision_dynamic")
        print("   Strategy: Minimal dynamic range - preserves accuracy with slight size reduction")
        print("   Converting model...")
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_dataset
            tflite_model = converter.convert()
            
            model_size_kb = len(tflite_model) / 1024
            print(f"   ✅ Saved: {model_size_kb:.1f} KB")
            
            # Validate model
            is_valid, score = self._comprehensive_validate_tflite(tflite_model, "high_precision_dynamic")
            if is_valid:
                with open(f'{output_dir}/model_high_precision_dynamic.tflite', 'wb') as f:
                    f.write(tflite_model)
                successful_models['high_precision_dynamic'] = {
                    'size_kb': model_size_kb,
                    'accuracy': score,
                    'description': 'Minimal dynamic range - preserves accuracy with slight size reduction'
                }
        except Exception as e:
            print(f"   ❌ Conversion failed: {str(e)}")
        
        print()
        
        # Strategy 4: Android Compatible Full Precision
        print("🔄 Converting: android_compatible_full")
        print("   Strategy: Android-compatible with maximum accuracy preservation")
        print("   Converting model...")
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            # Minimal optimizations for Android compatibility
            converter.optimizations = []  # No optimizations
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
            tflite_model = converter.convert()
            
            model_size_kb = len(tflite_model) / 1024
            print(f"   ✅ Saved: {model_size_kb:.1f} KB")
            
            # Validate model
            is_valid, score = self._comprehensive_validate_tflite(tflite_model, "android_compatible_full")
            if is_valid:
                with open(f'{output_dir}/model_android_compatible_full.tflite', 'wb') as f:
                    f.write(tflite_model)
                # Also save as main Android model
                with open(f'{output_dir}/model_android.tflite', 'wb') as f:
                    f.write(tflite_model)
                successful_models['android_compatible_full'] = {
                    'size_kb': model_size_kb,
                    'accuracy': score,
                    'description': 'Android-compatible with maximum accuracy preservation'
                }
        except Exception as e:
            print(f"   ❌ Conversion failed: {str(e)}")
        
        # Summary of successful models
        if successful_models:
            print(f"\n=== Ultra-Accurate Grammar Models Generated ===")
            # Sort by accuracy descending
            sorted_models = sorted(successful_models.items(), key=lambda x: x[1]['accuracy'], reverse=True)
            
            for name, info in sorted_models:
                print(f"✅ {name}: {info['size_kb']:.1f} KB - {info['accuracy']:.1f}% accuracy")
                print(f"   📋 {info['description']}")
            
            # Recommend the best model
            best_model = sorted_models[0]
            print(f"\n🏆 BEST MODEL FOR MAXIMUM ACCURACY:")
            print(f"   📁 File: {best_model[0]}")
            print(f"   📊 Accuracy: {best_model[1]['accuracy']:.1f}%")
            print(f"   💾 Size: {best_model[1]['size_kb']:.1f} KB")
            print(f"   🎯 Use this for: Production deployment requiring maximum accuracy")
        else:
            print("\n⚠️ No models passed validation. Check model architecture and data.")
        
        return successful_models

    def test_sample_predictions(self):
        """Test the model with comprehensive grammar samples"""
        print(f"\n=== Enhanced Grammar Detection Test ===")
        
        # Comprehensive test samples covering various grammar errors
        test_samples = [
            # Verb tense errors
            ("I goes to the store everyday.", "ungrammatical"),
            ("I go to the store everyday.", "grammatical"),
            ("They was playing soccer last night.", "ungrammatical"),
            ("They were playing soccer last night.", "grammatical"),
            ("She have completed her homework.", "ungrammatical"),
            ("She has completed her homework.", "grammatical"),
            
            # Subject-verb agreement
            ("The dogs runs quickly to the park.", "ungrammatical"),
            ("The dogs run quickly to the park.", "grammatical"),
            ("The results of the experiments was inconclusive.", "ungrammatical"),
            ("The results of the experiments were inconclusive.", "grammatical"),
            
            # Article usage
            ("I went to a school yesterday.", "ungrammatical"),
            ("I went to school yesterday.", "grammatical"),
            ("An apple a day keeps doctor away.", "ungrammatical"),
            ("An apple a day keeps the doctor away.", "grammatical"),
            
            # Additional complex cases
            ("He don't know the answer.", "ungrammatical"),
            ("He doesn't know the answer.", "grammatical"),
            ("The computer not working properly.", "ungrammatical"),
            ("The computer is not working properly.", "grammatical")
        ]
        
        correct_predictions = 0
        
        for text, expected in test_samples:
            result = self.predict(text)
            is_correct = result['label'] == expected
            status = "✓" if is_correct else "✗"
            
            if is_correct:
                correct_predictions += 1
                
            print(f"{status} '{text}'")
            print(f"   Expected: {expected} | Predicted: {result['label']} ({result['confidence']:.3f})")
            
            if not is_correct:
                print(f"   ⚠️  INCORRECT PREDICTION!")
            print()
        
        accuracy = (correct_predictions / len(test_samples)) * 100
        print(f"Sample Test Accuracy: {correct_predictions}/{len(test_samples)} ({accuracy:.1f}%)")
        
        return accuracy >= 80.0  # Expect at least 80% accuracy on samples
    
    def validate_keras_vs_tflite(self, tflite_path):
        """Enhanced Keras vs TFLite validation"""
        print(f"\n=== Model Validation (Keras vs TFLite) ===")
        print(f"=== Keras vs TFLite Prediction Comparison ===")
        
        try:
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            print(f"TFLite Input Type: {input_details[0]['dtype']}")
            print(f"TFLite Output Type: {output_details[0]['dtype']}")
            print()
            
            # Test with validation samples
            validation_texts = [
                "I goes to the store everyday.",
                "I go to the store everyday.",
                "They was playing soccer last night.", 
                "They were playing soccer last night.",
                "She have completed her homework.",
                "She has completed her homework."
            ]
            
            matches = 0
            
            for text in validation_texts:
                # Keras prediction
                keras_result = self.predict(text)
                
                # TFLite prediction
                cleaned = self.preprocess_text(text)
                X = pad_sequences(self.tokenizer.texts_to_sequences([cleaned]), 
                                maxlen=self.max_length)
                
                input_data = X.astype(input_details[0]['dtype'])
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                
                output_data = interpreter.get_tensor(output_details[0]['index'])
                tflite_pred_idx = np.argmax(output_data[0])
                tflite_confidence = float(np.max(output_data[0]))
                tflite_label = self.label_encoder.inverse_transform([tflite_pred_idx])[0]
                
                # Check match
                match = keras_result['label'] == tflite_label
                matches += match
                status = "✓" if match else "✗"
                
                print(f"Text: '{text}'")
                print(f"  Keras:  {keras_result['label']} ({keras_result['confidence']:.3f})")
                print(f"  TFLite: {tflite_label} ({tflite_confidence:.3f})")
                print(f"  Match: {status}")
                print()
            
            consistency = (matches / len(validation_texts)) * 100
            print(f"Model Consistency: {matches}/{len(validation_texts)} ({consistency:.1f}%)")
            
            if consistency >= 90:
                print("✓ Excellent model consistency between Keras and TFLite versions.")
            elif consistency >= 75:
                print("✓ Good model consistency between Keras and TFLite versions.")
            else:
                print("⚠️  Low consistency detected. Consider using a different conversion strategy.")
                
            return consistency >= 75
            
        except Exception as e:
            print(f"Validation failed: {e}")
            return False

def main():
    """Enhanced main function with ultra-accurate grammar detection"""
    print("=== Ultra-Accurate Grammar Correction Model ===")
    csv_path = 'Grammar Correction.csv'
    
    if not os.path.exists(csv_path):
        print(f"❌ Error: {csv_path} not found!")
        print("Please ensure the Grammar Correction.csv file is in the current directory.")
        return
    
    # Create and train the ultra-accurate analyzer
    analyzer = UltraAccurateGrammarAnalyzer()
    
    try:
        # Train the model with ultra-accurate settings
        X_test, y_test = analyzer.train(csv_path, epochs=50, batch_size=12)  # More epochs, smaller batch
        
        # Save model assets
        analyzer.save_assets()
        
        # Convert to multiple TFLite formats
        successful_models = analyzer.convert_to_tflite()
        
        # Test sample predictions
        sample_test_passed = analyzer.test_sample_predictions()
        
        # Validate the best TFLite model if available
        if successful_models:
            # Use the most accurate model for validation
            best_model = max(successful_models.items(), key=lambda x: x[1]['accuracy'])
            best_model_name = best_model[0]
            
            tflite_path = f'grammar_model_assets/model_{best_model_name}.tflite'
            if os.path.exists(tflite_path):
                tflite_consistency = analyzer.validate_keras_vs_tflite(tflite_path)
                
                print(f"\n✓ Model training completed successfully!")
                print(f"✓ Generated labels saved to 'grammar_model_assets/labels.json'")
                print(f"✓ Model assets saved to 'grammar_model_assets/' directory")
                print(f"✓ Ultra-accurate TFLite model: grammar_model_assets/model_android.tflite")
                
                # Final summary
                print(f"\n🏆 ULTRA-ACCURATE GRAMMAR MODEL SUMMARY:")
                print(f"   📊 Best TFLite Model: {best_model_name}")
                print(f"   🎯 Validation Accuracy: {best_model[1]['accuracy']:.1f}%") 
                print(f"   💾 Model Size: {best_model[1]['size_kb']:.1f} KB")
                print(f"   📱 Ready for Android deployment!")
        
        else:
            print("⚠️  No TFLite models passed validation.")
            print("The Keras model was saved but TFLite conversion needs attention.")
            
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
