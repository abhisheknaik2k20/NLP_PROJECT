import tensorflow as tf
import numpy as np
import json
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os


class MainTest:
    def __init__(self,test):
        self.test=test
        self.pass_test()


class MobileTFLitePredictor:
    def __init__(self, model_path, tokenizer_path, labels_path):
        """Initialize mobile-optimized TFLite predictor"""
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.labels_path = labels_path
        
        # Load model assets
        self._load_assets()
        
        # Initialize TFLite interpreter
        self._load_tflite_model()
    
    def _load_assets(self):
        """Load tokenizer and labels"""
        print("Loading tokenizer and labels...")
        
        # Load tokenizer
        with open(self.tokenizer_path, 'r') as f:
            tokenizer_data = json.load(f)
        
        self.word_index = tokenizer_data['word_index']
        self.max_length = tokenizer_data['max_length']
        self.num_words = tokenizer_data.get('num_words', 5000)
        
        # Load labels
        with open(self.labels_path, 'r') as f:
            labels_data = json.load(f)
        
        self.classes = labels_data['classes']
        print(f"Loaded {len(self.classes)} classes: {self.classes}")
        print(f"Max sequence length: {self.max_length}")
        print(f"Vocabulary size: {len(self.word_index)} words")
    
    def _load_tflite_model(self):
        """Load and configure TFLite interpreter"""
        print(f"Loading TFLite model from: {self.model_path}")
        
        try:
            # Initialize interpreter
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            
            # Get input/output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            print("✓ TFLite model loaded successfully!")
            print(f"Input shape: {self.input_details[0]['shape']}")
            print(f"Input dtype: {self.input_details[0]['dtype']}")
            print(f"Output shape: {self.output_details[0]['shape']}")
            print(f"Output dtype: {self.output_details[0]['dtype']}")
            
            # Check for quantization
            input_quantization = self.input_details[0].get('quantization_parameters', {})
            if input_quantization.get('scales') is not None:
                print(f"Input quantization: scales={input_quantization['scales']}, zero_points={input_quantization['zero_points']}")
            else:
                print("Input: No quantization (float32)")
                
            output_quantization = self.output_details[0].get('quantization_parameters', {})
            if output_quantization.get('scales') is not None:
                print(f"Output quantization: scales={output_quantization['scales']}, zero_points={output_quantization['zero_points']}")
            else:
                print("Output: No quantization (float32)")
                
        except Exception as e:
            print(f"✗ Failed to load TFLite model: {e}")
            raise
    
    def preprocess_text(self, text):
        """Preprocess text exactly like the training script"""
        if not text or pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return re.sub(r'\s+', ' ', text).strip()
    
    def text_to_sequences(self, text):
        """Convert text to padded sequences"""
        cleaned_text = self.preprocess_text(text)
        words = cleaned_text.split()
        
        # Convert words to indices
        sequence = []
        for word in words:
            if word in self.word_index and self.word_index[word] < self.num_words:
                sequence.append(self.word_index[word])
            else:
                # Use OOV token index (should be 1)
                sequence.append(1)
        
        # Pad sequences
        padded = pad_sequences([sequence], maxlen=self.max_length)
        return padded.astype(np.float32)
    
    def predict(self, text):
        """Make prediction using TFLite model"""
        try:
            # Preprocess input
            input_data = self.text_to_sequences(text)
            
            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            predictions = output_data[0]  # Remove batch dimension
            
            # Apply softmax to get probabilities
            exp_preds = np.exp(predictions - np.max(predictions))
            probabilities = exp_preds / np.sum(exp_preds)
            
            # Get prediction results
            predicted_class_idx = np.argmax(probabilities)
            confidence = float(probabilities[predicted_class_idx])
            predicted_label = self.classes[predicted_class_idx]
            
            return {
                'text': text,
                'sentiment': predicted_label,
                'confidence': confidence,
                'probabilities': {self.classes[i]: float(prob) for i, prob in enumerate(probabilities)},
                'class_scores': predictions.tolist(),
                'input_shape': input_data.shape
            }
            
        except Exception as e:
            print(f"✗ Prediction failed for '{text}': {e}")
            return {
                'text': text,
                'sentiment': 'ERROR',
                'confidence': 0.0,
                'error': str(e)
            }

def test_mobile_compatibility():
    """Test multiple TFLite models for mobile compatibility"""
    print("=== Mobile TFLite Compatibility Test ===\n")
    
    # Define test models
    test_models = [
        ('model_assets/model_mobile_optimized.tflite', 'Mobile Optimized'),
        ('model_assets/model_basic_float32.tflite', 'Basic Float32'),
        ('model_assets/model_int8_mobile.tflite', 'INT8 Mobile'),
        ('model_assets/model_float16_quantized.tflite', 'Float16 Quantized (may not work on mobile)')
    ]
    
    # Test texts covering all classes
    test_texts = [
        "I love this product! Amazing quality!",          # Should be Positive
        "This is terrible! Worst experience ever!",      # Should be Negative  
        "The product is okay, nothing special",          # Should be Neutral
        "Random technical specifications here",          # Should be Irrelevant
        "Great service and excellent support!",          # Should be Positive
        "Disappointing quality, not worth it",           # Should be Negative
    ]
    
    successful_models = []
    
    for model_path, model_name in test_models:
        print(f"\n{'='*50}")
        print(f"Testing: {model_name}")
        print(f"Path: {model_path}")
        print('='*50)
        
        if not os.path.exists(model_path):
            print(f"✗ Model not found: {model_path}")
            continue
        
        try:
            # Initialize predictor
            predictor = MobileTFLitePredictor(
                model_path=model_path,
                tokenizer_path='model_assets/tokenizer.json',
                labels_path='model_assets/labels.json'
            )
            
            print(f"\n--- Predictions for {model_name} ---")
            predictions = []
            
            for i, text in enumerate(test_texts, 1):
                result = predictor.predict(text)
                predictions.append(result)
                
                if 'error' not in result:
                    print(f"{i}. '{text[:40]}...'")
                    print(f"   → {result['sentiment']} ({result['confidence']:.3f})")
                    print(f"   Input shape: {result['input_shape']}")
                else:
                    print(f"{i}. '{text[:40]}...' → ERROR: {result['error']}")
            
            # Performance test
            print(f"\n--- Performance Test for {model_name} ---")
            import time
            perf_text = "This is a performance test message"
            num_runs = 50
            
            start_time = time.time()
            for _ in range(num_runs):
                predictor.predict(perf_text)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / num_runs * 1000
            print(f"Average inference time: {avg_time:.2f} ms")
            print(f"Predictions per second: {1000/avg_time:.0f}")
            
            # Check class distribution
            class_counts = {}
            for pred in predictions:
                if 'error' not in pred:
                    sentiment = pred['sentiment']
                    class_counts[sentiment] = class_counts.get(sentiment, 0) + 1
            
            print(f"Class distribution: {class_counts}")
            
            # Mark as successful if no errors and all classes represented
            if len([p for p in predictions if 'error' not in p]) == len(test_texts):
                print(f"✓ {model_name} - All predictions successful!")
                if len(set(class_counts.keys())) >= 3:  # At least 3 different classes
                    print(f"✓ {model_name} - Good class diversity!")
                    successful_models.append((model_path, model_name, avg_time))
                else:
                    print(f"⚠️ {model_name} - Limited class diversity: {list(class_counts.keys())}")
            else:
                print(f"✗ {model_name} - Some predictions failed!")
                
        except Exception as e:
            print(f"✗ {model_name} failed to load: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("MOBILE COMPATIBILITY SUMMARY")
    print('='*60)
    
    if successful_models:
        print("✓ Successfully tested models:")
        for model_path, model_name, avg_time in successful_models:
            file_size = os.path.getsize(model_path) / 1024
            print(f"  • {model_name}")
            print(f"    Path: {model_path}")
            print(f"    Size: {file_size:.1f} KB")
            print(f"    Speed: {avg_time:.1f} ms/prediction")
            print()
        
        # Recommend best model
        best_model = min(successful_models, key=lambda x: x[2])  # Fastest model
        print(f"🏆 RECOMMENDED FOR MOBILE: {best_model[1]}")
        print(f"   Fastest inference: {best_model[2]:.1f} ms")
        print(f"   Use: {best_model[0]}")
        
        # Mobile integration tips
        print(f"\n📱 MOBILE INTEGRATION TIPS:")
        print(f"1. Use the recommended model: {best_model[0]}")
        print(f"2. Ensure your mobile app includes:")
        print(f"   - model_assets/tokenizer.json")
        print(f"   - model_assets/labels.json") 
        print(f"   - {best_model[0]}")
        print(f"3. Use float32 input/output (avoid int8 on mobile)")
        print(f"4. Preprocess text exactly as shown in this script")
        print(f"5. Expected classes: {predictor.classes}")
        
    else:
        print("✗ No models passed mobile compatibility tests!")
        print("   Try running: python main.py")
        print("   This will generate mobile-optimized models.")

def create_mobile_integration_guide():
    """Create a guide for mobile integration"""
    guide_content = '''# Mobile TFLite Integration Guide

## Files Needed for Mobile App:
- `model_mobile_optimized.tflite` (recommended)
- `tokenizer.json`
- `labels.json`

## Text Preprocessing (CRITICAL - must match training):
```python
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\\S+|www\\S+|@\\w+|#\\w+', '', text)  # Remove URLs, mentions, hashtags
    text = re.sub(r'[^a-zA-Z\\s]', '', text)                 # Keep only letters and spaces
    text = re.sub(r'\\s+', ' ', text).strip()                # Normalize whitespace
    return text
```

## Expected Input/Output:
- **Input**: Float32 array of shape [1, 50] (padded sequences)
- **Output**: Float32 array of shape [1, 4] (class probabilities)
- **Classes**: ["Irrelevant", "Negative", "Neutral", "Positive"]

## Performance Expectations:
- Inference time: ~20-50ms on mobile devices
- Model size: ~600-800 KB
- Memory usage: <10MB

## Troubleshooting:
1. **Wrong predictions**: Ensure text preprocessing matches exactly
2. **Model won't load**: Use mobile_optimized.tflite (not float16_quantized)
3. **Slow inference**: Avoid int8 quantized models on mobile
4. **Class mismatch**: Check labels.json for correct class order
'''
    
    with open('MOBILE_INTEGRATION_GUIDE.md', 'w') as f:
        f.write(guide_content)
    print("✓ Created MOBILE_INTEGRATION_GUIDE.md")

if __name__ == "__main__":
    # Fix pandas import for preprocessing
    import pandas as pd
    
    # Test mobile compatibility
    test_mobile_compatibility()
    
    # Create integration guide
    create_mobile_integration_guide()
