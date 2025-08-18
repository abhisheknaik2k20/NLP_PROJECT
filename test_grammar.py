#!/usr/bin/env python3
"""
Simple command-line script to test grammar correction using the ultra-accurate TFLite model
Usage: python test_grammar.py
"""

import numpy as np
import tensorflow as tf
import json
import sys
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import re

class GrammarTester:
    def __init__(self, model_path='grammar_model_assets/model_android_compatible_full.tflite', 
                 tokenizer_path='grammar_model_assets/tokenizer.json',
                 labels_path='grammar_model_assets/labels.json'):
        """Initialize the grammar tester with TFLite model"""
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.labels_path = labels_path
        self.max_length = 120  # Same as training
        
        print("🚀 Loading Ultra-Accurate Grammar Tester...")
        
        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print(f"✅ TFLite model loaded: {model_path}")
        print(f"   Input shape: {self.input_details[0]['shape']}")
        print(f"   Output shape: {self.output_details[0]['shape']}")
        
        # Load tokenizer
        with open(tokenizer_path, 'r') as f:
            tokenizer_data = json.load(f)
        
        # Create tokenizer from the saved data
        self.tokenizer = Tokenizer(num_words=tokenizer_data.get('num_words', 25000), 
                                  oov_token='<OOV>')
        self.tokenizer.word_index = tokenizer_data['word_index']
        
        print(f"✅ Tokenizer loaded: {tokenizer_path}")
        print(f"   Vocabulary size: {len(self.tokenizer.word_index):,}")
        print(f"   Max features: {tokenizer_data.get('num_words', 25000):,}")
        
        # Load labels
        with open(labels_path, 'r') as f:
            labels_data = json.load(f)
        
        # Handle different label formats
        if 'classes' in labels_data:
            self.labels = labels_data['classes']
        else:
            self.labels = labels_data
            
        print(f"✅ Labels loaded: {labels_path}")
        print(f"   Classes: {self.labels}")
        
        print("\n🎯 Grammar Tester Ready!")
        print("=" * 50)
    
    def preprocess_text(self, text):
        """Preprocess text (same as training)"""
        if not text or pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Basic cleaning while preserving sentence structure
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\'\"]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def predict(self, text):
        """Make prediction using TFLite model"""
        # Preprocess text
        cleaned_text = self.preprocess_text(text)
        if not cleaned_text:
            return None
        
        # Tokenize and pad
        sequences = self.tokenizer.texts_to_sequences([cleaned_text])
        padded = pad_sequences(sequences, maxlen=self.max_length, dtype='float32')
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], padded)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        
        # Get prediction
        predicted_class_idx = np.argmax(output)
        confidence = float(np.max(output))
        
        # Map to label
        predicted_label = self.labels[predicted_class_idx]
        
        return {
            'text': text,
            'cleaned_text': cleaned_text,
            'prediction': predicted_label,
            'confidence': confidence,
            'probabilities': {
                self.labels[i]: float(prob) 
                for i, prob in enumerate(output)
            }
        }
    
    def interactive_test(self):
        """Interactive command-line testing"""
        print("\n🎯 Interactive Grammar Testing")
        print("Type sentences to check grammar, or 'quit' to exit")
        print("=" * 50)
        
        while True:
            try:
                # Get user input
                text = input("\n📝 Enter text to check: ").strip()
                
                if not text:
                    continue
                
                if text.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                # Make prediction
                result = self.predict(text)
                
                if result is None:
                    print("❌ Could not process the text")
                    continue
                
                # Display results
                print(f"\n🔍 Analysis Results:")
                print(f"   Original: '{result['text']}'")
                print(f"   Cleaned:  '{result['cleaned_text']}'")
                print(f"   📊 Prediction: {result['prediction'].upper()}")
                print(f"   🎯 Confidence: {result['confidence']:.3f}")
                
                # Show probabilities
                print(f"   📈 Probabilities:")
                for label, prob in result['probabilities'].items():
                    bar_length = int(prob * 20)
                    bar = "█" * bar_length + "░" * (20 - bar_length)
                    print(f"      {label:>12}: {prob:.3f} [{bar}]")
                
                # Interpretation
                if result['confidence'] > 0.8:
                    confidence_level = "High"
                    emoji = "🎯"
                elif result['confidence'] > 0.6:
                    confidence_level = "Medium"
                    emoji = "⚖️"
                else:
                    confidence_level = "Low"
                    emoji = "❓"
                
                print(f"\n{emoji} Confidence Level: {confidence_level}")
                
                if result['prediction'] == 'ungrammatical' and result['confidence'] > 0.7:
                    print("💡 Suggestion: This sentence may contain grammatical errors")
                elif result['prediction'] == 'grammatical' and result['confidence'] > 0.7:
                    print("✅ This sentence appears to be grammatically correct")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")

def main():
    """Main function"""
    print("=" * 60)
    print("🧠 Ultra-Accurate Grammar Correction Tester")
    print("📱 Using Android-Compatible TFLite Model")
    print("=" * 60)
    
    # Check if files exist
    required_files = [
        'grammar_model_assets/model_android_compatible_full.tflite',
        'grammar_model_assets/tokenizer.json',
        'grammar_model_assets/labels.json'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease ensure you have trained the grammar model first by running:")
        print("   python grammar.py")
        return
    
    try:
        # Initialize tester
        tester = GrammarTester()
        
        # Check if command line argument provided
        if len(sys.argv) > 1:
            # Single prediction mode
            text = ' '.join(sys.argv[1:])
            print(f"\n🔍 Testing: '{text}'")
            
            result = tester.predict(text)
            if result:
                print(f"\n📊 Result: {result['prediction'].upper()}")
                print(f"🎯 Confidence: {result['confidence']:.3f}")
                
                if result['prediction'] == 'ungrammatical':
                    print("💡 This sentence may contain grammatical errors")
                else:
                    print("✅ This sentence appears to be grammatically correct")
        else:
            # Interactive mode
            tester.interactive_test()
    
    except Exception as e:
        print(f"❌ Error initializing grammar tester: {str(e)}")
        print("\nPlease ensure the model files are present and valid.")

if __name__ == "__main__":
    # Add pandas import for preprocessing (if not available, use simpler version)
    try:
        import pandas as pd
    except ImportError:
        # Fallback version without pandas
        class pd:
            @staticmethod
            def isna(x):
                return x is None or (isinstance(x, float) and np.isnan(x))
    
    main()
