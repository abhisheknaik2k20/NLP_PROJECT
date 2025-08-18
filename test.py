import tensorflow as tf
import numpy as np
import json
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

def create_tflite_model():
    """Create TFLite model with proper configuration for LSTM"""
    try:
        # Load your Keras model
        print("Loading Keras model...")
        model = tf.keras.models.load_model('model_assets/model.h5')
        
        # Convert the Keras model to TensorFlow Lite with Flex delegate
        print("Converting to TensorFlow Lite...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Enable Flex ops for LSTM compatibility
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter._experimental_lower_tensor_list_ops = False
        converter.allow_custom_ops = True
        
        tflite_model = converter.convert()
        
        # Save the TensorFlow Lite model
        with open('model_flex.tflite', 'wb') as f:
            f.write(tflite_model)
        
        print(f"TensorFlow Lite model saved as 'model_flex.tflite' ({len(tflite_model)/1024:.1f} KB)")
        return True
        
    except Exception as e:
        print(f"TFLite conversion failed: {e}")
        return False

def test_tflite_model():
    """Test the TensorFlow Lite model"""
    try:
        # Load the TensorFlow Lite model
        interpreter = tf.lite.Interpreter(model_path='model_flex.tflite')
        interpreter.allocate_tensors()
        
        # Load tokenizer and labels
        with open('model_assets/tokenizer.json', 'r') as f:
            tokenizer_data = json.load(f)
        
        with open('model_assets/labels.json', 'r') as f:
            labels_data = json.load(f)
        
        word_index = tokenizer_data['word_index']
        max_length = tokenizer_data['max_length']
        classes = labels_data['classes']
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("TensorFlow Lite model loaded successfully!")
        print(f"Input shape: {input_details[0]['shape']}")
        print(f"Output shape: {output_details[0]['shape']}")
        print(f"Classes: {classes}")
        
        def preprocess_text(text):
            """Preprocess text for prediction"""
            text = str(text).lower()
            text = re.sub(r'http\S+|www\S+|https\S+|@\w+|#\w+', '', text)
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        
        def text_to_sequences(text):
            """Convert text to sequences using word_index"""
            words = preprocess_text(text).split()
            sequence = []
            for word in words:
                if word in word_index:
                    sequence.append(word_index[word])
            return [sequence]  # Return as list for pad_sequences
        
        def predict_tflite(text):
            """Make prediction using TFLite model"""
            # Preprocess text
            sequences = text_to_sequences(text)
            padded = pad_sequences(sequences, maxlen=max_length)
            
            # Convert to float32
            input_data = padded.astype(np.float32)
            
            # Set input tensor
            interpreter.set_tensor(input_details[0]['index'], input_data)
            
            # Run inference
            interpreter.invoke()
            
            # Get output
            output_data = interpreter.get_tensor(output_details[0]['index'])
            predictions = output_data[0]
            
            # Get predicted class and confidence
            predicted_class = np.argmax(predictions)
            confidence = float(np.max(predictions))
            predicted_label = classes[predicted_class]
            
            return {
                'text': text,
                'sentiment': predicted_label,
                'confidence': confidence,
                'probabilities': {classes[i]: float(prob) for i, prob in enumerate(predictions)}
            }
        
        # Test with sample texts
        test_texts = [
            "I love this product! It's amazing!",
            "This is terrible, worst experience ever",
            "The game is okay, could be better",
            "Absolutely fantastic! Highly recommend!",
            "Not good at all, very disappointing"
        ]
        
        print("\n=== TensorFlow Lite Model Predictions ===")
        for i, text in enumerate(test_texts, 1):
            result = predict_tflite(text)
            print(f"\n{i}. Text: {text}")
            print(f"   Prediction: {result['sentiment']}")
            print(f"   Confidence: {result['confidence']:.3f}")
            print("   Probabilities:", end=" ")
            for label, prob in result['probabilities'].items():
                print(f"{label}: {prob:.3f}", end=" ")
            print()
        
        # Performance test
        print("\n=== Performance Test ===")
        import time
        perf_text = "This is a performance test text"
        num_runs = 100
        
        start_time = time.time()
        for _ in range(num_runs):
            predict_tflite(perf_text)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs * 1000
        print(f"Average inference time: {avg_time:.2f} ms")
        print(f"Predictions per second: {1000/avg_time:.0f}")
        
        return True
        
    except Exception as e:
        print(f"Error testing TFLite model: {e}")
        return False

def main():
    print("=== TensorFlow Lite Model Test ===\n")
    
    # Check if model files exist
    import os
    if not os.path.exists('model_assets/model.h5'):
        print("Error: model_assets/model.h5 not found!")
        print("Please run the training script first.")
        return
    
    if not os.path.exists('model_flex.tflite'):
        print("TensorFlow Lite model not found. Creating it...")
        if not create_tflite_model():
            print("Failed to create TFLite model. Exiting.")
            return
    else:
        print("Found existing TensorFlow Lite model.")
    
    # Test the TFLite model
    test_tflite_model()

if __name__ == "__main__":
    main()