# Flutter Spam Detection Integration Guide

## 🚀 Minimal Setup Integration

### Step 1: Add Dependencies

Add these to your `pubspec.yaml`:

```yaml
dependencies:
  flutter:
    sdk: flutter
  tflite_flutter: ^0.10.4
  tflite_flutter_helper: ^0.3.1

dev_dependencies:
  flutter_test:
    sdk: flutter

flutter:
  assets:
    - assets/models/
```

### Step 2: Copy Model Files

1. Create folder: `assets/models/` in your Flutter project root
2. Copy these files from `spam_model_assets/`:
   - `model.tflite` → `assets/models/spam_model.tflite`
   - `labels.json` → `assets/models/spam_labels.json`
   - `tokenizer.json` → `assets/models/spam_tokenizer.json`

### Step 3: Create the Spam Detector Service

Create `lib/services/spam_detector.dart`:

```dart
import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

class SpamDetector {
  static const String _modelPath = 'assets/models/spam_model.tflite';
  static const String _labelsPath = 'assets/models/spam_labels.json';
  static const String _tokenizerPath = 'assets/models/spam_tokenizer.json';
  
  Interpreter? _interpreter;
  List<String>? _labels;
  Map<String, dynamic>? _tokenizer;
  int _maxLength = 50; // Default, will be updated from tokenizer
  int _numWords = 5000; // Default, will be updated from tokenizer
  
  // Initialize the model
  Future<void> initialize() async {
    try {
      // Load TFLite model
      _interpreter = await Interpreter.fromAsset(_modelPath);
      print('✓ Spam model loaded successfully');
      
      // Load labels
      final labelsData = await rootBundle.loadString(_labelsPath);
      final labelsJson = json.decode(labelsData);
      _labels = List<String>.from(labelsJson['classes']);
      print('✓ Labels loaded: $_labels');
      
      // Load tokenizer
      final tokenizerData = await rootBundle.loadString(_tokenizerPath);
      _tokenizer = json.decode(tokenizerData);
      _maxLength = _tokenizer!['max_length'] ?? 50;
      _numWords = _tokenizer!['num_words'] ?? 5000;
      print('✓ Tokenizer loaded (max_length: $_maxLength, num_words: $_numWords)');
      
    } catch (e) {
      print('✗ Error initializing spam detector: $e');
      rethrow;
    }
  }
  
  // Preprocess text (similar to Python preprocessing)
  String _preprocessText(String text) {
    if (text.isEmpty) return "";
    
    // Convert to lowercase
    text = text.toLowerCase();
    
    // Remove URLs, mentions, hashtags
    text = text.replaceAll(RegExp(r'http\S+|www\S+|@\w+|#\w+'), '');
    
    // Remove non-alphabetic characters except spaces
    text = text.replaceAll(RegExp(r'[^a-zA-Z\s]'), '');
    
    // Remove extra whitespace
    text = text.replaceAll(RegExp(r'\s+'), ' ').trim();
    
    return text;
  }
  
  // Convert text to sequence (tokenization)
  List<int> _textToSequence(String text) {
    final words = text.split(' ');
    final wordIndex = Map<String, int>.from(_tokenizer!['word_index']);
    
    List<int> sequence = [];
    for (String word in words) {
      if (wordIndex.containsKey(word)) {
        int index = wordIndex[word]!;
        if (index < _numWords) {
          sequence.add(index);
        } else {
          sequence.add(1); // OOV token
        }
      } else {
        sequence.add(1); // OOV token
      }
    }
    return sequence;
  }
  
  // Pad sequence to fixed length
  List<int> _padSequence(List<int> sequence, int maxLength) {
    if (sequence.length >= maxLength) {
      return sequence.take(maxLength).toList();
    } else {
      return sequence + List.filled(maxLength - sequence.length, 0);
    }
  }
  
  // Main prediction method
  Future<SpamResult> detectSpam(String message) async {
    if (_interpreter == null || _labels == null || _tokenizer == null) {
      throw Exception('Model not initialized. Call initialize() first.');
    }
    
    try {
      // Preprocess text
      String cleanedText = _preprocessText(message);
      
      // Convert to sequence
      List<int> sequence = _textToSequence(cleanedText);
      
      // Pad sequence
      List<int> paddedSequence = _padSequence(sequence, _maxLength);
      
      // Convert to Float32List for model input
      var input = Float32List.fromList(
        paddedSequence.map((e) => e.toDouble()).toList()
      );
      
      // Reshape input: [1, max_length]
      var inputTensor = input.reshape([1, _maxLength]);
      
      // Prepare output tensor
      var outputTensor = List.filled(1 * _labels!.length, 0.0)
          .reshape([1, _labels!.length]);
      
      // Run inference
      _interpreter!.run(inputTensor, outputTensor);
      
      // Get predictions
      List<double> predictions = outputTensor[0].cast<double>();
      
      // Find max prediction
      double maxScore = predictions.reduce((a, b) => a > b ? a : b);
      int maxIndex = predictions.indexOf(maxScore);
      
      // Apply softmax to get proper probabilities
      List<double> exp = predictions.map((x) => math.exp(x - maxScore)).toList();
      double sumExp = exp.reduce((a, b) => a + b);
      List<double> softmax = exp.map((x) => x / sumExp).toList();
      
      return SpamResult(
        label: _labels![maxIndex],
        confidence: softmax[maxIndex],
        isSpam: _labels![maxIndex].toLowerCase() == 'spam',
        allScores: Map.fromIterables(_labels!, softmax),
        originalText: message,
        cleanedText: cleanedText,
      );
      
    } catch (e) {
      print('✗ Error during spam detection: $e');
      throw Exception('Spam detection failed: $e');
    }
  }
  
  // Cleanup
  void dispose() {
    _interpreter?.close();
    print('✓ Spam detector disposed');
  }
}

// Result class
class SpamResult {
  final String label;
  final double confidence;
  final bool isSpam;
  final Map<String, double> allScores;
  final String originalText;
  final String cleanedText;
  
  SpamResult({
    required this.label,
    required this.confidence,
    required this.isSpam,
    required this.allScores,
    required this.originalText,
    required this.cleanedText,
  });
  
  @override
  String toString() {
    return 'SpamResult(label: $label, confidence: ${confidence.toStringAsFixed(3)}, isSpam: $isSpam)';
  }
  
  // Convert to JSON for easy serialization
  Map<String, dynamic> toJson() {
    return {
      'label': label,
      'confidence': confidence,
      'isSpam': isSpam,
      'allScores': allScores,
      'originalText': originalText,
      'cleanedText': cleanedText,
    };
  }
}
```

### Step 4: Create a Simple UI Widget

Create `lib/widgets/spam_checker_widget.dart`:

```dart
import 'package:flutter/material.dart';
import '../services/spam_detector.dart';
import 'dart:math' as math;

class SpamCheckerWidget extends StatefulWidget {
  @override
  _SpamCheckerWidgetState createState() => _SpamCheckerWidgetState();
}

class _SpamCheckerWidgetState extends State<SpamCheckerWidget> {
  final SpamDetector _spamDetector = SpamDetector();
  final TextEditingController _textController = TextEditingController();
  
  bool _isInitialized = false;
  bool _isLoading = false;
  SpamResult? _lastResult;
  String? _error;
  
  @override
  void initState() {
    super.initState();
    _initializeModel();
  }
  
  Future<void> _initializeModel() async {
    setState(() {
      _isLoading = true;
      _error = null;
    });
    
    try {
      await _spamDetector.initialize();
      setState(() {
        _isInitialized = true;
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _error = e.toString();
        _isLoading = false;
      });
    }
  }
  
  Future<void> _checkSpam() async {
    if (!_isInitialized || _textController.text.trim().isEmpty) return;
    
    setState(() {
      _isLoading = true;
      _error = null;
    });
    
    try {
      final result = await _spamDetector.detectSpam(_textController.text);
      setState(() {
        _lastResult = result;
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _error = e.toString();
        _isLoading = false;
      });
    }
  }
  
  @override
  void dispose() {
    _spamDetector.dispose();
    _textController.dispose();
    super.dispose();
  }
  
  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: EdgeInsets.all(16.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Text(
            'Spam Message Detector',
            style: Theme.of(context).textTheme.headlineSmall,
            textAlign: TextAlign.center,
          ),
          SizedBox(height: 20),
          
          // Input field
          TextField(
            controller: _textController,
            maxLines: 4,
            decoration: InputDecoration(
              hintText: 'Enter message to check for spam...',
              border: OutlineInputBorder(),
              labelText: 'Message',
            ),
          ),
          SizedBox(height: 16),
          
          // Check button
          ElevatedButton(
            onPressed: _isInitialized && !_isLoading ? _checkSpam : null,
            child: _isLoading 
                ? Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      SizedBox(
                        width: 16,
                        height: 16,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      ),
                      SizedBox(width: 8),
                      Text('Analyzing...'),
                    ],
                  )
                : Text('Check for Spam'),
          ),
          
          SizedBox(height: 20),
          
          // Results
          if (_error != null)
            Card(
              color: Colors.red.shade50,
              child: Padding(
                padding: EdgeInsets.all(12),
                child: Text(
                  'Error: $_error',
                  style: TextStyle(color: Colors.red.shade700),
                ),
              ),
            ),
          
          if (_lastResult != null) ...[
            Card(
              color: _lastResult!.isSpam ? Colors.red.shade50 : Colors.green.shade50,
              child: Padding(
                padding: EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Icon(
                          _lastResult!.isSpam ? Icons.warning : Icons.check_circle,
                          color: _lastResult!.isSpam ? Colors.red : Colors.green,
                        ),
                        SizedBox(width: 8),
                        Text(
                          _lastResult!.isSpam ? 'SPAM DETECTED' : 'NOT SPAM',
                          style: TextStyle(
                            fontWeight: FontWeight.bold,
                            color: _lastResult!.isSpam ? Colors.red.shade700 : Colors.green.shade700,
                          ),
                        ),
                      ],
                    ),
                    SizedBox(height: 8),
                    Text('Confidence: ${(_lastResult!.confidence * 100).toStringAsFixed(1)}%'),
                    Text('Classification: ${_lastResult!.label}'),
                    
                    if (_lastResult!.allScores.length > 2) ...[
                      SizedBox(height: 12),
                      Text('All Scores:', style: TextStyle(fontWeight: FontWeight.bold)),
                      ..._lastResult!.allScores.entries.map((entry) => 
                        Text('${entry.key}: ${(entry.value * 100).toStringAsFixed(1)}%')
                      ),
                    ],
                  ],
                ),
              ),
            ),
          ],
          
          // Quick test buttons
          SizedBox(height: 16),
          Text('Quick Tests:', style: TextStyle(fontWeight: FontWeight.bold)),
          SizedBox(height: 8),
          Wrap(
            spacing: 8,
            children: [
              _buildTestButton('Thanks for the meeting today'),
              _buildTestButton('FREE entry! Win $1000 NOW!'),
              _buildTestButton('Your account will be suspended!'),
              _buildTestButton('How are you doing?'),
            ],
          ),
        ],
      ),
    );
  }
  
  Widget _buildTestButton(String text) {
    return ElevatedButton(
      onPressed: _isInitialized && !_isLoading ? () {
        _textController.text = text;
        _checkSpam();
      } : null,
      style: ElevatedButton.styleFrom(
        backgroundColor: Colors.grey.shade200,
        foregroundColor: Colors.black87,
      ),
      child: Text(text, style: TextStyle(fontSize: 12)),
    );
  }
}
```

### Step 5: Use in Your Main App

Update your `lib/main.dart`:

```dart
import 'package:flutter/material.dart';
import 'widgets/spam_checker_widget.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Spam Detector',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: Scaffold(
        appBar: AppBar(
          title: Text('AI Spam Detector'),
        ),
        body: SpamCheckerWidget(),
      ),
    );
  }
}
```

## 🎯 That's It!

### What This Setup Provides:

1. **✅ Automatic Model Loading**: Loads TFLite model on app start
2. **✅ Text Preprocessing**: Same preprocessing as Python model
3. **✅ Real-time Predictions**: Instant spam detection
4. **✅ User-Friendly UI**: Simple input field and results display
5. **✅ Quick Tests**: Pre-defined test messages
6. **✅ Error Handling**: Graceful error management
7. **✅ Memory Management**: Proper cleanup when done

### Usage:

1. Type or paste any message
2. Tap "Check for Spam"
3. Get instant results with confidence scores
4. Use quick test buttons for demo

The model will automatically classify messages as **ham** (not spam) or **spam** with confidence percentages!

## 📱 Build and Test

Run your Flutter app:
```bash
flutter run
```

The spam detector will initialize automatically and be ready to use! 🚀
