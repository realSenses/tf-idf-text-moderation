# Toxic Comment Moderation Model Documentation

## Overview

This notebook implements a multi-label toxic comment classification system using machine learning techniques. The model is trained on the Kaggle Toxic Comments dataset to detect six types of toxicity in text: toxic, severe_toxic, obscene, threat, insult, and identity_hate.

## Dataset

- **Source**: [Kaggle Toxic Comments Dataset](https://www.kaggle.com/datasets/get2jawa/toxic-comments-train)
- **File Required**: `train.csv` (must be downloaded and placed in the same directory as the notebook)
- **Size**: ~160K comments with multi-label annotations

## Model Architecture

### Feature Engineering

The model uses a comprehensive feature extraction approach:

1. **Enhanced Text Preprocessing**
   - Contraction expansion (e.g., "won't" → "will not")
   - Pattern preservation for punctuation (!!! → EXCLAMATION)
   - Repeated character detection (sooooo → so REPEAT)
   - All-caps word detection (HELLO → hello ALLCAPS)
   - URL and email standardization

2. **TF-IDF Features**
   - Word-level TF-IDF (20,000 features, 1-3 ngrams)
   - Character-level TF-IDF (10,000 features, 3-5 char ngrams)
   - Helps catch misspellings and leetspeak

3. **Statistical Features**
   - Character count
   - Word count
   - Capital letter ratio
   - Exclamation/question mark counts
   - Punctuation ratio

### Model Training

- **Algorithm**: Logistic Regression with class weight balancing
- **Architecture**: Six independent binary classifiers (one per toxicity type)
- **Optimization**: SAGA solver for large-scale datasets
- **Parallelization**: Multi-core training support

## Performance Metrics

### Overall Results
- **Mean Accuracy**: ~94%
- **Mean AUC**: ~0.96

### Per-Category Performance
| Category | Accuracy | AUC | Precision | Recall |
|----------|----------|-----|-----------|--------|
| toxic | 0.95+ | 0.97+ | High | High |
| severe_toxic | 0.99+ | 0.98+ | Medium | Medium |
| obscene | 0.97+ | 0.98+ | High | High |
| threat | 0.99+ | 0.98+ | Low | Low |
| insult | 0.96+ | 0.97+ | High | Medium |
| identity_hate | 0.99+ | 0.97+ | Low | Low |

*Note: Threat and identity_hate have lower precision/recall due to class imbalance*

## Usage

### Prerequisites
```python
import pandas as pd
import numpy as np
import sklearn
import joblib
```

### Loading Saved Models
```python
# Load classifiers
classifiers = []
toxic_categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
for category in toxic_categories:
    clf = joblib.load(f'toxic_classifier_{category}.pkl')
    classifiers.append(clf)

# Load vectorizers and scaler
tfidf_word = joblib.load('tfidf_word_vectorizer.pkl')
tfidf_char = joblib.load('tfidf_char_vectorizer.pkl')
scaler = joblib.load('feature_scaler.pkl')
```

### Making Predictions
The notebook includes a `predict_toxicity_simple()` function that:
1. Cleans the input text
2. Extracts all features
3. Makes predictions for each toxicity category
4. Returns confidence scores

### Example Output
```
Text: This content is inappropriate and offensive.

Predictions:
  ⚠️ toxic: TOXIC (confidence: 0.876)
  ✓ severe_toxic: OK (confidence: 0.923)
  ⚠️ obscene: TOXIC (confidence: 0.712)
  ✓ threat: OK (confidence: 0.991)
  ⚠️ insult: TOXIC (confidence: 0.654)
  ✓ identity_hate: OK (confidence: 0.978)

⚠️ Comment flagged as: toxic, obscene, insult
```

## Model Files

After training, the following files are saved:
- `toxic_classifier_[category].pkl` (6 files, one per toxicity type)
- `tfidf_word_vectorizer.pkl` (word-level TF-IDF vectorizer)
- `tfidf_char_vectorizer.pkl` (character-level TF-IDF vectorizer)
- `feature_scaler.pkl` (statistical feature scaler)

Total model size: ~50-100MB

## Deployment Considerations

- **Inference Time**: ~5-10ms per comment
- **Memory Requirements**: ~500MB RAM for loaded models
- **Scalability**: Can process thousands of comments per second
- **API Integration**: Models can be wrapped in REST API for production use

## Limitations and Future Improvements

### Current Limitations
1. English-only support
2. May miss context-dependent toxicity
3. Vulnerable to adversarial examples (e.g., l33tspeak variations)
4. Class imbalance affects rare categories (threat, identity_hate)

### Suggested Improvements
1. **Deep Learning**: Implement BERT/RoBERTa for better context understanding
2. **Data Augmentation**: Generate synthetic examples for minority classes
3. **Active Learning**: Continuously improve with user feedback
4. **Explainability**: Add LIME/SHAP for prediction explanations
5. **Multi-lingual Support**: Extend to other languages
6. **Ensemble Methods**: Combine with other models for better performance

## Ethical Considerations

- **Bias**: Model may reflect biases present in training data
- **False Positives**: May flag legitimate criticism as toxic
- **Context**: Cannot understand sarcasm or cultural context perfectly
- **Human Review**: Should be used as a tool to assist, not replace, human moderation

## Dependencies

```
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
scipy
```

## Citation

If using this model, please cite the original Kaggle dataset:
```
Toxic Comment Classification Challenge
https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
```