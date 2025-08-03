# Toxic Comment Moderation Model

A multi-label classification model for detecting toxic comments across 6 categories: toxic, severe toxic, obscene, threat, insult, and identity hate.

## Model Description

This model is trained on the Kaggle Toxic Comments dataset to identify various types of toxic content in user-generated text. It uses a combination of:
- Enhanced text preprocessing with pattern recognition
- Word-level TF-IDF features (n-grams 1-3)
- Character-level TF-IDF features (n-grams 3-5)
- Statistical text features
- Separate Logistic Regression classifiers for each toxic category

## Performance Metrics

| Category      | Accuracy | AUC    | Precision | Recall |
|---------------|----------|--------|-----------|--------|
| Toxic         | 0.9477   | 0.9671 | 0.7391    | 0.7689 |
| Severe Toxic  | 0.9891   | 0.9858 | 0.3846    | 0.5618 |
| Obscene       | 0.9753   | 0.9851 | 0.7898    | 0.8513 |
| Threat        | 0.9968   | 0.9889 | 0.2222    | 0.4444 |
| Insult        | 0.9651   | 0.9774 | 0.6571    | 0.7630 |
| Identity Hate | 0.9916   | 0.9823 | 0.4444    | 0.5714 |
| **MEAN**      | 0.9776   | 0.9811 | 0.5395    | 0.6601 |

## Installation

### Using pip
```bash
pip install -r requirements.txt
```

### Using Docker
```bash
docker build -t toxic-comment-moderation .
docker run -p 8000:8000 toxic-comment-moderation
```

## Usage

### 1. Training the Model

First, download the dataset from [Kaggle Toxic Comments](https://www.kaggle.com/datasets/get2jawa/toxic-comments-train) and place `train.csv` in the project directory.

```bash
python train_model.py
```

This will:
- Load and preprocess the data
- Train the classifiers
- Save the model files (*.pkl)

### 2. Using the Gradio Interface

```bash
python app.py
```

Then open http://localhost:7860 in your browser.

### 3. Using the API

Start the FastAPI server:
```bash
python api.py
```

The API will be available at http://localhost:8000

#### API Endpoints:

- `GET /` - API information
- `GET /health` - Health check
- `GET /categories` - List of toxic categories
- `POST /predict` - Analyze a single comment
- `POST /predict/batch` - Analyze multiple comments

#### Example API Usage:

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "Your comment here"}
)
result = response.json()

# Batch prediction
response = requests.post(
    "http://localhost:8000/predict/batch",
    json={"texts": ["Comment 1", "Comment 2"]}
)
results = response.json()
```

### 4. Using the Python Interface

```python
from inference import ToxicCommentClassifier

# Initialize classifier
classifier = ToxicCommentClassifier()

# Single prediction
result = classifier.predict("Your comment here")
print(f"Is toxic: {result['is_toxic']}")
print(f"Categories: {result['toxic_categories']}")
print(f"Probabilities: {result['probabilities']}")

# Batch prediction
results = classifier.predict_batch(["Comment 1", "Comment 2"])
```

## Deployment to Hugging Face Spaces

1. Create a new Space on Hugging Face
2. Upload these files:
   - `app.py`
   - `inference.py`
   - `requirements.txt`
   - All `*.pkl` model files
3. Set the Space SDK to "Gradio"
4. The Space will automatically build and deploy

## File Structure

```
.
├── train_model.py          # Training script
├── inference.py            # Inference class
├── api.py                  # FastAPI application
├── app.py                  # Gradio interface
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker configuration
├── README.md              # This file
└── *.pkl                  # Model files (generated after training)
```

## Model Files

After training, these files will be generated:
- `toxic_classifier_toxic.pkl`
- `toxic_classifier_severe_toxic.pkl`
- `toxic_classifier_obscene.pkl`
- `toxic_classifier_threat.pkl`
- `toxic_classifier_insult.pkl`
- `toxic_classifier_identity_hate.pkl`
- `tfidf_word_vectorizer.pkl`
- `tfidf_char_vectorizer.pkl`
- `feature_scaler.pkl`

## Limitations

- Model is trained on English text only
- May not detect subtle or context-dependent toxicity
- Performance varies by category (threat and identity hate have lower recall)
- Requires careful threshold tuning for production use

## Ethical Considerations

This model is designed for content moderation to create safer online spaces. However:
- It should complement, not replace, human moderation
- False positives may unfairly flag legitimate content
- Regular monitoring and updates are needed to maintain effectiveness
- Consider the impact on free expression when implementing

## License

This project is released under the MIT License.

## Citation

If you use this model, please cite:
```
@software{toxic_comment_moderation,
  title = {Toxic Comment Moderation Model},
  year = {2024},
  publisher = {Hugging Face}
}
```