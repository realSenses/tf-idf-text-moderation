---
title: Toxic Comment Moderation
emoji: 🛡️
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
---

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

## Usage

The Gradio interface provides an easy way to analyze comments for toxic content. Simply enter a comment and click "Analyze Comment" to see:
- Overall toxicity status
- Detection results for each category
- Confidence scores for all predictions
- Visual probability chart

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