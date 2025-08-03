import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from scipy.sparse import hstack
import joblib
import warnings
warnings.filterwarnings('ignore')


def enhanced_clean_text(text):
    text = text.lower()
    
    contractions = {
        "won't": "will not", "can't": "cannot", "n't": " not",
        "i'm": "i am", "it's": "it is", "he's": "he is", "she's": "she is",
        "you're": "you are", "we're": "we are", "they're": "they are",
        "i've": "i have", "you've": "you have", "we've": "we have",
        "i'd": "i would", "you'd": "you would", "he'd": "he would",
        "i'll": "i will", "you'll": "you will", "he'll": "he will"
    }
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    
    text = re.sub(r'!+', ' EXCLAMATION ', text)
    text = re.sub(r'\?+', ' QUESTION ', text)
    text = re.sub(r'\.{3,}', ' ELLIPSIS ', text)
    text = re.sub(r'(\w)\1{2,}', r'\1 REPEAT', text)
    
    words = text.split()
    new_words = []
    for word in words:
        if len(word) > 2 and word.isupper() and word.isalpha():
            new_words.append(word.lower() + ' ALLCAPS')
        else:
            new_words.append(word)
    text = ' '.join(new_words)
    
    text = re.sub(r'http\S+|www\S+', ' URL ', text)
    text = re.sub(r'\S+@\S+', ' EMAIL ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = ' '.join(text.split())
    
    return text


def get_text_features(text):
    features = {}
    features['char_count'] = len(text)
    features['word_count'] = len(text.split())
    features['capital_ratio'] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    features['punctuation_ratio'] = sum(1 for c in text if c in string.punctuation) / max(len(text), 1)
    return features


def train_toxic_comment_model(data_path='train.csv'):
    print("Loading dataset...")
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")
    
    toxic_categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    print("\nPreprocessing text...")
    df['enhanced_text'] = df['comment_text'].apply(enhanced_clean_text)
    
    text_features = df['comment_text'].apply(get_text_features).apply(pd.Series)
    df = pd.concat([df, text_features], axis=1)
    
    X_text = df['enhanced_text']
    X_stats = df[['char_count', 'word_count', 'capital_ratio', 'exclamation_count', 'question_count', 'punctuation_ratio']]
    y = df[toxic_categories]
    
    print("\nSplitting data...")
    X_text_train, X_text_test, X_stats_train, X_stats_test, y_train, y_test = train_test_split(
        X_text, X_stats, y, test_size=0.2, random_state=42, stratify=y['toxic']
    )
    
    scaler = StandardScaler()
    X_stats_train_scaled = scaler.fit_transform(X_stats_train)
    X_stats_test_scaled = scaler.transform(X_stats_test)
    
    print("\nExtracting features...")
    tfidf_word = TfidfVectorizer(
        max_features=20000,
        stop_words='english',
        ngram_range=(1, 3),
        min_df=3,
        max_df=0.9,
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True
    )
    
    tfidf_char = TfidfVectorizer(
        max_features=10000,
        analyzer='char',
        ngram_range=(3, 5),
        min_df=3,
        max_df=0.9,
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True
    )
    
    X_train_tfidf_word = tfidf_word.fit_transform(X_text_train)
    X_test_tfidf_word = tfidf_word.transform(X_text_test)
    
    X_train_tfidf_char = tfidf_char.fit_transform(X_text_train)
    X_test_tfidf_char = tfidf_char.transform(X_text_test)
    
    X_train_combined = hstack([X_train_tfidf_word, X_train_tfidf_char, X_stats_train_scaled])
    X_test_combined = hstack([X_test_tfidf_word, X_test_tfidf_char, X_stats_test_scaled])
    
    print(f"\nCombined feature shape: {X_train_combined.shape}")
    
    class_weights = {}
    for category in toxic_categories:
        classes = np.unique(y_train[category])
        weights = compute_class_weight('balanced', classes=classes, y=y_train[category])
        class_weights[category] = dict(zip(classes, weights))
    
    print("\nTraining classifiers...")
    trained_classifiers = []
    for category in toxic_categories:
        print(f"Training {category}...")
        clf = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight=class_weights[category],
            solver='saga',
            n_jobs=-1
        )
        clf.fit(X_train_combined, y_train[category])
        trained_classifiers.append(clf)
    
    print("\nEvaluating model...")
    y_pred = np.zeros((X_test_combined.shape[0], len(toxic_categories)))
    y_pred_proba = np.zeros((X_test_combined.shape[0], len(toxic_categories)))
    
    for i, clf in enumerate(trained_classifiers):
        y_pred[:, i] = clf.predict(X_test_combined)
        y_pred_proba[:, i] = clf.predict_proba(X_test_combined)[:, 1]
    
    print("\nPerformance metrics:")
    print("-" * 60)
    print(f"{'Category':<15} {'Accuracy':<10} {'AUC':<10} {'Precision':<10} {'Recall':<10}")
    print("-" * 60)
    
    category_metrics = []
    for i, category in enumerate(toxic_categories):
        accuracy = accuracy_score(y_test[category], y_pred[:, i])
        auc = roc_auc_score(y_test[category], y_pred_proba[:, i])
        precision = precision_score(y_test[category], y_pred[:, i], zero_division=0)
        recall = recall_score(y_test[category], y_pred[:, i], zero_division=0)
        
        category_metrics.append({
            'category': category,
            'accuracy': accuracy,
            'auc': auc,
            'precision': precision,
            'recall': recall
        })
        
        print(f"{category:<15} {accuracy:<10.4f} {auc:<10.4f} {precision:<10.4f} {recall:<10.4f}")
    
    mean_accuracy = np.mean([m['accuracy'] for m in category_metrics])
    mean_auc = np.mean([m['auc'] for m in category_metrics])
    mean_precision = np.mean([m['precision'] for m in category_metrics])
    mean_recall = np.mean([m['recall'] for m in category_metrics])
    
    print("-" * 60)
    print(f"{'MEAN':<15} {mean_accuracy:<10.4f} {mean_auc:<10.4f} {mean_precision:<10.4f} {mean_recall:<10.4f}")
    print("-" * 60)
    
    print("\nSaving models...")
    for i, (clf, category) in enumerate(zip(trained_classifiers, toxic_categories)):
        joblib.dump(clf, f'toxic_classifier_{category}.pkl')
        print(f"Saved classifier for {category}")
    
    joblib.dump(tfidf_word, 'tfidf_word_vectorizer.pkl')
    joblib.dump(tfidf_char, 'tfidf_char_vectorizer.pkl')
    joblib.dump(scaler, 'feature_scaler.pkl')
    
    print("\nAll models saved successfully!")
    
    return trained_classifiers, tfidf_word, tfidf_char, scaler


if __name__ == "__main__":
    train_toxic_comment_model()