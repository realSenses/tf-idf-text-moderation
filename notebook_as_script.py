# Cell 3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.multioutput import MultiOutputClassifier
import warnings
warnings.filterwarnings('ignore')

# Cell 5
# Load the dataset
# Make sure to download 'train.csv' from the Kaggle link and place it in the current directory
df = pd.read_csv('train.csv')

# Display basic information
print(f"Dataset shape: {df.shape}")
print("\nColumn names:")
print(df.columns.tolist())
print("\nFirst few rows:")
df.head()

# Cell 7
# Check for missing values
print("Missing values:")
print(df.isnull().sum())

# Define toxic categories
toxic_categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Distribution of toxic categories
print("\nDistribution of toxic categories:")
for category in toxic_categories:
    print(f"{category}: {df[category].sum()} ({df[category].mean()*100:.2f}%)")

# Cell 8
# Visualize distribution of toxic categories
plt.figure(figsize=(10, 6))
category_counts = [df[cat].sum() for cat in toxic_categories]
plt.bar(toxic_categories, category_counts)
plt.xlabel('Toxic Category')
plt.ylabel('Number of Comments')
plt.title('Distribution of Toxic Comment Categories')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Cell 9
# Check correlation between different toxic categories
plt.figure(figsize=(8, 6))
correlation_matrix = df[toxic_categories].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation between Toxic Categories')
plt.tight_layout()
plt.show()

# Cell 11
# Enhanced text preprocessing
import re
import string

def enhanced_clean_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Expand contractions
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
    
    # Preserve important punctuation patterns
    text = re.sub(r'!+', ' EXCLAMATION ', text)
    text = re.sub(r'\?+', ' QUESTION ', text)
    text = re.sub(r'\.{3,}', ' ELLIPSIS ', text)
    
    # Preserve repeated characters (e.g., "sooooo" -> "so REPEAT")
    text = re.sub(r'(\w)\1{2,}', r'\1 REPEAT', text)
    
    # Handle all caps words
    words = text.split()
    new_words = []
    for word in words:
        if len(word) > 2 and word.isupper() and word.isalpha():
            new_words.append(word.lower() + ' ALLCAPS')
        else:
            new_words.append(word)
    text = ' '.join(new_words)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', ' URL ', text)
    
    # Remove emails
    text = re.sub(r'\S+@\S+', ' EMAIL ', text)
    
    # Remove non-alphanumeric characters except preserved patterns
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

# Calculate text statistics features
def get_text_features(text):
    features = {}
    
    # Original text features
    features['char_count'] = len(text)
    features['word_count'] = len(text.split())
    features['capital_ratio'] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    features['punctuation_ratio'] = sum(1 for c in text if c in string.punctuation) / max(len(text), 1)
    
    return features

# Apply enhanced cleaning to comment text
df['enhanced_text'] = df['comment_text'].apply(enhanced_clean_text)

# Extract text features
text_features = df['comment_text'].apply(get_text_features).apply(pd.Series)
df = pd.concat([df, text_features], axis=1)

# Display example of enhanced cleaning
print("Original text:")
print(df['comment_text'].iloc[5])
print("\nEnhanced cleaned text:")
print(df['enhanced_text'].iloc[5])
print("\nText features:")
print(text_features.iloc[5])

# Cell 12
# Prepare features and labels
X_text = df['enhanced_text']
X_stats = df[['char_count', 'word_count', 'capital_ratio', 'exclamation_count', 'question_count', 'punctuation_ratio']]
y = df[toxic_categories]

# Split the data
from sklearn.preprocessing import StandardScaler

X_text_train, X_text_test, X_stats_train, X_stats_test, y_train, y_test = train_test_split(
    X_text, X_stats, y, test_size=0.2, random_state=42, stratify=y['toxic']
)

# Scale the statistical features
scaler = StandardScaler()
X_stats_train_scaled = scaler.fit_transform(X_stats_train)
X_stats_test_scaled = scaler.transform(X_stats_test)

print(f"Training set size: {X_text_train.shape[0]}")
print(f"Test set size: {X_text_test.shape[0]}")
print(f"Statistical features shape: {X_stats_train.shape}")

# Cell 14
# Enhanced feature extraction with TF-IDF and character n-grams
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

# Word-level TF-IDF with more features
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

# Character-level TF-IDF for catching misspellings/leetspeak
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

# Fit and transform training data
print("Extracting word-level features...")
X_train_tfidf_word = tfidf_word.fit_transform(X_text_train)
X_test_tfidf_word = tfidf_word.transform(X_text_test)

print("Extracting character-level features...")
X_train_tfidf_char = tfidf_char.fit_transform(X_text_train)
X_test_tfidf_char = tfidf_char.transform(X_text_test)

# Combine all features: word TF-IDF + char TF-IDF + statistical features
X_train_combined = hstack([
    X_train_tfidf_word,
    X_train_tfidf_char,
    X_stats_train_scaled
])

X_test_combined = hstack([
    X_test_tfidf_word,
    X_test_tfidf_char,
    X_stats_test_scaled
])

print(f"\nCombined feature shape: {X_train_combined.shape}")
print(f"- Word TF-IDF features: {X_train_tfidf_word.shape[1]}")
print(f"- Char TF-IDF features: {X_train_tfidf_char.shape[1]}")
print(f"- Statistical features: {X_stats_train_scaled.shape[1]}")
print(f"Total features: {X_train_combined.shape[1]}")

# Cell 16
# Train improved classifiers with combined features
from sklearn.utils.class_weight import compute_class_weight

# Calculate class weights for each category
class_weights = {}
for category in toxic_categories:
    classes = np.unique(y_train[category])
    weights = compute_class_weight('balanced', classes=classes, y=y_train[category])
    class_weights[category] = dict(zip(classes, weights))
    print(f"{category} weights: {class_weights[category]}")

# Create individual classifiers with class weights
classifiers = []
for category in toxic_categories:
    clf = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight=class_weights[category],
        solver='saga',  # Better for large datasets
        n_jobs=-1  # Use all CPU cores
    )
    classifiers.append((category, clf))

# Train each classifier individually
print("\nTraining improved model with combined features...")
trained_classifiers = []
for category, clf in classifiers:
    print(f"Training {category}...")
    clf.fit(X_train_combined, y_train[category])
    trained_classifiers.append(clf)

print("Training completed!")

# Cell 18
# Make predictions with improved model
y_pred = np.zeros((X_test_combined.shape[0], len(toxic_categories)))
y_pred_proba = np.zeros((X_test_combined.shape[0], len(toxic_categories)))

for i, clf in enumerate(trained_classifiers):
    y_pred[:, i] = clf.predict(X_test_combined)
    y_pred_proba[:, i] = clf.predict_proba(X_test_combined)[:, 1]

# Convert to DataFrame for easier analysis
y_pred_df = pd.DataFrame(y_pred.astype(int), columns=toxic_categories)

# Calculate overall accuracy
overall_accuracy = accuracy_score(y_test.values.flatten(), y_pred.flatten())
print(f"Overall Model Accuracy: {overall_accuracy:.4f}")
print()

# Calculate accuracy and AUC for each category
from sklearn.metrics import roc_auc_score
print("Performance metrics for each toxic category:")
print("-" * 60)
print(f"{'Category':<15} {'Accuracy':<10} {'AUC':<10} {'Precision':<10} {'Recall':<10}")
print("-" * 60)

category_metrics = []
for i, category in enumerate(toxic_categories):
    accuracy = accuracy_score(y_test[category], y_pred[:, i])
    auc = roc_auc_score(y_test[category], y_pred_proba[:, i])
    
    # Calculate precision and recall
    from sklearn.metrics import precision_score, recall_score
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

# Calculate mean metrics
mean_accuracy = np.mean([m['accuracy'] for m in category_metrics])
mean_auc = np.mean([m['auc'] for m in category_metrics])
mean_precision = np.mean([m['precision'] for m in category_metrics])
mean_recall = np.mean([m['recall'] for m in category_metrics])

print("-" * 60)
print(f"{'MEAN':<15} {mean_accuracy:<10.4f} {mean_auc:<10.4f} {mean_precision:<10.4f} {mean_recall:<10.4f}")
print("-" * 60)

# Cell 19
# Confusion matrix for 'toxic' category
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test['toxic'], y_pred[:, 0])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Toxic Category')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Cell 21
# Simple prediction function
def predict_toxicity_simple(text, classifiers, tfidf_word, tfidf_char, scaler):
    # Enhanced cleaning
    cleaned = enhanced_clean_text(text)
    
    # Extract text features
    text_features = get_text_features(text)
    text_features_array = np.array([[
        text_features['char_count'],
        text_features['word_count'], 
        text_features['capital_ratio'],
        text_features['exclamation_count'],
        text_features['question_count'],
        text_features['punctuation_ratio']
    ]])
    
    # Scale features
    text_features_scaled = scaler.transform(text_features_array)
    
    # Transform to TF-IDF
    text_tfidf_word = tfidf_word.transform([cleaned])
    text_tfidf_char = tfidf_char.transform([cleaned])
    
    # Combine features
    from scipy.sparse import hstack
    text_combined = hstack([text_tfidf_word, text_tfidf_char, text_features_scaled])
    
    # Make predictions
    predictions = []
    probabilities = []
    
    for i, clf in enumerate(classifiers):
        # Get prediction and probability
        pred = clf.predict(text_combined)[0]
        prob = clf.predict_proba(text_combined)[0, 1]
        
        predictions.append(pred)
        probabilities.append(prob)
    
    # Display results
    print(f"Text: {text}")
    print("\nPredictions:")
    for i, category in enumerate(toxic_categories):
        if predictions[i]:
            print(f"  ⚠️ {category}: TOXIC (confidence: {probabilities[i]:.3f})")
        else:
            print(f"  ✓ {category}: OK (confidence: {1-probabilities[i]:.3f})")
    
    # Overall assessment
    if any(predictions):
        toxic_cats = [toxic_categories[i] for i, pred in enumerate(predictions) if pred]
        print(f"\n⚠️ Comment flagged as: {', '.join(toxic_cats)}")
    else:
        print("\n✓ Comment appears to be non-toxic")

# Test with example comments
test_comments = [
    "This is a great article, thank you for sharing!",
    "I disagree with your opinion but respect your viewpoint.",
    "This content is inappropriate and offensive."
]

print("Testing the model:\n")
for comment in test_comments:
    predict_toxicity_simple(comment, trained_classifiers, tfidf_word, tfidf_char, scaler)
    print("-" * 60)

# Cell 23
import joblib

# Save the models and vectorizers
print("Saving models...")

# Save individual classifiers
for i, (clf, category) in enumerate(zip(trained_classifiers, toxic_categories)):
    joblib.dump(clf, f'toxic_classifier_{category}.pkl')
    print(f"Saved classifier for {category}")

# Save vectorizers and scaler
joblib.dump(tfidf_word, 'tfidf_word_vectorizer.pkl')
joblib.dump(tfidf_char, 'tfidf_char_vectorizer.pkl')
joblib.dump(scaler, 'feature_scaler.pkl')

print("\nAll models and vectorizers saved successfully!")
print("\nFiles created:")
print("- toxic_classifier_[category].pkl (6 files)")
print("- tfidf_word_vectorizer.pkl")
print("- tfidf_char_vectorizer.pkl")
print("- feature_scaler.pkl")

# Cell 25
# Test with more diverse examples
diverse_comments = [
    "This is a great article, thank you for sharing!",
    "I disagree with your opinion but respect your viewpoint.",
    "You're an idiot and don't know what you're talking about!!!",
    "I HATE people like you, you should just disappear",
    "This is the WORST thing I've ever seen... complete garbage!!!",
    "Kill yourself",
    "I hope something bad happens to you",
    "Your analysis is flawed but interesting nonetheless"
]

print("Testing with diverse examples:\n")
for comment in diverse_comments:
    predict_toxicity_simple(comment, trained_classifiers, tfidf_word, tfidf_char, scaler)
    print("-" * 60)
