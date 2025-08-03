import numpy as np
import re
import string
import joblib
from scipy.sparse import hstack
from typing import Dict, List, Tuple


class ToxicCommentClassifier:
    def __init__(self, model_dir: str = '.'):
        self.toxic_categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        self.classifiers = []
        self.tfidf_word = None
        self.tfidf_char = None
        self.scaler = None
        self.model_dir = model_dir
        self.load_models()
    
    def load_models(self):
        print("Loading models...")
        for category in self.toxic_categories:
            clf = joblib.load(f'{self.model_dir}/toxic_classifier_{category}.pkl')
            self.classifiers.append(clf)
        
        self.tfidf_word = joblib.load(f'{self.model_dir}/tfidf_word_vectorizer.pkl')
        self.tfidf_char = joblib.load(f'{self.model_dir}/tfidf_char_vectorizer.pkl')
        self.scaler = joblib.load(f'{self.model_dir}/feature_scaler.pkl')
        print("Models loaded successfully!")
    
    def enhanced_clean_text(self, text: str) -> str:
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
    
    def get_text_features(self, text: str) -> Dict[str, float]:
        features = {}
        features['char_count'] = len(text)
        features['word_count'] = len(text.split())
        features['capital_ratio'] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['punctuation_ratio'] = sum(1 for c in text if c in string.punctuation) / max(len(text), 1)
        return features
    
    def predict(self, text: str) -> Dict[str, any]:
        cleaned = self.enhanced_clean_text(text)
        
        text_features = self.get_text_features(text)
        text_features_array = np.array([[
            text_features['char_count'],
            text_features['word_count'], 
            text_features['capital_ratio'],
            text_features['exclamation_count'],
            text_features['question_count'],
            text_features['punctuation_ratio']
        ]])
        
        text_features_scaled = self.scaler.transform(text_features_array)
        
        text_tfidf_word = self.tfidf_word.transform([cleaned])
        text_tfidf_char = self.tfidf_char.transform([cleaned])
        
        text_combined = hstack([text_tfidf_word, text_tfidf_char, text_features_scaled])
        
        predictions = {}
        probabilities = {}
        
        for i, (clf, category) in enumerate(zip(self.classifiers, self.toxic_categories)):
            pred = clf.predict(text_combined)[0]
            prob = clf.predict_proba(text_combined)[0, 1]
            
            predictions[category] = bool(pred)
            probabilities[category] = float(prob)
        
        is_toxic = any(predictions.values())
        toxic_categories = [cat for cat, is_toxic in predictions.items() if is_toxic]
        
        return {
            'text': text,
            'is_toxic': is_toxic,
            'toxic_categories': toxic_categories,
            'predictions': predictions,
            'probabilities': probabilities
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, any]]:
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results


def main():
    classifier = ToxicCommentClassifier()
    
    test_comments = [
        "This is a great article, thank you for sharing!",
        "I disagree with your opinion but respect your viewpoint.",
        "You're an idiot and don't know what you're talking about!!!",
        "I HATE people like you",
        "This content is inappropriate and offensive."
    ]
    
    print("Testing inference:\n")
    for comment in test_comments:
        result = classifier.predict(comment)
        print(f"Text: {result['text']}")
        print(f"Is Toxic: {result['is_toxic']}")
        if result['is_toxic']:
            print(f"Categories: {', '.join(result['toxic_categories'])}")
        print(f"Probabilities: {result['probabilities']}")
        print("-" * 60)


if __name__ == "__main__":
    main()