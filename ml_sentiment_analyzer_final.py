"""
ML-Based Sentiment Analyzer
Trains on Twitter data and analyzes user input text
Using 4 different ML models for high accuracy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import pickle
import warnings
warnings.filterwarnings('ignore')

class MLSentimentAnalyzer:
    """ML-based sentiment analyzer with multiple models"""
    
    def __init__(self):
        self.models = {}
        self.vectorizer = None
        self.label_map = {}
        self.reverse_label_map = {}
        self.best_model_name = None
        self.best_model = None
        self.is_trained = False
        
    def load_and_train(self, file_path, text_column='text', label_column='sentiment'):
        """Load data and train all models"""
        print("\n" + "="*70)
        print("🚀 LOADING DATA AND TRAINING ML MODELS")
        print("="*70)
        
        try:
            # Load data
            print(f"\n📂 Loading data from: {file_path}")
            df = pd.read_csv(file_path)
            print(f"✅ Loaded {len(df)} records")
            
            # Clean data
            df = df.dropna(subset=[text_column, label_column])
            print(f"✅ After cleaning: {len(df)} records")
            
            # Show label distribution
            print(f"\n📊 Sentiment Distribution:")
            print(df[label_column].value_counts())
            
            # Prepare data
            texts = df[text_column].astype(str).values
            labels = df[label_column].values
            
            # Create label mapping
            unique_labels = sorted(df[label_column].unique())
            self.label_map = {label: i for i, label in enumerate(unique_labels)}
            self.reverse_label_map = {v: k for k, v in self.label_map.items()}
            
            print(f"\n🏷️ Label Mapping: {self.label_map}")
            
            numeric_labels = np.array([self.label_map[label] for label in labels])
            
            # Vectorize text
            print(f"\n📝 Vectorizing text using TF-IDF...")
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                min_df=2,
                max_df=0.8,
                ngram_range=(1, 2),
                stop_words='english',
                lowercase=True
            )
            X = self.vectorizer.fit_transform(texts)
            print(f"✅ Created {X.shape[1]} features from {X.shape[0]} texts")
            
            # Split data
            print(f"\n📊 Splitting data (80% train, 20% test)...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, numeric_labels,
                test_size=0.2,
                random_state=42,
                stratify=numeric_labels
            )
            print(f"✅ Train: {len(y_train)} samples | Test: {len(y_test)} samples")
            
            # Train models
            self._train_models(X_train, X_test, y_train, y_test)
            
            self.is_trained = True
            print("\n✅ Training Complete!")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            return False
        
        return True
    
    def _train_models(self, X_train, X_test, y_train, y_test):
        """Train all 4 ML models"""
        print("\n" + "="*70)
        print("🤖 TRAINING 4 ML MODELS")
        print("="*70)
        
        model_configs = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, C=1.0),
            'Naive Bayes': MultinomialNB(alpha=1.0),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for model_name, model in model_configs.items():
            print(f"\n⏳ Training {model_name}...")
            
            # Train
            model.fit(X_train, y_train)
            self.models[model_name] = model
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted')
            
            results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred
            }
            
            print(f"✅ {model_name}")
            print(f"   Accuracy:  {accuracy:.4f}")
            print(f"   Precision: {precision:.4f}")
            print(f"   Recall:    {recall:.4f}")
            print(f"   F1-Score:  {f1:.4f}")
            print(f"   CV Score:  {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Print comparison table
        print("\n" + "="*70)
        print("📊 MODEL COMPARISON")
        print("="*70)
        comparison_df = pd.DataFrame(results).T
        print("\n" + comparison_df.to_string())
        
        # Find best model
        self.best_model_name = comparison_df['f1'].idxmax()
        self.best_model = self.models[self.best_model_name]
        
        print(f"\n🏆 BEST MODEL: {self.best_model_name}")
        print(f"   F1-Score: {results[self.best_model_name]['f1']:.4f}")
    
    def analyze_text(self, text):
        """Analyze sentiment of input text using all models"""
        if not self.is_trained:
            print("❌ Models not trained yet!")
            return
        
        print(f"\n" + "="*70)
        print(f"📝 ANALYZING TEXT")
        print("="*70)
        print(f"\nText: \"{text}\"\n")
        
        # Vectorize
        X = self.vectorizer.transform([text])
        
        results = []
        
        for model_name, model in self.models.items():
            # Predict
            pred = model.predict(X)[0]
            proba = model.predict_proba(X)[0] if hasattr(model, 'predict_proba') else None
            confidence = max(proba) if proba is not None else 0
            
            # Get sentiment label
            sentiment = self.reverse_label_map.get(pred, str(pred))
            
            results.append({
                'Model': model_name,
                'Sentiment': sentiment.upper(),
                'Confidence': f"{confidence:.2%}",
                'probabilities': proba
            })
        
        # Print results for each model
        print("📊 RESULTS FROM EACH MODEL:")
        print("-"*70)
        
        for i, result in enumerate(results, 1):
            emoji = self._get_emoji(result['Sentiment'])
            is_best = " ⭐ BEST" if result['Model'] == self.best_model_name else ""
            print(f"\n{i}. {result['Model']}{is_best}")
            print(f"   {emoji} Sentiment: {result['Sentiment']}")
            print(f"   📈 Confidence: {result['Confidence']}")
        
        # Consensus prediction
        print("\n" + "-"*70)
        print("🎯 ENSEMBLE VOTING (Consensus):")
        print("-"*70)
        sentiments = [r['Sentiment'] for r in results]
        consensus_sentiment = max(set(sentiments), key=sentiments.count)
        consensus_count = sentiments.count(consensus_sentiment)
        consensus_percentage = (consensus_count / len(sentiments)) * 100
        
        emoji = self._get_emoji(consensus_sentiment)
        print(f"\n{emoji} Consensus Sentiment: {consensus_sentiment}")
        print(f"📊 Agreement: {consensus_count}/{len(sentiments)} models ({consensus_percentage:.0f}%)")
        
        # Best model prediction
        print("\n" + "-"*70)
        print(f"🏆 BEST MODEL PREDICTION ({self.best_model_name}):")
        print("-"*70)
        best_result = next(r for r in results if r['Model'] == self.best_model_name)
        emoji = self._get_emoji(best_result['Sentiment'])
        print(f"\n{emoji} Sentiment: {best_result['Sentiment']}")
        print(f"📈 Confidence: {best_result['Confidence']}")
    
    def _get_emoji(self, sentiment):
        """Get emoji for sentiment"""
        sentiment = sentiment.lower()
        if 'positive' in sentiment:
            return "😊"
        elif 'negative' in sentiment:
            return "😞"
        else:
            return "😐"
    
    def save_model(self, filename='sentiment_model.pkl'):
        """Save trained model and vectorizer"""
        if not self.is_trained:
            print("❌ No trained model to save!")
            return
        
        try:
            with open(filename, 'wb') as f:
                pickle.dump({
                    'models': self.models,
                    'vectorizer': self.vectorizer,
                    'label_map': self.label_map,
                    'reverse_label_map': self.reverse_label_map,
                    'best_model_name': self.best_model_name
                }, f)
            print(f"\n✅ Model saved: {filename}")
        except Exception as e:
            print(f"\n❌ Error saving model: {e}")
    
    def load_model(self, filename='sentiment_model.pkl'):
        """Load trained model and vectorizer"""
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            
            self.models = data['models']
            self.vectorizer = data['vectorizer']
            self.label_map = data['label_map']
            self.reverse_label_map = data['reverse_label_map']
            self.best_model_name = data['best_model_name']
            self.best_model = self.models[self.best_model_name]
            self.is_trained = True
            
            print(f"\n✅ Model loaded: {filename}")
        except Exception as e:
            print(f"\n❌ Error loading model: {e}")


def main():
    """Main program"""
    print("\n" + "="*70)
    print("🤖 ML-BASED SENTIMENT ANALYZER")
    print("Train on Twitter Data & Analyze User Text")
    print("="*70)
    
    analyzer = MLSentimentAnalyzer()
    
    # Load training data
    print("\n📌 STEP 1: LOAD AND TRAIN ON DATA")
    file_path = input("\nEnter path to your Twitter CSV file: ").strip()
    
    text_col = input("Enter text column name (default: 'text'): ").strip() or 'text'
    label_col = input("Enter sentiment label column name (default: 'sentiment'): ").strip() or 'sentiment'
    
    if not analyzer.load_and_train(file_path, text_column=text_col, label_column=label_col):
        return
    
    # Ask to save model
    save_choice = input("\n💾 Save trained model? (yes/no): ").strip().lower()
    if save_choice in ['yes', 'y']:
        analyzer.save_model('sentiment_model.pkl')
    
    # Analyze user input
    print("\n\n" + "="*70)
    print("📌 STEP 2: ANALYZE YOUR TEXT")
    print("="*70)
    print("\n📝 Enter text to analyze (type 'quit' to exit)\n")
    
    while True:
        text = input("Enter text: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Thank you for using ML Sentiment Analyzer!")
            break
        
        if not text:
            print("❌ Please enter some text\n")
            continue
        
        analyzer.analyze_text(text)
        
        another = input("\n\n🔄 Analyze another text? (yes/no): ").strip().lower()
        if another not in ['yes', 'y']:
            print("\n👋 Thank you for using ML Sentiment Analyzer!")
            break


if __name__ == "__main__":
    main()
