# 🚀 ML Sentiment Analyzer - Quick Start Guide

## What Is This?

A **streamlined ML-based sentiment analyzer** that:
1. **Trains** on your Twitter data (loading CSV)
2. **Compares** 4 different ML models
3. **Analyzes** any text you input
4. **Outputs** sentiment (Positive/Negative/Neutral) with confidence scores

---

## 📦 Files

### `ml_sentiment_analyzer_final.py` ⭐ **USE THIS**
- **Standalone Python script**
- Interactive command-line interface
- No Jupyter needed
- Run and start analyzing immediately

### `ML_Sentiment_Analyzer_Final.ipynb`
- **Jupyter notebook version**
- Step-by-step learning
- Great for understanding each step
- Visualizations included

---

## ⚡ Quick Start (2 Minutes)

### Step 1: Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib
```

### Step 2: Run the Script
```bash
python ml_sentiment_analyzer_final.py
```

### Step 3: Follow Prompts
```
1. Enter path to your Twitter CSV file
2. Enter text column name (e.g., 'tweet', 'text')
3. Enter sentiment column name (e.g., 'sentiment', 'label')
4. Wait for training (~30 seconds)
5. Start analyzing text!
```

---

## 📝 CSV File Format

Your CSV should have at least 2 columns:

### Example 1 (Text labels)
```csv
tweet,sentiment
"I love this product!",positive
"This is terrible",negative
"It's okay",neutral
```

### Example 2 (Numeric labels)
```csv
text,sentiment
"Great experience!",1
"Bad experience",0
"Average",2
```

### Example 3 (Different column names)
```csv
content,label
"Amazing!",Positive
"Awful!",Negative
```

---

## 🤖 4 ML Models Trained

The script trains and compares these models:

1. **Logistic Regression**
   - Fast and simple
   - Good baseline
   - Speed: ⚡⚡⚡

2. **Naive Bayes**
   - Good for text
   - Very fast
   - Speed: ⚡⚡⚡

3. **Random Forest**
   - Ensemble method
   - Often most accurate
   - Speed: ⚡⚡

4. **Gradient Boosting**
   - Advanced ensemble
   - Very accurate
   - Speed: ⚡

---

## 📊 Output Explanation

### Individual Model Results
Shows prediction from each model with confidence:
```
1. Logistic Regression
   😊 Sentiment: POSITIVE
   📈 Confidence: 85.23%

2. Naive Bayes
   😊 Sentiment: POSITIVE
   📈 Confidence: 78.45%
```

### Ensemble Voting
Shows consensus from all 4 models:
```
🎯 ENSEMBLE VOTING (Consensus):

😊 Consensus Sentiment: POSITIVE
📊 Agreement: 4/4 models (100%)
```

### Best Model Prediction
Shows prediction from the single best model:
```
🏆 BEST MODEL PREDICTION (Random Forest):

😊 Sentiment: POSITIVE
📈 Confidence: 92.15%
```

---

## 💻 Usage Example

### Run the Script
```bash
python ml_sentiment_analyzer_final.py
```

### Input Your File Path
```
📂 Loading data and training ML models

📁 Enter path to your Twitter CSV file: 
C:\Users\Lenovo\Downloads\Twitter_Data.csv
```

### Enter Column Names
```
📝 Enter text column name (default: 'text'): 
tweet

📝 Enter sentiment label column name (default: 'sentiment'): 
sentiment
```

### Training (Automatic)
```
🤖 TRAINING 4 ML MODELS

⏳ Training Logistic Regression...
✅ Logistic Regression
   Accuracy:  0.8523
   Precision: 0.8456
   Recall:    0.8532
   F1-Score:  0.8494
   CV Score:  0.8412 ± 0.0234

[... training other models ...]

🏆 BEST MODEL: Random Forest
   F1-Score: 0.8934
```

### Save Model (Optional)
```
💾 Save trained model? (yes/no): yes
✅ Model saved: sentiment_model.pkl
```

### Analyze Text
```
📌 STEP 2: ANALYZE YOUR TEXT

📝 Enter text to analyze (type 'quit' to exit)

Enter text: I absolutely love this product!

================================================================
📝 ANALYZING TEXT
================================================================

Text: "I absolutely love this product!"

📊 RESULTS FROM EACH MODEL:
-----

1. Logistic Regression
   😊 Sentiment: POSITIVE
   📈 Confidence: 89.23%

2. Naive Bayes
   😊 Sentiment: POSITIVE
   📈 Confidence: 85.12%

3. Random Forest ⭐ BEST
   😊 Sentiment: POSITIVE
   📈 Confidence: 94.56%

4. Gradient Boosting
   😊 Sentiment: POSITIVE
   📈 Confidence: 91.23%

---
🎯 ENSEMBLE VOTING (Consensus):

😊 Consensus Sentiment: POSITIVE
📊 Agreement: 4/4 models (100%)

---
🏆 BEST MODEL PREDICTION (Random Forest):

😊 Sentiment: POSITIVE
📈 Confidence: 94.56%
```

---

## 🔍 Understanding Sentiment Labels

### Three Classes
- **POSITIVE** (😊): Good, love, great, excellent, amazing, etc.
- **NEGATIVE** (😞): Bad, hate, terrible, awful, disappointed, etc.
- **NEUTRAL** (😐): Okay, average, could be better, etc.

### Confidence Score
- **0-50%**: Model is uncertain
- **50-75%**: Model is fairly confident
- **75-90%**: Model is very confident
- **90-100%**: Model is extremely confident

### Agreement
- **4/4 (100%)**: All models agree
- **3/4 (75%)**: Most models agree
- **2/4 (50%)**: Split prediction (unreliable)

---

## 💾 Save and Reuse Models

### Automatic Save
```bash
# When running the script, choose to save the model
💾 Save trained model? (yes/no): yes
✅ Model saved: sentiment_model.pkl
```

### Manual Save in Python
```python
import pickle

with open('my_model.pkl', 'wb') as f:
    pickle.dump({
        'models': analyzer.models,
        'vectorizer': analyzer.vectorizer,
        'label_map': analyzer.label_map,
        'reverse_label_map': analyzer.reverse_label_map,
        'best_model_name': analyzer.best_model_name
    }, f)
```

### Load and Use Later
```python
import pickle

with open('sentiment_model.pkl', 'rb') as f:
    data = pickle.load(f)
    
models = data['models']
vectorizer = data['vectorizer']
# Now analyze new text without retraining
```

---

## 🎯 Expected Accuracy

With good training data (100+ samples):

| Metric | Expected Value |
|--------|-----------------|
| Accuracy | 85-95% |
| Precision | 85-95% |
| Recall | 85-95% |
| F1-Score | 85-95% |

---

## ⚠️ Troubleshooting

### "File not found"
**Solution:**
- Use full path: `C:/Users/Lenovo/Downloads/Twitter_Data.csv`
- Use forward slashes `/` instead of backslashes `\`
- Check file exists in that location

### "Column not found"
**Solution:**
- Check exact column name in CSV (case-sensitive)
- Use `head()` to see column names first

### "Memory error"
**Solution:**
- Use smaller CSV file
- Reduce `max_features` in TfidfVectorizer (change 5000 to 1000)

### "Low accuracy"
**Solution:**
- Use more training data (100+ samples per class minimum)
- Check label quality
- Try different column names

---

## 🔧 Customization

### Change Models
Edit the `model_configs` dictionary:
```python
model_configs = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Naive Bayes': MultinomialNB(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}
```

### Change TF-IDF Parameters
```python
vectorizer = TfidfVectorizer(
    max_features=3000,      # Change from 5000
    min_df=1,              # Change from 2
    max_df=0.9,            # Change from 0.8
    ngram_range=(1, 3),    # Add trigrams
    stop_words='english'
)
```

### Exclude a Model
```python
# Remove Gradient Boosting from comparison
del model_configs['Gradient Boosting']
```

---

## 📚 Key Concepts

### TF-IDF
- Converts text to numbers
- TF = how often word appears
- IDF = how unique/important the word is
- ML models need numbers, not text

### Training vs Testing
- **Training**: Model learns patterns (80% of data)
- **Testing**: Check if model works on new data (20% of data)

### Cross-Validation
- Test model on multiple data splits
- More reliable than single test
- 5-fold = split data 5 ways, test 5 times

### Overfitting
- Model memorizes training data
- Fails on new data
- Solved with: more data, regularization, cross-validation

---

## 🎓 Learning Resources

- **Scikit-learn**: https://scikit-learn.org/
- **TF-IDF Explanation**: https://en.wikipedia.org/wiki/Tf%E2%80%93idf
- **ML Basics**: https://www.coursera.org/learn/machine-learning
- **NLP Guide**: https://www.coursera.org/learn/natural-language-processing

---

## ✅ Checklist Before Running

- [ ] Python 3.7+ installed
- [ ] Required libraries installed: `pip install pandas numpy scikit-learn`
- [ ] Twitter CSV file ready
- [ ] CSV has 'text' and 'sentiment' columns (or similar)
- [ ] At least 50-100 tweets in CSV
- [ ] File path ready to paste

---

## 🎉 Summary

This ML Sentiment Analyzer:

✅ **Trains** 4 models automatically
✅ **Compares** their performance
✅ **Analyzes** any text you input
✅ **Shows** results from each model
✅ **Provides** consensus prediction
✅ **Saves** models for reuse
✅ **Achieves** 85-95% accuracy

**Start with:** `python ml_sentiment_analyzer_final.py`

**Questions?** Check the code comments or run the Jupyter notebook for step-by-step explanation.

---

**Ready to analyze sentiment? Run the script now!** 🚀
