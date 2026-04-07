import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# --- Task 1: Load and Inspect Dataset ---
if not os.path.exists('IMDB Dataset.csv'):
    print("Error: IMDB Dataset.csv not found. Please download it first.")
    exit()

print("Task 1: Loading and Cleaning Dataset...")
df = pd.read_csv('IMDB Dataset.csv')

# The standard IMDB dataset uses 'review' and 'sentiment' columns
# Some versions use 'text' and 'label'. Adjusting for compatibility:
if 'text' in df.columns:
    df.rename(columns={'text': 'review', 'label': 'sentiment'}, inplace=True)

# Function to clean text (Task 1.3)
def clean_text(text):
    text = re.sub(r'<[^>]+>', ' ', text)  # Remove HTML tags (br, etc)
    text = re.sub(r'[^\w\s]', '', text)    # Remove punctuation
    text = text.lower()                   # Normalize to lowercase
    return text

df['review'] = df['review'].apply(clean_text)
print(f"✓ Dataset loaded. Total reviews: {len(df)}")

# --- Task 2: Use a Simple ML Model for Classification ---
print("\nTask 2: Vectorizing text and training model...")

X = df['review']
y = df['sentiment']

# Split Data (75% Train, 25% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Text Vectorization (TF-IDF)
# max_features=5000 keeps the 5000 most 'important' words
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# --- Task 3: Evaluate Accuracy ---
print("\nTask 3: Evaluating Results...")
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print("-" * 40)
print(f'OVERALL MODEL ACCURACY: {accuracy:.2%}')
print("-" * 40)

# Show a few example predictions
print("\n[Example Predictions]")
samples = X_test.iloc[:3].tolist()
preds = y_pred[:3]
for text, pred in zip(samples, preds):
    print(f"Review: {text[:70]}...")
    print(f"Predicted Sentiment: {pred.upper()}\n")

# --- Analysis & Discussion ---
print("[Lab Discussion]")
print("- TF-IDF helps because it lowers the weight of common words like 'the' ")
print("  and raises the weight of meaningful words like 'excellent' or 'boring'.")
print("- Accuracy above 85% is considered strong for this simple model.")

