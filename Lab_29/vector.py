import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import metrics

# --- Task 1 & 2: Prepare the Text Data ---
documents = [
    'I love machine learning.',
    'Machine learning is amazing.',
    'I love creating machine learning models.',
    'Models are crucial for predictive analytics.'
]
# 1 represents positive/learning sentiment, 0 represents neutral/analytics
labels = [1, 1, 1, 0] 

# --- Step 1: Implement Bag-of-Words (CountVectorizer) ---
print("Task 1: Applying Bag-of-Words...")
vectorizer_bow = CountVectorizer()
X_bow = vectorizer_bow.fit_transform(documents)

print("BoW Feature Names:", vectorizer_bow.get_feature_names_out())
print("BoW Matrix:\n", X_bow.toarray())

# --- Step 2: Implement TF-IDF Vectorization ---
print("\nTask 2: Applying TF-IDF Vectorization...")
vectorizer_tfidf = TfidfVectorizer()
X_tfidf = vectorizer_tfidf.fit_transform(documents)

print("TF-IDF Feature Names:", vectorizer_tfidf.get_feature_names_out())
print("TF-IDF Matrix:\n", X_tfidf.toarray())

# --- Task 3: Train Classifier and Compare Results ---
print("\nTask 3: Comparing Classifier Performance...")

# Split Data (Note: test_size=0.5 because we only have 4 samples)
X_train_bow, X_test_bow, y_train, y_test = train_test_split(
    X_bow, labels, test_size=0.5, random_state=42
)

X_train_tfidf, X_test_tfidf, _, _ = train_test_split(
    X_tfidf, labels, test_size=0.5, random_state=42
)

# Train Naive Bayes using Bag-of-Words
clf_bow = MultinomialNB()
clf_bow.fit(X_train_bow, y_train)
y_pred_bow = clf_bow.predict(X_test_bow)
acc_bow = metrics.accuracy_score(y_test, y_pred_bow)

# Train Naive Bayes using TF-IDF
clf_tfidf = MultinomialNB()
clf_tfidf.fit(X_train_tfidf, y_train)
y_pred_tfidf = clf_tfidf.predict(X_test_tfidf)
acc_tfidf = metrics.accuracy_score(y_test, y_pred_tfidf)

# --- Final Output ---
print("-" * 45)
print(f"{'Vectorization Method':<25} | {'Accuracy':<10}")
print("-" * 45)
print(f"{'Bag-of-Words':<25} | {acc_bow:<10}")
print(f"{'TF-IDF':<25} | {acc_tfidf:<10}")
print("-" * 45)

print("\nConclusion:")
print("- Bag-of-Words counts raw frequency.")
print("- TF-IDF penalizes common words to highlight unique meaning.")
