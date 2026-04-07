import nltk
import pandas as pd
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

# --- Task 1: Load a Text Dataset ---
print("Task 1: Downloading and Loading Movie Reviews...")
# Download necessary NLTK resources
nltk.download('movie_reviews', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)  # Add this line

# Load documents: (words, category)
# Categories are usually 'pos' (positive) or 'neg' (negative)
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

print(f"✓ Total number of documents loaded: {len(documents)}")

# --- Task 2: Tokenize Text ---
print("\nTask 2: Tokenizing the first document...")
# We take the first document and join the list of words into a string to demonstrate tokenization
first_doc_raw_text = ' '.join(documents[0][0])
tokens = word_tokenize(first_doc_raw_text)

print(f"✓ Total tokens in first document: {len(tokens)}")
print(f"✓ First 10 tokens: {tokens[:10]}")

# --- Task 3: Generate Word Frequencies ---
print("\nTask 3: Calculating Word Frequencies...")
# Generate frequency distribution
fdist = FreqDist(tokens)

# Display top 10
most_common_words = fdist.most_common(10)

print("-" * 30)
print(f"{'WORD':<10} | {'COUNT':<6}")
print("-" * 30)
for word, count in most_common_words:
    print(f"{word:<10} | {count:<6}")
print("-" * 30)

# --- Analysis & Discussion ---
print("\n[Lab Observations]")
print("1. Notice that the most common words are punctuation (',', '.') and ")
print("   functional words ('the', 'a', 'and'). These are called 'Stopwords'.")
print("2. In real-world NLP, we usually remove these because they don't carry ")
print("   much sentimental meaning.")
