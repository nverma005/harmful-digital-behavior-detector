# ============================================================
# STEP 3: Data Cleaning & Preprocessing
# Project: Harmful Digital Behavior Detector
# ============================================================

# We need these libraries:
# pandas  — for data handling (you already know this)
# re      — built-in Python library for text pattern matching
#            "re" stands for Regular Expressions
#            It lets us find/remove things like URLs, symbols
import pandas as pd
import re
import os

# Create output folder if it doesn't exist
OUTPUT_FOLDER = "output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
# os.makedirs creates the folder automatically
# exist_ok=True means no error if folder already exists
print(f"Output folder ready: {OUTPUT_FOLDER}/")

# ============================================================
# LOAD THE EXPLORED DATA (output from Step 2)
# ============================================================

df = pd.read_csv(f"{OUTPUT_FOLDER}/explored_data.csv")
print("Loaded data. Shape:", df.shape)

# ============================================================
# STEP 3A: DROP USELESS COLUMN
# ============================================================

# "Unnamed: 0" is just a copy of the row index.
# It carries zero information. We drop it.
# axis=1 means "drop a column" (axis=0 would drop a row)
df = df.drop(columns=['Unnamed: 0'])
print("Dropped 'Unnamed: 0' column. Columns now:", df.columns.tolist())

# ============================================================
# STEP 3B: HANDLE MISSING TEXT (the 350 nulls)
# ============================================================

# Strategy: if text is missing, use the title as the text.
# Why? Because every post has a title. A title like
# "I feel like nobody understands me" is still useful data.
# We don't throw away 350 rows — that's 6% of our dataset.

# fillna() fills null values with whatever you specify
# Here we fill missing 'text' with the value from 'title'
df['text'] = df['text'].fillna(df['title'])

# Also catch empty strings (text exists but is "" or "  ")
# str.strip() removes spaces. If result is empty, replace with title
df['text'] = df.apply(
    lambda row: row['title'] if str(row['text']).strip() == '' else row['text'],
    axis=1   # axis=1 means apply function row by row
)

print("Missing text after fix:", df['text'].isnull().sum())  # Should be 0

# ============================================================
# STEP 3C: TEXT CLEANING FUNCTION
# ============================================================

# We will write ONE function that cleans a single piece of text.
# Then we apply it to every row automatically.

def clean_text(text):
    """
    Cleans a raw Reddit post text.
    Input:  a messy string like "Check [this](http://...) out!!!"
    Output: a clean string like "check this out"
    """

    # 1. Convert to string (safety — in case any value is not a string)
    text = str(text)

    # 2. Remove URLs (http://... or https://...)
    # re.sub(pattern, replacement, text) finds the pattern and replaces it
    # r'http\S+' means: "http" followed by any non-space characters
    text = re.sub(r'http\S+', '', text)

    # 3. Remove Reddit markdown formatting
    # Things like **bold**, *italic*, ~~strikethrough~~
    # \* means literal asterisk (not special regex symbol)
    text = re.sub(r'\*+', '', text)     # removes * and **
    text = re.sub(r'~+', '', text)      # removes ~~
    text = re.sub(r'#+', '', text)      # removes ### headings
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # Above: [link text](url) → keeps just "link text"

    # 4. Remove HTML entities like &amp; &lt; &#x200B;
    # These are leftover HTML codes that appear in Reddit data
    text = re.sub(r'&[a-z]+;', ' ', text)
    text = re.sub(r'&#x[0-9A-Fa-f]+;', ' ', text)

    # 5. Remove newlines and tabs — replace with a space
    text = re.sub(r'[\n\r\t]', ' ', text)

    # 6. Remove special characters — keep only letters, numbers, spaces
    # [^a-zA-Z0-9\s] means "anything that is NOT a letter, digit, or space"
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    # 7. Convert to lowercase
    # Why? So "Depression" and "depression" are treated as the same word
    text = text.lower()

    # 8. Remove extra spaces (multiple spaces → single space)
    text = re.sub(r'\s+', ' ', text)

    # 9. Strip leading/trailing spaces
    text = text.strip()

    return text

# ============================================================
# STEP 3D: APPLY CLEANING TO ALL ROWS
# ============================================================

# .apply() runs the function on every single row
# This may take a few seconds — it's processing 5,957 posts
print("\nCleaning text... please wait...")
df['clean_text'] = df['text'].apply(clean_text)
df['clean_title'] = df['title'].apply(clean_text)
print("Text cleaning done!")

# ============================================================
# STEP 3E: COMBINED TEXT COLUMN
# ============================================================

# For analysis, we want ONE text field per post.
# Combining title + text gives us maximum information.
# We add a space " " between them.
df['full_text'] = df['clean_title'] + ' ' + df['clean_text']

# ============================================================
# STEP 3F: FEATURE ENGINEERING — CREATE USEFUL COLUMNS
# ============================================================

# "Feature engineering" means creating NEW columns
# that will help our detection algorithms.
# Each column below captures a real behavioral signal.

print("\nEngineering features...")

# --- FEATURE 1: Word Count ---
# How many words does the post contain?
# .split() breaks text into a list of words by spaces
# len() counts how many words
df['word_count'] = df['clean_text'].apply(lambda x: len(x.split()))

# --- FEATURE 2: Character Count (clean) ---
df['char_count'] = df['clean_text'].apply(len)

# --- FEATURE 3: Sentence Count ---
# We count sentences by counting punctuation (. ! ?)
# in the ORIGINAL text (before cleaning removed punctuation)
df['sentence_count'] = df['text'].apply(
    lambda x: len(re.findall(r'[.!?]', str(x)))
)

# --- FEATURE 4: Negativity Score ---
# We count how many negative/distress words appear in the post.
# This is a simplified version of sentiment analysis.
# More negative words = higher distress = potential doom scroll content.

# These are our "negativity keywords" — words linked to
# mental health distress and doom-scroll-type content
negative_words = [
    'depressed', 'depression', 'anxiety', 'hopeless', 'worthless',
    'suicidal', 'die', 'death', 'hate', 'pain', 'hurt', 'suffering',
    'lonely', 'alone', 'empty', 'numb', 'exhausted', 'tired',
    'fail', 'failure', 'ugly', 'stupid', 'useless', 'broken',
    'crying', 'cry', 'sad', 'miserable', 'awful', 'terrible',
    'nightmare', 'panic', 'fear', 'scared', 'terrified', 'helpless',
    'trapped', 'stuck', 'lost', 'dark', 'darkness', 'void'
]

def count_negative_words(text):
    """Count how many negativity keywords appear in the text."""
    text_lower = str(text).lower()
    count = 0
    for word in negative_words:
        # We use word boundaries to avoid partial matches
        # e.g., "pain" should not match "painting"
        count += len(re.findall(r'\b' + word + r'\b', text_lower))
    return count

df['negative_word_count'] = df['clean_text'].apply(count_negative_words)

# --- FEATURE 5: Negativity Density ---
# Raw count is affected by post length (long posts have more words).
# Density = negativity per 100 words — a FAIRER measure.
# We add 1 to word_count in the denominator to avoid dividing by zero.
df['negativity_density'] = (
    df['negative_word_count'] / (df['word_count'] + 1) * 100
).round(2)

# --- FEATURE 6: Is Long Post ---
# Posts longer than 500 words are flagged as "long"
# Long + negative = strong doom scroll indicator
df['is_long_post'] = (df['word_count'] > 500).astype(int)
# .astype(int) converts True→1, False→0

# --- FEATURE 7: Category name (readable label) ---
category_map = {
    0: "Stress",
    1: "Depression",
    2: "Bipolar",
    3: "Social Anxiety",
    4: "General Anxiety"
}
df['category_name'] = df['target'].map(category_map)

# ============================================================
# STEP 3G: QUALITY CHECK
# ============================================================

print("\n=== QUALITY CHECK ===")
print("Shape after cleaning:", df.shape)
print("Any nulls remaining?")
print(df[['clean_text', 'full_text', 'word_count', 
          'negativity_density']].isnull().sum())

print("\nSample of engineered features (first 5 rows):")
print(df[['category_name', 'word_count', 
          'negative_word_count', 'negativity_density', 
          'is_long_post']].head())

print("\nNegative word count stats:")
print(df['negative_word_count'].describe().round(2))

print("\nNegativity density stats (per 100 words):")
print(df['negativity_density'].describe().round(2))

# ============================================================
# STEP 3H: SAVE CLEAN DATA
# ============================================================

df.to_csv(f"{OUTPUT_FOLDER}/clean_data.csv", index=False)
print("\nclean_data.csv saved successfully!")
print("Total rows:", len(df))
print("Total columns:", len(df.columns))
print("Columns:", df.columns.tolist())