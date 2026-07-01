# ============================================================
# STEP 2: Data Exploration
# Project: Harmful Digital Behavior Detector
# ============================================================

# WHAT IS PANDAS?
# pandas is a Python library that lets you work with data
# like a spreadsheet — rows, columns, filters, summaries.
# We import it and give it a short nickname "pd"
import pandas as pd
import os

# Create output folder if it doesn't exist
OUTPUT_FOLDER = "output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
# os.makedirs creates the folder automatically
# exist_ok=True means no error if folder already exists
print(f"Output folder ready: {OUTPUT_FOLDER}/")

# ============================================================
# LOAD THE DATASET
# ============================================================

# pd.read_csv() reads your CSV file into a "DataFrame"
# A DataFrame is like an Excel table inside Python
df = pd.read_csv("data\Reddit Mental Health Data.csv")

# ============================================================
# BASIC INSPECTION — always do this first with any dataset
# ============================================================

'''print(df['target'].head(50)) 
print(df['target'].tail(50))'''

# How many rows and columns?
print("Shape (rows, columns):", df.shape)

# What are the column names?
print("\nColumn names:", df.columns.tolist())

# Show first 3 rows
print("\nFirst 3 rows:")
print(df.head(3))

# Show data types of each column
print("\nData types:")
print(df.dtypes)

# ============================================================
# CHECK FOR MISSING (NULL) VALUES
# ============================================================

# isnull() marks every missing cell as True
# .sum() counts how many True values per column
print("\nMissing values per column:")
print(df.isnull().sum())

# ============================================================
# UNDERSTAND THE TARGET COLUMN (our categories)
# ============================================================

# value_counts() tells us how many posts are in each category
print("\nPosts per category (target):")
print(df['target'].value_counts().sort_index())

# ============================================================
# TEXT LENGTH ANALYSIS
# ============================================================

# fillna('') replaces missing text with empty string (safe)
# .apply(len) applies the len() function to every row
df['text_length'] = df['text'].fillna('').apply(len)
df['title_length'] = df['title'].fillna('').apply(len)

# describe() gives count, mean, min, max, percentiles
print("\nText length statistics (in characters):")
print(df['text_length'].describe().round(0))

print("\nTitle length statistics (in characters):")
print(df['title_length'].describe().round(0))

# ============================================================
# CATEGORY LABELS — give human-readable names to 0,1,2,3,4
# ============================================================

# A dictionary maps each number to a name
category_map = {
    0: "Stress",
    1: "Depression",
    2: "Bipolar",
    3: "Social Anxiety",
    4: "General Anxiety"
}

# Create a new column with the readable name
df['category_name'] = df['target'].map(category_map)

print("\nCategory distribution:")
print(df['category_name'].value_counts())

# ============================================================
# SAVE THE EXPLORED VERSION (with new columns)
# ============================================================

df.to_csv(f"{OUTPUT_FOLDER}/explored_data.csv", index=False)
print("\nSaved explored_data.csv successfully!")
print("Total rows:", len(df))