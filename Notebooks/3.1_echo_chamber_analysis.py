# ============================================================
# STEP 4A: Echo Chamber Detector
# Project: Harmful Digital Behavior Detector
# ============================================================

# CONCEPT FIRST — what is an echo chamber, mathematically?
#
# Imagine User A reads 10 posts.
# All 10 are from "Depression" category.
# → They are in a perfect echo chamber (zero diversity).
#
# User B reads 10 posts spread across all 5 categories.
# → They have maximum diversity (no echo chamber).
#
# We measure this using "Shannon Entropy" — a formula from
# information theory that measures diversity/variety.
# High entropy = diverse content = healthy = no echo chamber
# Low entropy  = narrow content  = risky  = echo chamber
#
# We also simulate "users" because our dataset has posts,
# not individual user session histories. We assign posts to
# simulated users randomly to mimic real browsing behavior.

import pandas as pd
import numpy as np
import os
# numpy is Python's number-crunching library
# "np" is the standard short nickname

# Create output folder if it doesn't exist
OUTPUT_FOLDER = "output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
# os.makedirs creates the folder automatically
# exist_ok=True means no error if folder already exists
print(f"Output folder ready: {OUTPUT_FOLDER}/")

# ============================================================
# LOAD CLEAN DATA
# ============================================================

df = pd.read_csv(f"{OUTPUT_FOLDER}/clean_data.csv")
print("Loaded clean data. Shape:", df.shape)

# ============================================================
# STEP 4A-1: SIMULATE USERS
# ============================================================

# In reality, each user reads a sequence of posts.
# We don't have real user IDs in this dataset, so we simulate them.
#
# We create 100 users, each assigned a random set of posts.
# Some users will be assigned posts mostly from one category
# (to simulate real echo chamber behavior).

np.random.seed(42)
# random.seed(42) makes randomness reproducible.
# Every time you run this, you get the same "random" result.
# 42 is just a convention (from The Hitchhiker's Guide to the Galaxy).

NUM_USERS = 100
NUM_POSTS_PER_USER = 30 # each simulated user reads 30 posts

# We will store each user's data in a list, then convert to DataFrame
user_records = []

for user_id in range(NUM_USERS):

    # --- HIGH RISK users (user 0–29): extreme bias ---
    # 90% posts from one category, 10% from others
    # These simulate someone deep in one subreddit bubble
    if user_id < 30:
        dominant_category = user_id % 5  # cycles through 0,1,2,3,4
        dominant_posts = df[df['target'] == dominant_category].sample(
            n=27, replace=True, random_state=user_id       # 90% = 27/30
        )
        other_posts = df[df['target'] != dominant_category].sample(
            n=3, replace=True, random_state=user_id + 1000 # 10% = 3/30
        )
        user_posts = pd.concat([dominant_posts, other_posts])

    # --- MEDIUM RISK users (user 30–49): moderate bias ---
    # 60% from one category, 40% spread across others
    elif user_id < 50:
        dominant_category = user_id % 5
        dominant_posts = df[df['target'] == dominant_category].sample(
            n=18, replace=True, random_state=user_id       # 60% = 18/30
        )
        other_posts = df[df['target'] != dominant_category].sample(
            n=12, replace=True, random_state=user_id + 1000 # 40% = 12/30
        )
        user_posts = pd.concat([dominant_posts, other_posts])

    # --- LOW RISK users (user 50–99): balanced browsing ---
    # Roughly equal posts from all 5 categories
    else:
        user_posts = df.sample(
            n=NUM_POSTS_PER_USER, replace=True, random_state=user_id
        )

    # Count posts per category for this user
    category_counts = user_posts['target'].value_counts()

    # Make sure all 5 categories exist in counts (fill missing with 0)
    for cat in range(5):
        if cat not in category_counts:
            category_counts[cat] = 0

    category_map = {
        0: 'Stress', 1: 'Depression', 2: 'Bipolar',
        3: 'Social Anxiety', 4: 'General Anxiety'
    }

    user_records.append({
        'user_id':                f'user_{user_id:03d}',
        'total_posts_read':       NUM_POSTS_PER_USER,
        'count_stress':           category_counts.get(0, 0),
        'count_depression':       category_counts.get(1, 0),
        'count_bipolar':          category_counts.get(2, 0),
        'count_social_anxiety':   category_counts.get(3, 0),
        'count_general_anxiety':  category_counts.get(4, 0),
        'dominant_category':      category_counts.idxmax(),
        'dominant_category_name': category_map[category_counts.idxmax()],
        'dominant_pct':           round(
            category_counts.max() / NUM_POSTS_PER_USER * 100, 1
        ),
        'avg_negativity':         round(
            user_posts['negativity_density'].mean(), 2
        ),
        'avg_word_count':         round(
            user_posts['word_count'].mean(), 1
        ),
    })

users_df = pd.DataFrame(user_records)
print(f"\nUsers created: {len(users_df)}")

# ============================================================
# STEP 4A-2: SHANNON ENTROPY (Diversity Score)
# ============================================================

# Shannon Entropy formula:
# H = - sum( p * log2(p) ) for each category p
#
# Where p = proportion of posts from that category.
# Example:
#   User reads 30 posts, all from Depression:
#   p = [1.0, 0, 0, 0, 0]
#   H = -(1.0 * log2(1.0)) = 0   ← minimum entropy (echo chamber)
#
#   User reads 6 posts from each of 5 categories:
#   p = [0.2, 0.2, 0.2, 0.2, 0.2]
#   H = -(5 * 0.2 * log2(0.2)) = 2.32  ← maximum entropy (diverse)

MAX_ENTROPY = np.log2(5)   # 2.3219 = maximum possible diversity

def shannon_entropy(row):
    counts = np.array([
        row['count_stress'],       row['count_depression'],
        row['count_bipolar'],      row['count_social_anxiety'],
        row['count_general_anxiety']
    ])
    total = counts.sum()
    proportions = counts / total
    entropy = 0
    for p in proportions:
        if p > 0:
            entropy -= p * np.log2(p)
    return round(entropy, 4)

users_df['diversity_score']  = users_df.apply(shannon_entropy, axis=1)
users_df['diversity_pct']    = (
    users_df['diversity_score'] / MAX_ENTROPY * 100
).round(1)

# ============================================================
# STEP 4A-3: RECALIBRATED RISK THRESHOLDS
# ============================================================

# With 5 categories:
#   90/10 split → entropy ≈ 0.47 → diversity_pct ≈ 20%  → HIGH risk
#   60/40 split → entropy ≈ 1.50 → diversity_pct ≈ 65%  → MEDIUM risk
#   Equal split → entropy ≈ 2.32 → diversity_pct ≈ 100% → LOW risk
#
# Thresholds:
#   < 50%  = HIGH   (clearly in a bubble)
#   50–75% = MEDIUM (mild bubble)
#   > 75%  = LOW    (healthy diverse browsing)

def echo_chamber_risk(diversity_pct):
    if diversity_pct < 50:
        return 'HIGH'
    elif diversity_pct < 75:
        return 'MEDIUM'
    else:
        return 'LOW'

users_df['echo_chamber_risk'] = users_df['diversity_pct'].apply(
    echo_chamber_risk
)

# ============================================================
# STEP 4A-4: RESULTS
# ============================================================

print("\n=== ECHO CHAMBER DETECTION RESULTS ===")

print("\nRisk distribution:")
print(users_df['echo_chamber_risk'].value_counts())

print("\nDiversity score stats:")
print(users_df['diversity_score'].describe().round(3))

print("\nDiversity % by risk group:")
print(users_df.groupby('echo_chamber_risk')['diversity_pct']
      .agg(['min','mean','max']).round(1))

print("\nSample HIGH risk users (echo chamber):")
high = users_df[users_df['echo_chamber_risk'] == 'HIGH']
print(high[['user_id','dominant_category_name',
            'dominant_pct','diversity_pct',
            'echo_chamber_risk']].head(5).to_string(index=False))

print("\nSample MEDIUM risk users:")
med = users_df[users_df['echo_chamber_risk'] == 'MEDIUM']
print(med[['user_id','dominant_category_name',
           'dominant_pct','diversity_pct',
           'echo_chamber_risk']].head(5).to_string(index=False))

print("\nSample LOW risk users (healthy):")
low = users_df[users_df['echo_chamber_risk'] == 'LOW']
print(low[['user_id','dominant_category_name',
           'dominant_pct','diversity_pct',
           'echo_chamber_risk']].head(5).to_string(index=False))

# ============================================================
# STEP 4A-5: SAVE
# ============================================================

users_df.to_csv(f"{OUTPUT_FOLDER}/echo_chamber_scores.csv", index=False)
print("\necho_chamber_scores.csv saved!")
print("Columns:", users_df.columns.tolist())