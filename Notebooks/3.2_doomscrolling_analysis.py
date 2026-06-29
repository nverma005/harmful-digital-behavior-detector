# ============================================================
# STEP 4B: Doom Scroll Detector
# Project: Harmful Digital Behavior Detector
# ============================================================

# CONCEPT FIRST — what makes a post "doom scroll" content?
#
# Doom scrolling happens when someone compulsively consumes
# a stream of distressing, negative content without stopping.
# Three signals drive it at the post level:
#
# SIGNAL 1 — NEGATIVITY: how distressing is the content?
#            (we already have negativity_density from Step 3)
#
# SIGNAL 2 — LENGTH: longer posts take more time to read.
#            A very long, very negative post is a deep doom-hole.
#
# SIGNAL 3 — CATEGORY WEIGHT: some subreddits are structurally
#            more doom-prone than others. A Depression post is
#            more likely to be doom scroll content than a Stress
#            tip post about time management.
#
# We combine these into a single DOOM SCORE per post (0–100).
# Then we simulate each user's session and flag if they
# consumed too many high-doom posts in one sitting.

import pandas as pd
import numpy as np
import os 

# Create output folder if it doesn't exist
OUTPUT_FOLDER = "output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
# os.makedirs creates the folder automatically
# exist_ok=True means no error if folder already exists
print(f"Output folder ready: {OUTPUT_FOLDER}/")

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(f"{OUTPUT_FOLDER}/clean_data.csv")
users_df = pd.read_csv(f"{OUTPUT_FOLDER}/echo_chamber_scores.csv")
print("Posts loaded:", len(df))
print("Users loaded:", len(users_df))

# ============================================================
# STEP 4B-1: CATEGORY DOOM WEIGHT
# ============================================================

# Each subreddit has a base "doom weight" — how structurally
# negative/distressing that community's content tends to be.
# Scale: 0.0 (benign) to 1.0 (highly distressing)
#
# These weights are domain-knowledge judgments:
# Depression community posts are structurally heavier than
# a general Stress tips post.

category_doom_weight = {
    0: 0.4,   # Stress          — moderate, often solution-focused
    1: 0.9,   # Depression      — high, heavy emotional content
    2: 0.8,   # Bipolar         — high, intense mood content
    3: 0.7,   # Social Anxiety  — medium-high, isolation themes
    4: 0.6,   # General Anxiety — medium, varies widely
}

df['category_weight'] = df['target'].map(category_doom_weight)

# ============================================================
# STEP 4B-2: NORMALIZE FEATURES TO 0–1 SCALE
# ============================================================

# Our three signals are in different units:
#   negativity_density → 0 to 28.57 (per 100 words)
#   word_count         → 0 to ~3000
#   category_weight    → already 0.4 to 0.9
#
# To combine them fairly, we normalize each to 0–1.
# Formula: (value - min) / (max - min)
# This is called "min-max normalization".

def normalize(series):
    """Scale a pandas Series to range 0–1."""
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return series * 0       # avoid division by zero
    return (series - min_val) / (max_val - min_val)

df['neg_normalized']  = normalize(df['negativity_density'])
df['len_normalized']  = normalize(df['word_count'])
# category_weight is already in 0–1 range, just rescale to 0–1
df['cat_normalized']  = normalize(df['category_weight'])

print("\nNormalization done. Sample check (should all be 0–1):")
print("neg_normalized range:", df['neg_normalized'].min().round(3),
      "to", df['neg_normalized'].max().round(3))
print("len_normalized range:", df['len_normalized'].min().round(3),
      "to", df['len_normalized'].max().round(3))

# ============================================================
# STEP 4B-3: WEIGHTED DOOM SCORE PER POST
# ============================================================

# We combine the three normalized signals with weights.
# The weights reflect how much each signal contributes
# to doom scroll risk:
#
#   Negativity  → 50% weight  (most important — it's the "doom" part)
#   Length      → 30% weight  (longer = more time lost scrolling)
#   Category    → 20% weight  (base risk of the subreddit)
#
# Final score is multiplied by 100 so it reads as 0–100.

WEIGHT_NEGATIVITY = 0.50
WEIGHT_LENGTH     = 0.30
WEIGHT_CATEGORY   = 0.20

df['doom_score'] = (
    (df['neg_normalized']  * WEIGHT_NEGATIVITY) +
    (df['len_normalized']  * WEIGHT_LENGTH) +
    (df['cat_normalized']  * WEIGHT_CATEGORY)
) * 100

df['doom_score'] = df['doom_score'].round(2)

print("\nDoom score stats (0–100 scale):")
print(df['doom_score'].describe().round(2))

# ============================================================
# STEP 4B-4: POST-LEVEL DOOM RISK LABEL
# ============================================================

# Thresholds derived from the score distribution:
#   Top ~15% of posts  → HIGH doom content
#   Next ~30%          → MEDIUM doom content
#   Bottom ~55%        → LOW doom content

p85 = df['doom_score'].quantile(0.85)   # top 15% threshold
p55 = df['doom_score'].quantile(0.55)   # top 45% threshold

print(f"\nThresholds — HIGH: >{p85:.2f},  MEDIUM: >{p55:.2f}")

def post_doom_label(score):
    if score >= p85:
        return 'HIGH'
    elif score >= p55:
        return 'MEDIUM'
    else:
        return 'LOW'

df['post_doom_risk'] = df['doom_score'].apply(post_doom_label)

print("\nPost doom risk distribution:")
print(df['post_doom_risk'].value_counts())

print("\nDoom score by category:")
print(df.groupby('category_name')['doom_score']
      .agg(['mean','min','max']).round(2))

# ============================================================
# STEP 4B-5: USER SESSION DOOM SCORE
# ============================================================

# Now we roll up post-level scores to user level.
# We re-simulate each user's session (same seed as Step 4A)
# and calculate their AVERAGE doom score and HIGH doom post ratio.

np.random.seed(42)
NUM_POSTS_PER_USER = 30

session_records = []

for user_id in range(100):
    user_label = f'user_{user_id:03d}'

    # Recreate exact same post selection as Step 4A
    if user_id < 30:
        dominant_category = user_id % 5
        dominant_posts = df[df['target'] == dominant_category].sample(
            n=27, replace=True, random_state=user_id
        )
        other_posts = df[df['target'] != dominant_category].sample(
            n=3, replace=True, random_state=user_id + 1000
        )
        user_posts = pd.concat([dominant_posts, other_posts])

    elif user_id < 50:
        dominant_category = user_id % 5
        dominant_posts = df[df['target'] == dominant_category].sample(
            n=18, replace=True, random_state=user_id
        )
        other_posts = df[df['target'] != dominant_category].sample(
            n=12, replace=True, random_state=user_id + 1000
        )
        user_posts = pd.concat([dominant_posts, other_posts])

    else:
        user_posts = df.sample(
            n=NUM_POSTS_PER_USER, replace=True, random_state=user_id
        )

    # How many of their 30 posts were HIGH doom content?
    high_doom_count = (user_posts['post_doom_risk'] == 'HIGH').sum()
    high_doom_ratio = round(high_doom_count / NUM_POSTS_PER_USER * 100, 1)

    # Average doom score across their session
    avg_doom = round(user_posts['doom_score'].mean(), 2)

    # Total estimated reading time (rough: avg 200 words/minute)
    total_words = user_posts['word_count'].sum()
    est_read_minutes = round(total_words / 200, 1)

    session_records.append({
        'user_id':           user_label,
        'avg_doom_score':    avg_doom,
        'high_doom_posts':   high_doom_count,
        'high_doom_pct':     high_doom_ratio,
        'est_read_minutes':  est_read_minutes,
        'total_words_read':  total_words,
    })

session_df = pd.DataFrame(session_records)

# ============================================================
# STEP 4B-6: USER-LEVEL DOOM SCROLL RISK LABEL
# ============================================================

# A user is flagged for doom scrolling if their session contains
# a high concentration of distressing content.
#
# Rules (based on 30-post session):
#   HIGH   → avg doom score ≥ 35  OR  high doom posts ≥ 35%
#   MEDIUM → avg doom score ≥ 20  OR  high doom posts ≥ 20%
#   LOW    → everything else

def doom_scroll_risk(row):
    if row['avg_doom_score'] >= 35 or row['high_doom_pct'] >= 35:
        return 'HIGH'
    elif row['avg_doom_score'] >= 20 or row['high_doom_pct'] >= 20:
        return 'MEDIUM'
    else:
        return 'LOW'

session_df['doom_scroll_risk'] = session_df.apply(doom_scroll_risk, axis=1)

print("\n=== DOOM SCROLL DETECTION RESULTS ===")
print("\nUser doom scroll risk distribution:")
print(session_df['doom_scroll_risk'].value_counts())

print("\nAvg doom score by risk group:")
print(session_df.groupby('doom_scroll_risk')['avg_doom_score']
      .agg(['min','mean','max']).round(2))

print("\nSample HIGH doom scroll users:")
high_doom = session_df[session_df['doom_scroll_risk'] == 'HIGH']
print(high_doom[['user_id','avg_doom_score','high_doom_pct',
                  'est_read_minutes','doom_scroll_risk'
                  ]].head(5).to_string(index=False))

print("\nSample LOW doom scroll users:")
low_doom = session_df[session_df['doom_scroll_risk'] == 'LOW']
print(low_doom[['user_id','avg_doom_score','high_doom_pct',
                 'est_read_minutes','doom_scroll_risk'
                 ]].head(5).to_string(index=False))

# ============================================================
# STEP 4B-7: SAVE BOTH OUTPUTS
# ============================================================

df.to_csv(f"{OUTPUT_FOLDER}/posts_with_doom_scores.csv", index=False)
session_df.to_csv(f"{OUTPUT_FOLDER}/doom_scroll_scores.csv", index=False)

print("\nposts_with_doom_scores.csv saved!")
print("doom_scroll_scores.csv saved!")
print("\nPost columns:", df.columns.tolist())
print("Session columns:", session_df.columns.tolist())