# ============================================================
# STEP 4B: Doom Scroll Detector
# Project: Harmful Digital Behavior Detector
# ============================================================

import pandas as pd
import numpy as np
import os

OUTPUT_FOLDER = "output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
print(f"Output folder ready: {OUTPUT_FOLDER}/")

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(f"{OUTPUT_FOLDER}/posts_with_sessions.csv")
sessions_df = pd.read_csv(f"{OUTPUT_FOLDER}/echo_chamber_scores.csv")
print("Posts loaded:", len(df))
print("Sessions loaded:", len(sessions_df))

# ============================================================
# STEP 4B-1: CATEGORY DOOM WEIGHT
# ============================================================

# Each subreddit has a base doom weight reflecting how
# structurally distressing its content tends to be.
# Scale: 0.0 (benign) to 1.0 (highly distressing)
# These are domain-knowledge judgments, not fitted values.

category_doom_weight = {
    0: 0.4,   # Stress          — moderate, often solution-focused
    1: 0.9,   # Depression      — high, heavy emotional content
    2: 0.8,   # Bipolar         — high, intense mood content
    3: 0.7,   # Social Anxiety  — medium-high, isolation themes
    4: 0.6,   # General Anxiety — medium, varies widely
}

df['category_weight'] = df['target'].map(category_doom_weight)

# ============================================================
# STEP 4B-2: NORMALIZE FEATURES TO 0-1 SCALE
# ============================================================

# Three signals, different units — normalize each to 0-1
# using min-max normalization before combining.

def normalize(series):
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return series * 0
    return (series - min_val) / (max_val - min_val)

df['neg_normalized'] = normalize(df['negativity_density'])
df['len_normalized'] = normalize(df['word_count'])
df['cat_normalized'] = normalize(df['category_weight'])

print("\nNormalization done. Sample check (should all be 0-1):")
print("neg_normalized range:", df['neg_normalized'].min().round(3),
      "to", df['neg_normalized'].max().round(3))
print("len_normalized range:", df['len_normalized'].min().round(3),
      "to", df['len_normalized'].max().round(3))

# ============================================================
# STEP 4B-3: WEIGHTED DOOM SCORE PER POST (0-100)
# ============================================================

# Weights reflect signal importance:
#   Negativity  50% — the core "doom" signal
#   Length      30% — longer negative posts = deeper doom-hole
#   Category    20% — base risk of the subreddit

WEIGHT_NEGATIVITY = 0.50
WEIGHT_LENGTH     = 0.30
WEIGHT_CATEGORY   = 0.20

df['doom_score'] = (
    (df['neg_normalized'] * WEIGHT_NEGATIVITY) +
    (df['len_normalized'] * WEIGHT_LENGTH) +
    (df['cat_normalized'] * WEIGHT_CATEGORY)
) * 100

df['doom_score'] = df['doom_score'].round(2)

print("\nDoom score stats (0-100 scale):")
print(df['doom_score'].describe().round(2))

# ============================================================
# STEP 4B-4: POST-LEVEL DOOM RISK LABEL
# ============================================================

# Thresholds are percentile-based, derived from score distribution:
#   Top 15% of posts  (85th percentile) → HIGH
#   Next 30%          (55th percentile) → MEDIUM
#   Bottom 55%                          → LOW

p85 = df['doom_score'].quantile(0.85)
p55 = df['doom_score'].quantile(0.55)

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
      .agg(['mean', 'min', 'max']).round(2))

# ============================================================
# STEP 4B-5: SESSION-LEVEL DOOM SCORE
# ============================================================

# Roll post-level scores up to session level.
# Sessions come from 3.1 — same shuffle, same seed.
# No synthetic construction. Every post used exactly once.

# Session-level thresholds derived from score distribution.
# Consistent with post-level approach — percentile-based, not fixed.
# Calculated after groupby so the distribution is known first.

def assign_doom_risk(df, score_col='avg_doom_score', pct_col='high_doom_pct'):
    sp85_score = df[score_col].quantile(0.85)
    sp55_score = df[score_col].quantile(0.55)
    sp85_pct   = df[pct_col].quantile(0.85)
    sp55_pct   = df[pct_col].quantile(0.55)

    print(f"\nSession thresholds (percentile-based):")
    print(f"  HIGH:   avg_doom > {sp85_score:.2f} OR high_doom_pct > {sp85_pct:.1f}%")
    print(f"  MEDIUM: avg_doom > {sp55_score:.2f} OR high_doom_pct > {sp55_pct:.1f}%")

    def label(row):
        if row[score_col] >= sp85_score or row[pct_col] >= sp85_pct:
            return 'HIGH'
        elif row[score_col] >= sp55_score or row[pct_col] >= sp55_pct:
            return 'MEDIUM'
        else:
            return 'LOW'

    return df.apply(label, axis=1)

session_doom = (df.groupby('session_id')
                .agg(
                    avg_doom_score=('doom_score', 'mean'),
                    high_doom_posts=('post_doom_risk',
                                     lambda x: (x == 'HIGH').sum()),
                    total_posts=('doom_score', 'count'),
                    est_read_minutes=('word_count',
                                      lambda x: round(x.sum() / 200, 1)),
                    total_words_read=('word_count', 'sum')
                )
                .reset_index())

session_doom['avg_doom_score'] = session_doom['avg_doom_score'].round(2)
session_doom['high_doom_pct'] = (
    session_doom['high_doom_posts'] / session_doom['total_posts'] * 100
).round(1)

session_doom['doom_scroll_risk'] = assign_doom_risk(session_doom)

print("\n=== DOOM SCROLL DETECTION RESULTS ===")
print("\nSession doom scroll risk distribution:")
print(session_doom['doom_scroll_risk'].value_counts())

print("\nAvg doom score by risk group:")
print(session_doom.groupby('doom_scroll_risk')['avg_doom_score']
      .agg(['min', 'mean', 'max']).round(2))

print("\nSample HIGH doom scroll sessions:")
high_doom = session_doom[session_doom['doom_scroll_risk'] == 'HIGH']
if len(high_doom) > 0:
    print(high_doom[['session_id', 'avg_doom_score', 'high_doom_pct',
                      'est_read_minutes', 'doom_scroll_risk'
                      ]].head(5).to_string(index=False))
else:
    print("No HIGH risk sessions detected.")

print("\nSample LOW doom scroll sessions:")
low_doom = session_doom[session_doom['doom_scroll_risk'] == 'LOW']
print(low_doom[['session_id', 'avg_doom_score', 'high_doom_pct',
                'est_read_minutes', 'doom_scroll_risk'
                ]].head(5).to_string(index=False))

# ============================================================
# STEP 4B-6: SAVE
# ============================================================

df.to_csv(f"{OUTPUT_FOLDER}/posts_with_doom_scores.csv", index=False)
session_doom.to_csv(f"{OUTPUT_FOLDER}/doom_scroll_scores.csv", index=False)

print("\nposts_with_doom_scores.csv saved!")
print("doom_scroll_scores.csv saved!")
print("\nPost columns:", df.columns.tolist())
print("Session columns:", session_doom.columns.tolist())