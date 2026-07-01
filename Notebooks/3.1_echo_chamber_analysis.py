# ============================================================
# STEP 4A: Echo Chamber Detector — Category Content Analysis
# Project: Harmful Digital Behavior Detector
# ============================================================

# APPROACH — why vocabulary overlap, not session diversity?
#
# This dataset has no user IDs. Grouping posts into sessions
# and measuring diversity produces trivially balanced results
# because the categories themselves are evenly distributed.
# That is not detection — it is arithmetic.
#
# Instead, we measure echo chambers two ways:
#
# MACRO: How linguistically isolated are the 5 subreddits
#        from each other? Low vocabulary overlap between
#        communities = strong echo chamber boundaries.
#        High overlap = content bleeds across communities.
#        Metric: Jaccard similarity on top-200 words per category.
#
# MICRO: How deeply embedded is each individual post in its
#        own category's unique language? A post that uses
#        mostly words exclusive to one subreddit is a stronger
#        echo chamber signal than one using shared vocabulary.
#        Metric: exclusivity score = % of post words that are
#        unique to that category's top vocabulary.
#
# Both analyses use all 5,957 posts. No synthetic construction.

import pandas as pd
import numpy as np
import os
from collections import Counter
from itertools import combinations

OUTPUT_FOLDER = "output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
print(f"Output folder ready: {OUTPUT_FOLDER}/")

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(f"{OUTPUT_FOLDER}/clean_data.csv")
print("Loaded clean data. Shape:", df.shape)

category_map = {
    0: 'Stress',
    1: 'Depression',
    2: 'Bipolar',
    3: 'Social Anxiety',
    4: 'General Anxiety'
}

# ============================================================
# STEP 4A-1: CREATE CONTENT SESSIONS
# ============================================================

# Sessions are used by 3.2 (doom scroll) and 04 (combined risk).
# Shuffle once with fixed seed — same shuffle every run.
# 3.2 loads posts_with_sessions.csv directly — do not reshuffle there.

POSTS_PER_SESSION = 30
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
NUM_SESSIONS = len(df_shuffled) // POSTS_PER_SESSION
df_shuffled = df_shuffled.iloc[:NUM_SESSIONS * POSTS_PER_SESSION].copy()

df_shuffled['session_id'] = [
    f'session_{i // POSTS_PER_SESSION:03d}'
    for i in range(len(df_shuffled))
]

print(f"\nTotal posts:           {len(df)}")
print(f"Posts used:            {len(df_shuffled)}")
print(f"Posts dropped:         {len(df) - len(df_shuffled)}")
print(f"Sessions created:      {NUM_SESSIONS}")

# ============================================================
# STEP 4A-2: CATEGORY VOCABULARY PROFILES
# ============================================================

# Build top-200 word vocabulary per category.
# These are the words that define each subreddit's language.

TOP_N = 200

category_vocab = {}
for cat_id, cat_name in category_map.items():
    cat_text = df[df['target'] == cat_id]['clean_text'].dropna()
    all_words = ' '.join(cat_text).split()
    word_freq = Counter(all_words)
    top_words = set(w for w, _ in word_freq.most_common(TOP_N))
    category_vocab[cat_name] = top_words
    print(f"  {cat_name}: {len(top_words)} unique top words")

# ============================================================
# STEP 4A-3: MACRO — INTER-CATEGORY JACCARD SIMILARITY
# ============================================================

# Jaccard similarity = shared words / total unique words (both sets)
# Range: 0.0 (no overlap, strong boundaries) to 1.0 (identical vocab)
#
# Echo boundary strength:
#   Jaccard < 0.15  → STRONG boundary (distinct communities)
#   Jaccard < 0.30  → MODERATE boundary
#   Jaccard >= 0.30 → WEAK boundary (content bleeds across)

jaccard_records = []
for cat_a, cat_b in combinations(category_map.values(), 2):
    vocab_a = category_vocab[cat_a]
    vocab_b = category_vocab[cat_b]
    intersection = len(vocab_a & vocab_b)
    union = len(vocab_a | vocab_b)
    jaccard = round(intersection / union, 4) if union > 0 else 0.0

    jaccard_records.append({
        'category_a':         cat_a,
        'category_b':         cat_b,
        'shared_words':       intersection,
        'total_unique_words': union,
        'jaccard_similarity': jaccard,
        'echo_boundary':      (
            'STRONG' if jaccard < 0.15 else
            'MODERATE' if jaccard < 0.30 else
            'WEAK'
        )
    })

jaccard_df = (pd.DataFrame(jaccard_records)
              .sort_values('jaccard_similarity')
              .reset_index(drop=True))

print("\n=== MACRO: INTER-CATEGORY VOCABULARY OVERLAP ===")
print(jaccard_df[['category_a', 'category_b',
                   'jaccard_similarity', 'echo_boundary']]
      .to_string(index=False))

print("\nEcho boundary distribution:")
print(jaccard_df['echo_boundary'].value_counts())

# ============================================================
# STEP 4A-4: MICRO — POST-LEVEL EXCLUSIVITY SCORE
# ============================================================

# Exclusive words = words in one category's top vocabulary
# that do not appear in any other category's top vocabulary.
# A high exclusivity score means the post uses language
# deeply rooted in one community — a stronger echo signal.

exclusive_vocab = {}
for cat_name, vocab in category_vocab.items():
    other_words = set()
    for other_name, other_vocab in category_vocab.items():
        if other_name != cat_name:
            other_words |= other_vocab
    exclusive_vocab[cat_name] = vocab - other_words

print("\nExclusive vocabulary sizes (words unique to each category):")
for cat_name, words in exclusive_vocab.items():
    print(f"  {cat_name}: {len(words)} exclusive words")

def post_exclusivity(row):
    if pd.isna(row.get('clean_text')):
        return 0.0
    cat_name = category_map.get(row['target'])
    if not cat_name:
        return 0.0
    words = set(str(row['clean_text']).split())
    if not words:
        return 0.0
    exclusive = exclusive_vocab.get(cat_name, set())
    return round(len(words & exclusive) / len(words) * 100, 2)

df['exclusivity_score'] = df.apply(post_exclusivity, axis=1)

print("\nExclusivity score stats (0–100):")
print(df['exclusivity_score'].describe().round(2))

# ============================================================
# STEP 4A-5: POST-LEVEL ECHO CHAMBER RISK LABEL
# ============================================================

# Thresholds derived from score distribution:
#   Top 15% of posts by exclusivity → HIGH echo signal
#   Next 30%                        → MEDIUM
#   Bottom 55%                      → LOW

p85 = df['exclusivity_score'].quantile(0.85)
p55 = df['exclusivity_score'].quantile(0.55)

print(f"\nThresholds — HIGH: >{p85:.2f}, MEDIUM: >{p55:.2f}")

def echo_risk_label(score):
    if score >= p85:
        return 'HIGH'
    elif score >= p55:
        return 'MEDIUM'
    else:
        return 'LOW'

df['echo_chamber_risk'] = df['exclusivity_score'].apply(echo_risk_label)

print("\nPost echo chamber risk distribution:")
print(df['echo_chamber_risk'].value_counts())

print("\nExclusivity score by category:")
print(df.groupby('category_name')['exclusivity_score']
      .agg(['mean', 'min', 'max']).round(2))

# ============================================================
# STEP 4A-6: SESSION-LEVEL AGGREGATION
# ============================================================

# Roll post-level scores up to session level.
# This feeds into 04_final_output.py for combined risk scoring.

session_echo = (df_shuffled
                .merge(df[['exclusivity_score', 'echo_chamber_risk']],
                       left_index=True, right_index=True, how='left')
                .groupby('session_id')
                .agg(
                    avg_exclusivity=('exclusivity_score', 'mean'),
                    high_echo_posts=('echo_chamber_risk',
                                     lambda x: (x == 'HIGH').sum()),
                    total_posts=('exclusivity_score', 'count')
                )
                .reset_index())

session_echo['high_echo_pct'] = (
    session_echo['high_echo_posts'] / session_echo['total_posts'] * 100
).round(1)

session_echo['avg_exclusivity'] = session_echo['avg_exclusivity'].round(2)

# Session echo risk: same percentile logic
sp85 = session_echo['avg_exclusivity'].quantile(0.85)
sp55 = session_echo['avg_exclusivity'].quantile(0.55)

session_echo['session_echo_risk'] = session_echo['avg_exclusivity'].apply(
    lambda x: 'HIGH' if x >= sp85 else 'MEDIUM' if x >= sp55 else 'LOW'
)

print("\n=== SESSION-LEVEL ECHO CHAMBER RESULTS ===")
print(session_echo['session_echo_risk'].value_counts())

print("\nSample HIGH echo sessions:")
high_s = session_echo[session_echo['session_echo_risk'] == 'HIGH']
if len(high_s) > 0:
    print(high_s[['session_id', 'avg_exclusivity',
                   'high_echo_pct', 'session_echo_risk']]
          .head(5).to_string(index=False))

# ============================================================
# STEP 4A-7: SAVE ALL OUTPUTS
# ============================================================

# Shared session file for 3.2 and 04
df_shuffled_with_scores = df_shuffled.merge(
    df[['exclusivity_score', 'echo_chamber_risk']],
    left_index=True, right_index=True, how='left'
)
df_shuffled_with_scores.to_csv(
    f"{OUTPUT_FOLDER}/posts_with_sessions.csv", index=False
)

# Category-level Jaccard matrix
jaccard_df.to_csv(f"{OUTPUT_FOLDER}/jaccard_similarity.csv", index=False)

# Post-level scores
df.to_csv(f"{OUTPUT_FOLDER}/posts_with_echo_scores.csv", index=False)

# Session-level echo scores
session_echo.to_csv(
    f"{OUTPUT_FOLDER}/echo_chamber_scores.csv", index=False
)

print("\nAll files saved:")
print("  posts_with_sessions.csv")
print("  jaccard_similarity.csv")
print("  posts_with_echo_scores.csv")
print("  echo_chamber_scores.csv")
print("\nSession columns:", session_echo.columns.tolist())