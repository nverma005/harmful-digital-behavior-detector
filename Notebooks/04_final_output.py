# ============================================================
# STEP 5: Combined Scoring Engine & Final Risk Report
# Project: Harmful Digital Behavior Detector
# ============================================================

# WHAT THIS STEP DOES:
# We now have two separate risk scores per user:
#   - Echo Chamber risk  (from step 4A)
#   - Doom Scroll risk   (from step 4B)
#
# This step:
#   1. Merges both into one master user table
#   2. Creates a COMBINED RISK SCORE (0–100)
#   3. Assigns a final OVERALL RISK LABEL per user
#   4. Adds actionable insight flags
#   5. Exports the final CSV that Power BI will read

import pandas as pd
import numpy as np

# ============================================================
# OUTPUT FOLDER SETUP
# ============================================================

import os

# Create output folder if it doesn't exist
OUTPUT_FOLDER = "output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
# os.makedirs creates the folder automatically
# exist_ok=True means no error if folder already exists
print(f"Output folder ready: {OUTPUT_FOLDER}/")

# ============================================================
# LOAD BOTH SCORE FILES
# ============================================================

echo_df   = pd.read_csv(f"{OUTPUT_FOLDER}/echo_chamber_scores.csv")
doom_df   = pd.read_csv(f"{OUTPUT_FOLDER}/doom_scroll_scores.csv")

print("Echo chamber users:", len(echo_df))
print("Doom scroll users: ", len(doom_df))

# ============================================================
# STEP 5-1: MERGE ON user_id
# ============================================================

# pd.merge() joins two DataFrames on a common column.
# how='inner' keeps only rows that appear in BOTH files.
# Since both have 100 users with the same IDs, all 100 merge.

master_df = pd.merge(echo_df, doom_df, on='user_id', how='inner')
print("\nAfter merge:", master_df.shape)
print("Columns:", master_df.columns.tolist())

# ============================================================
# STEP 5-2: CONVERT RISK LABELS TO NUMERIC SCORES
# ============================================================

# To calculate a combined score, we need numbers not words.
# We map: HIGH=3, MEDIUM=2, LOW=1
# This is called "ordinal encoding"

risk_to_num = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3}

master_df['echo_risk_num'] = master_df['echo_chamber_risk'].map(risk_to_num)
master_df['doom_risk_num'] = master_df['doom_scroll_risk'].map(risk_to_num)

# ============================================================
# STEP 5-3: COMBINED RISK SCORE (0–100)
# ============================================================

# We weight the two risks:
#   Echo Chamber → 40% of final score
#   Doom Scroll  → 60% of final score
#
# Why doom scroll weighted higher?
# Doom scrolling has a more immediate mental health impact.
# Echo chambers are slower-building but also harmful.
#
# We also incorporate the raw numeric signals:
#   - diversity_pct     (lower = worse, so we invert it)
#   - avg_doom_score    (higher = worse)
#   - high_doom_pct     (higher = worse)

# Invert diversity: 100% diverse → 0 risk, 0% diverse → 100 risk
master_df['echo_risk_raw'] = 100 - master_df['diversity_pct']

# Normalize avg_doom_score to 0–100 scale
doom_min = master_df['avg_doom_score'].min()
doom_max = master_df['avg_doom_score'].max()
master_df['doom_risk_raw'] = (
    (master_df['avg_doom_score'] - doom_min) /
    (doom_max - doom_min) * 100
).round(2)

# Weighted combination
ECHO_WEIGHT = 0.40
DOOM_WEIGHT = 0.60

master_df['combined_risk_score'] = (
    (master_df['echo_risk_raw']  * ECHO_WEIGHT) +
    (master_df['doom_risk_raw']  * DOOM_WEIGHT)
).round(2)

print("\nCombined risk score stats:")
print(master_df['combined_risk_score'].describe().round(2))

# ============================================================
# STEP 5-4: OVERALL RISK LABEL
# ============================================================

# Use percentile-based thresholds so labels are always meaningful
# regardless of the actual score range.
# Top 20% = HIGH, Next 30% = MEDIUM, Bottom 50% = LOW

p80 = master_df['combined_risk_score'].quantile(0.80)
p50 = master_df['combined_risk_score'].quantile(0.50)

print(f"\nOverall risk thresholds:")
print(f"  HIGH   : score > {p80:.2f}")
print(f"  MEDIUM : score > {p50:.2f}")
print(f"  LOW    : score ≤ {p50:.2f}")

def overall_risk(score):
    if score > p80:
        return 'HIGH'
    elif score > p50:
        return 'MEDIUM'
    else:
        return 'LOW'

master_df['overall_risk'] = master_df['combined_risk_score'].apply(overall_risk)

print("\nOverall risk distribution:")
print(master_df['overall_risk'].value_counts())

# ============================================================
# STEP 5-5: INSIGHT FLAGS
# ============================================================

# These are boolean flags (0 or 1) that give specific,
# actionable signals. Power BI can filter on these directly.

# Flag 1: Pure echo chamber — reads >80% from one category
master_df['flag_pure_bubble'] = (
    master_df['dominant_pct'] >= 80
).astype(int)

# Flag 2: Heavy doom consumer — >40% of posts are HIGH doom
master_df['flag_heavy_doom'] = (
    master_df['high_doom_pct'] >= 40
).astype(int)

# Flag 3: Double danger — HIGH in BOTH echo chamber AND doom scroll
master_df['flag_double_risk'] = (
    (master_df['echo_chamber_risk'] == 'HIGH') &
    (master_df['doom_scroll_risk']  == 'HIGH')
).astype(int)

# Flag 4: Long session — estimated reading > 60 minutes
master_df['flag_long_session'] = (
    master_df['est_read_minutes'] >= 60
).astype(int)

# Flag 5: High negativity consumption
master_df['flag_high_negativity'] = (
    master_df['avg_negativity'] >= 2.0
).astype(int)

print("\nInsight flag summary:")
flag_cols = ['flag_pure_bubble','flag_heavy_doom',
             'flag_double_risk','flag_long_session',
             'flag_high_negativity']
print(master_df[flag_cols].sum().to_string())

# ============================================================
# STEP 5-6: RECOMMENDATION TEXT
# ============================================================

# Each user gets a plain-English recommendation.
# This will appear as a tooltip/label in Power BI.

def get_recommendation(row):
    if row['flag_double_risk'] == 1:
        return "Critical: User is in a content bubble AND consuming high-distress content. Immediate intervention recommended."
    elif row['flag_pure_bubble'] == 1 and row['flag_heavy_doom'] == 0:
        return "Echo chamber detected: Diversify content sources. User reads mostly " + row['dominant_category_name'] + " content."
    elif row['flag_heavy_doom'] == 1 and row['flag_pure_bubble'] == 0:
        return "Doom scroll risk: User consumes high-negativity content frequently. Consider content filters."
    elif row['overall_risk'] == 'MEDIUM':
        return "Moderate risk: Monitor browsing patterns. Mild content bias detected."
    else:
        return "Healthy browsing patterns detected. No intervention needed."

master_df['recommendation'] = master_df.apply(get_recommendation, axis=1)

# ============================================================
# STEP 5-7: SELECT FINAL COLUMNS FOR EXPORT
# ============================================================

# We choose only the columns Power BI needs.
# Clean, purposeful, no redundant columns.

final_columns = [
    # Identity
    'user_id',

    # Echo chamber signals
    'diversity_score',
    'diversity_pct',
    'echo_chamber_risk',
    'dominant_category_name',
    'dominant_pct',
    'count_stress',
    'count_depression',
    'count_bipolar',
    'count_social_anxiety',
    'count_general_anxiety',

    # Doom scroll signals
    'avg_doom_score',
    'high_doom_pct',
    'est_read_minutes',
    'total_words_read',
    'avg_negativity',
    'doom_scroll_risk',

    # Combined score
    'echo_risk_raw',
    'doom_risk_raw',
    'combined_risk_score',
    'overall_risk',

    # Flags
    'flag_pure_bubble',
    'flag_heavy_doom',
    'flag_double_risk',
    'flag_long_session',
    'flag_high_negativity',

    # Recommendation
    'recommendation',
]

final_df = master_df[final_columns].copy()

# ============================================================
# STEP 5-8: FINAL SUMMARY REPORT
# ============================================================

print("\n" + "="*50)
print("       FINAL RISK REPORT SUMMARY")
print("="*50)

print(f"\nTotal users analysed : {len(final_df)}")
print(f"HIGH overall risk    : {(final_df['overall_risk']=='HIGH').sum()}")
print(f"MEDIUM overall risk  : {(final_df['overall_risk']=='MEDIUM').sum()}")
print(f"LOW overall risk     : {(final_df['overall_risk']=='LOW').sum()}")

print(f"\nUsers in pure bubble         : {final_df['flag_pure_bubble'].sum()}")
print(f"Users with heavy doom scroll : {final_df['flag_heavy_doom'].sum()}")
print(f"Users with DOUBLE risk       : {final_df['flag_double_risk'].sum()}")
print(f"Users with long sessions     : {final_df['flag_long_session'].sum()}")

print("\nTop 5 highest-risk users:")
top5 = final_df.nlargest(5, 'combined_risk_score')
print(top5[['user_id','combined_risk_score','overall_risk',
            'echo_chamber_risk','doom_scroll_risk']].to_string(index=False))

print("\nBottom 5 lowest-risk users:")
bot5 = final_df.nsmallest(5, 'combined_risk_score')
print(bot5[['user_id','combined_risk_score','overall_risk',
            'echo_chamber_risk','doom_scroll_risk']].to_string(index=False))

print("\nSample recommendations:")
for _, row in final_df.head(4).iterrows():
    print(f"  {row['user_id']}: {row['recommendation']}")

# ============================================================
# STEP 5-9: SAVE FINAL FILE
# ============================================================

final_df.to_csv(f"{OUTPUT_FOLDER}/final_risk_report.csv", index=False)
print(f"\nfinal_risk_report.csv saved!")
print(f"Rows: {len(final_df)}  |  Columns: {len(final_df.columns)}")

# ============================================================
# STEP 5-10: AUTO-GENERATE ALL POWER BI READY CSV FILES
# ============================================================

# --- FILE 1: Flag Summary (for bar chart in Power BI) ---
flag_summary = pd.DataFrame({
    'Flag': [
        'Pure Bubble',
        'Heavy Doom',
        'Double Risk',
        'High Negativity',
        'Long Session'
    ],
    'Count': [
        int(final_df['flag_pure_bubble'].sum()),
        int(final_df['flag_heavy_doom'].sum()),
        int(final_df['flag_double_risk'].sum()),
        int(final_df['flag_high_negativity'].sum()),
        int(final_df['flag_long_session'].sum())
    ],
    'Color_Hex': [
        '#F0A500',   # amber  — pure bubble
        '#E84040',   # red    — heavy doom
        '#8B0000',   # dark red — double risk
        '#F0A500',   # amber  — high negativity
        '#3498DB'    # blue   — long session
    ]
})
flag_summary.to_csv(f"{OUTPUT_FOLDER}/flag_summary.csv", index=False)
print("flag_summary.csv saved!")

# --- FILE 2: Risk Summary Table (for KPI cards in Power BI) ---
risk_summary = pd.DataFrame({
    'Metric': [
        'Total Users',
        'HIGH Risk Users',
        'MEDIUM Risk Users',
        'LOW Risk Users',
        'Avg Risk Score',
        'Double Risk Users',
        'In Echo Bubble',
        'Heavy Doom Users',
        'High Negativity Users',
        'Long Session Users'
    ],
    'Value': [
        len(final_df),
        int((final_df['overall_risk'] == 'HIGH').sum()),
        int((final_df['overall_risk'] == 'MEDIUM').sum()),
        int((final_df['overall_risk'] == 'LOW').sum()),
        round(final_df['combined_risk_score'].mean(), 2),
        int(final_df['flag_double_risk'].sum()),
        int(final_df['flag_pure_bubble'].sum()),
        int(final_df['flag_heavy_doom'].sum()),
        int(final_df['flag_high_negativity'].sum()),
        int(final_df['flag_long_session'].sum())
    ]
})
risk_summary.to_csv(f"{OUTPUT_FOLDER}/risk_summary.csv", index=False)
print("risk_summary.csv saved!")

# --- FILE 3: Category Distribution per Risk Group ---
# Power BI Page 2 — Echo Chamber stacked bar chart
category_risk = final_df.groupby('echo_chamber_risk').agg(
    Stress           = ('count_stress',          'mean'),
    Depression       = ('count_depression',       'mean'),
    Bipolar          = ('count_bipolar',          'mean'),
    Social_Anxiety   = ('count_social_anxiety',   'mean'),
    General_Anxiety  = ('count_general_anxiety',  'mean'),
    User_Count       = ('user_id',                'count')
).round(1).reset_index()
category_risk.to_csv(f"{OUTPUT_FOLDER}/category_risk_distribution.csv", index=False)
print("category_risk_distribution.csv saved!")

# --- FILE 4: Doom Score by Category ---
# Power BI Page 3 — Doom scroll box plot alternative
posts_df = pd.read_csv(f"{OUTPUT_FOLDER}/posts_with_doom_scores.csv")
doom_by_category = posts_df.groupby('category_name').agg(
    Avg_Doom_Score   = ('doom_score', 'mean'),
    Min_Doom_Score   = ('doom_score', 'min'),
    Max_Doom_Score   = ('doom_score', 'max'),
    Median_Doom      = ('doom_score', 'median'),
    HIGH_Risk_Posts  = ('post_doom_risk',
                        lambda x: (x == 'HIGH').sum()),
    MEDIUM_Risk_Posts= ('post_doom_risk',
                        lambda x: (x == 'MEDIUM').sum()),
    LOW_Risk_Posts   = ('post_doom_risk',
                        lambda x: (x == 'LOW').sum()),
    Total_Posts      = ('doom_score', 'count')
).round(2).reset_index()
doom_by_category.to_csv(f"{OUTPUT_FOLDER}/doom_by_category.csv", index=False)
print("doom_by_category.csv saved!")

# --- FILE 5: User Risk Full Detail ---
# Power BI — combined detail table with all signals
user_detail = final_df[[
    'user_id', 'overall_risk', 'combined_risk_score',
    'echo_chamber_risk', 'diversity_pct', 'dominant_category_name',
    'dominant_pct', 'doom_scroll_risk', 'avg_doom_score',
    'high_doom_pct', 'est_read_minutes', 'avg_negativity',
    'flag_pure_bubble', 'flag_heavy_doom', 'flag_double_risk',
    'flag_long_session', 'flag_high_negativity', 'recommendation'
]].copy()

# Add readable risk rank number (1=lowest risk, 100=highest)
user_detail = user_detail.sort_values(
    'combined_risk_score', ascending=False
).reset_index(drop=True)
user_detail['risk_rank'] = user_detail.index + 1

user_detail.to_csv(f"{OUTPUT_FOLDER}/user_risk_detail.csv", index=False)
print("user_risk_detail.csv saved!")

# --- FILE 6: Echo Chamber Detail per User ---
# Power BI Page 2 scatter plot data
echo_detail = final_df[[
    'user_id', 'echo_chamber_risk', 'diversity_pct',
    'diversity_score', 'dominant_category_name', 'dominant_pct',
    'avg_doom_score', 'avg_negativity',
    'count_stress', 'count_depression', 'count_bipolar',
    'count_social_anxiety', 'count_general_anxiety'
]].copy()
echo_detail.to_csv(f"{OUTPUT_FOLDER}/echo_chamber_detail.csv", index=False)
print("echo_chamber_detail.csv saved!")

# --- FILE 7: Post Level Sample for Power BI ---
# Use a clean sample of posts for the doom scroll page
# Full 5957 rows is too heavy — 1000 representative sample
posts_sample = posts_df[[
    'title', 'category_name', 'word_count',
    'negative_word_count', 'negativity_density',
    'doom_score', 'post_doom_risk', 'is_long_post'
]].copy()

# Sample 1000 rows proportionally from each category
posts_sample = posts_sample.groupby(
    'category_name', group_keys=False
).apply(lambda x: x.sample(
    min(200, len(x)), random_state=42
)).reset_index(drop=True)

posts_sample.to_csv(f"{OUTPUT_FOLDER}/posts_sample_powerbi.csv", index=False)
print("posts_sample_powerbi.csv saved!")

# ============================================================
# FINAL SUMMARY OF ALL FILES GENERATED
# ============================================================

print("\n" + "="*55)
print("   ALL POWER BI FILES GENERATED SUCCESSFULLY")
print("="*55)
print("""
FILE                          ROWS    USED FOR
─────────────────────────────────────────────────────
final_risk_report.csv          100    Master user table
flag_summary.csv                 5    Page 1 bar chart
risk_summary.csv                10    Page 1 KPI cards
category_risk_distribution.csv   3    Page 2 stacked bar
doom_by_category.csv             5    Page 3 category chart
user_risk_detail.csv           100    All pages detail table
echo_chamber_detail.csv        100    Page 2 scatter plot
posts_sample_powerbi.csv      1000    Page 3 doom posts
""")