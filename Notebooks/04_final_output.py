# ============================================================
# STEP 5: Combined Scoring Engine & Final Risk Report
# Project: Harmful Digital Behavior Detector
# ============================================================

import pandas as pd
import numpy as np
import os

OUTPUT_FOLDER = "output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
print(f"Output folder ready: {OUTPUT_FOLDER}/")

# ============================================================
# LOAD BOTH SCORE FILES
# ============================================================

echo_df = pd.read_csv(f"{OUTPUT_FOLDER}/echo_chamber_scores.csv")
doom_df = pd.read_csv(f"{OUTPUT_FOLDER}/doom_scroll_scores.csv")

print("Echo sessions:", len(echo_df))
print("Doom sessions:", len(doom_df))
print("\nEcho columns:", echo_df.columns.tolist())
print("Doom columns: ", doom_df.columns.tolist())

# ============================================================
# STEP 5-1: MERGE ON session_id
# ============================================================

master_df = pd.merge(echo_df, doom_df, on='session_id', how='inner',
                     suffixes=('_echo', '_doom'))
print("\nAfter merge:", master_df.shape)

# ============================================================
# STEP 5-2: COMBINED RISK SCORE (0-100)
# ============================================================

# Echo risk: normalize avg_exclusivity to 0-100
# Higher exclusivity = more echo chamber risk
exc_min = master_df['avg_exclusivity'].min()
exc_max = master_df['avg_exclusivity'].max()
master_df['echo_risk_raw'] = (
    (master_df['avg_exclusivity'] - exc_min) /
    (exc_max - exc_min) * 100
).round(2)

# Doom risk: normalize avg_doom_score to 0-100
doom_min = master_df['avg_doom_score'].min()
doom_max = master_df['avg_doom_score'].max()
master_df['doom_risk_raw'] = (
    (master_df['avg_doom_score'] - doom_min) /
    (doom_max - doom_min) * 100
).round(2)

# Weighted combination
# Doom scroll weighted higher — more immediate mental health impact
ECHO_WEIGHT = 0.40
DOOM_WEIGHT = 0.60

master_df['combined_risk_score'] = (
    (master_df['echo_risk_raw'] * ECHO_WEIGHT) +
    (master_df['doom_risk_raw'] * DOOM_WEIGHT)
).round(2)

print("\nCombined risk score stats:")
print(master_df['combined_risk_score'].describe().round(2))

# ============================================================
# STEP 5-3: OVERALL RISK LABEL
# ============================================================

# Percentile-based so labels are always meaningful
# Top 20% = HIGH, next 30% = MEDIUM, bottom 50% = LOW

p80 = master_df['combined_risk_score'].quantile(0.80)
p50 = master_df['combined_risk_score'].quantile(0.50)

print(f"\nOverall risk thresholds:")
print(f"  HIGH   : score > {p80:.2f}")
print(f"  MEDIUM : score > {p50:.2f}")
print(f"  LOW    : score <= {p50:.2f}")

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
# STEP 5-4: INSIGHT FLAGS
# ============================================================

# Flag 1: Strong echo signal — high % of posts use category-exclusive words
master_df['flag_pure_bubble'] = (
    master_df['high_echo_pct'] >= 30
).astype(int)

# Flag 2: Heavy doom consumer — >30% of posts are HIGH doom
master_df['flag_heavy_doom'] = (
    master_df['high_doom_pct'] >= 30
).astype(int)

# Flag 3: Double danger — HIGH in BOTH behaviors
master_df['flag_double_risk'] = (
    (master_df['session_echo_risk'] == 'HIGH') &
    (master_df['doom_scroll_risk']  == 'HIGH')
).astype(int)

# Flag 4: Long session — estimated reading > 45 minutes
master_df['flag_long_session'] = (
    master_df['est_read_minutes'] >= 45
).astype(int)

# Flag 5: High doom concentration — avg doom in top 25%
doom_75 = master_df['avg_doom_score'].quantile(0.75)
master_df['flag_high_doom_concentration'] = (
    master_df['avg_doom_score'] >= doom_75
).astype(int)

print("\nInsight flag summary:")
flag_cols = [
    'flag_pure_bubble', 'flag_heavy_doom', 'flag_double_risk',
    'flag_long_session', 'flag_high_doom_concentration'
]
print(master_df[flag_cols].sum().to_string())

# ============================================================
# STEP 5-5: RECOMMENDATION TEXT
# ============================================================

def get_recommendation(row):
    if row['flag_double_risk'] == 1:
        return "Critical: Session shows both echo chamber and high-distress content patterns."
    elif row['flag_pure_bubble'] == 1 and row['flag_heavy_doom'] == 0:
        return "Echo chamber risk: Session vocabulary heavily concentrated in one community."
    elif row['flag_heavy_doom'] == 1 and row['flag_pure_bubble'] == 0:
        return "Doom scroll risk: High proportion of distressing content in this session."
    elif row['overall_risk'] == 'MEDIUM':
        return "Moderate risk: Mild content concentration detected. Worth monitoring."
    else:
        return "Low risk: Session shows balanced, diverse content consumption."

master_df['recommendation'] = master_df.apply(get_recommendation, axis=1)

# ============================================================
# STEP 5-6: SELECT FINAL COLUMNS FOR EXPORT
# ============================================================

final_columns = [
    # Identity
    'session_id',

    # Echo chamber signals
    'avg_exclusivity',
    'high_echo_pct',
    'session_echo_risk',

    # Doom scroll signals
    'avg_doom_score',
    'high_doom_pct',
    'est_read_minutes',
    'total_words_read',
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
    'flag_high_doom_concentration',

    # Recommendation
    'recommendation',
]

final_df = master_df[final_columns].copy()

# ============================================================
# STEP 5-7: FINAL SUMMARY REPORT
# ============================================================

print("\n" + "="*50)
print("       FINAL RISK REPORT SUMMARY")
print("="*50)

print(f"\nTotal sessions analysed : {len(final_df)}")
print(f"HIGH overall risk       : {(final_df['overall_risk']=='HIGH').sum()}")
print(f"MEDIUM overall risk     : {(final_df['overall_risk']=='MEDIUM').sum()}")
print(f"LOW overall risk        : {(final_df['overall_risk']=='LOW').sum()}")
print(f"\nSessions with echo bubble    : {final_df['flag_pure_bubble'].sum()}")
print(f"Sessions with heavy doom     : {final_df['flag_heavy_doom'].sum()}")
print(f"Sessions with DOUBLE risk    : {final_df['flag_double_risk'].sum()}")
print(f"Sessions with long reading   : {final_df['flag_long_session'].sum()}")

print("\nTop 5 highest-risk sessions:")
top5 = final_df.nlargest(5, 'combined_risk_score')
print(top5[['session_id', 'combined_risk_score', 'overall_risk',
            'session_echo_risk', 'doom_scroll_risk']].to_string(index=False))

print("\nBottom 5 lowest-risk sessions:")
bot5 = final_df.nsmallest(5, 'combined_risk_score')
print(bot5[['session_id', 'combined_risk_score', 'overall_risk',
            'session_echo_risk', 'doom_scroll_risk']].to_string(index=False))

# ============================================================
# STEP 5-8: SAVE FINAL FILE
# ============================================================

final_df.to_csv(f"{OUTPUT_FOLDER}/final_risk_report.csv", index=False)
print(f"\nfinal_risk_report.csv saved!")
print(f"Rows: {len(final_df)}  |  Columns: {len(final_df.columns)}")

# ============================================================
# STEP 5-9: POWER BI READY CSV FILES
# ============================================================

# FILE 1: Flag Summary
flag_summary = pd.DataFrame({
    'Flag': [
        'Echo Bubble',
        'Heavy Doom',
        'Double Risk',
        'Long Session',
        'High Doom Concentration'
    ],
    'Count': [
        int(final_df['flag_pure_bubble'].sum()),
        int(final_df['flag_heavy_doom'].sum()),
        int(final_df['flag_double_risk'].sum()),
        int(final_df['flag_long_session'].sum()),
        int(final_df['flag_high_doom_concentration'].sum())
    ],
    'Color_Hex': [
        '#F0A500',
        '#E84040',
        '#8B0000',
        '#3498DB',
        '#E84040'
    ]
})
flag_summary.to_csv(f"{OUTPUT_FOLDER}/flag_summary.csv", index=False)
print("flag_summary.csv saved!")

# FILE 2: Risk Summary (KPI cards)
risk_summary = pd.DataFrame({
    'Metric': [
        'Total Sessions',
        'HIGH Risk Sessions',
        'MEDIUM Risk Sessions',
        'LOW Risk Sessions',
        'Avg Combined Risk Score',
        'Double Risk Sessions',
        'Echo Bubble Sessions',
        'Heavy Doom Sessions',
        'Long Reading Sessions',
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
        int(final_df['flag_long_session'].sum()),
    ]
})
risk_summary.to_csv(f"{OUTPUT_FOLDER}/risk_summary.csv", index=False)
print("risk_summary.csv saved!")

# FILE 3: Doom Score by Category
posts_df = pd.read_csv(f"{OUTPUT_FOLDER}/posts_with_doom_scores.csv")
doom_by_category = posts_df.groupby('category_name').agg(
    Avg_Doom_Score=('doom_score',     'mean'),
    Min_Doom_Score=('doom_score',     'min'),
    Max_Doom_Score=('doom_score',     'max'),
    Median_Doom=('doom_score',        'median'),
    HIGH_Risk_Posts=('post_doom_risk', lambda x: (x == 'HIGH').sum()),
    MEDIUM_Risk_Posts=('post_doom_risk', lambda x: (x == 'MEDIUM').sum()),
    LOW_Risk_Posts=('post_doom_risk',  lambda x: (x == 'LOW').sum()),
    Total_Posts=('doom_score',        'count')
).round(2).reset_index()
doom_by_category.to_csv(f"{OUTPUT_FOLDER}/doom_by_category.csv", index=False)
print("doom_by_category.csv saved!")

# FILE 4: Jaccard Similarity (new — echo chamber boundary strength)
jaccard_df = pd.read_csv(f"{OUTPUT_FOLDER}/jaccard_similarity.csv")
jaccard_df.to_csv(f"{OUTPUT_FOLDER}/jaccard_powerbi.csv", index=False)
print("jaccard_powerbi.csv saved!")

# FILE 5: Session Risk Detail (main Power BI table)
session_detail = final_df.sort_values(
    'combined_risk_score', ascending=False
).reset_index(drop=True)
session_detail['risk_rank'] = session_detail.index + 1
session_detail.to_csv(f"{OUTPUT_FOLDER}/session_risk_detail.csv", index=False)
print("session_risk_detail.csv saved!")

# FILE 6: Post Sample for Power BI
posts_sample = posts_df[[
    'title', 'category_name', 'word_count',
    'negative_word_count', 'negativity_density',
    'doom_score', 'post_doom_risk', 'is_long_post'
]].copy()
posts_sample = posts_sample.groupby(
    'category_name', group_keys=False
).apply(lambda x: x.sample(
    min(200, len(x)), random_state=42
)).reset_index(drop=True)
posts_sample.to_csv(f"{OUTPUT_FOLDER}/posts_sample_powerbi.csv", index=False)
print("posts_sample_powerbi.csv saved!")

print("\n" + "="*55)
print("   ALL POWER BI FILES GENERATED SUCCESSFULLY")
print("="*55)