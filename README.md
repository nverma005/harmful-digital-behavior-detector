# Harmful Digital Behavior Detection System

## Overview
An end-to-end data analytics project detecting two harmful digital behaviors — Echo Chambers and Doom Scrolling — using Reddit Mental Health data, Python, and Power BI. All 5,940 posts are processed across 198 real content sessions. No synthetic users. No sampling with replacement. Fixed random seed (42) ensures every run is reproducible.

## Dashboard Preview

![Overview](Screenshots/1_overview.png)
*Risk overview: combined scoring across all analyzed users*

![Echo Chamber Analysis](Screenshots/2_echo_chamber.png)
*Echo chamber detection using Shannon Entropy diversity scoring*

![Doom Scrolling Analysis](Screenshots/3_doomscrolling.png)
*Doom-scroll risk based on weighted negativity scoring*

## Problem Statement
- **Echo Chambers**: Users consume content from only one category, narrowing their perspective
- **Doom Scrolling**: Users compulsively consume high-negativity content, damaging mental health

## Dataset
- Source: [Reddit Mental Health Dataset](https://www.kaggle.com/datasets/neelghoshal/reddit-mental-health-data?select=data_to_be_cleansed.csv) (Kaggle, public dataset)
- Size: 5,957 posts across 5 mental health subreddits
- Categories: Stress, Depression, Bipolar, Social Anxiety, General Anxiety

## Limitations & Ethical Considerations
This project uses a publicly available, pre-anonymized dataset for educational and portfolio purposes only. It is not connected to real identifiable individuals and was never built for deployment against real users.

"Risk" scores reflect content patterns, not clinical or psychological assessments. Do not interpret them as medical judgments.

**Analytical choices made transparent:**
-> Risk thresholds at post and session level are percentile-based, derived from the actual data distribution, not arbitrary fixed numbers.
-> Category doom weights (Depression 0.9, Stress 0.4) are domain-knowledge estimates, not empirically fitted values.
-> Echo chamber detection uses vocabulary exclusivity scoring across all 5,940 posts and Jaccard similarity across 10 category pairs.
-> Key finding: mental health subreddits share 68-76% vocabulary overlap. Echo chamber risk here is about immersion in a shared distress vocabulary, not category isolation.

This project demonstrates data analysis and feature engineering skills. It makes no claims about individual mental health.

## Project Files
| File | Description |
|------|-------------|
| 01_data_exploration.py | Dataset profiling and EDA |
| 02_data_cleaning.py | Text cleaning and feature engineering |
| 3.1_echo_chamber_analysis.py | Shannon Entropy diversity scoring |
| 3.2_doomscrolling_analysis.py | Weighted doom score per post and user |
| 04_final_output.py | Combined risk engine and Power BI exports |

## Key Results
| Metric | Value |
|--------|-------|
| Total Sessions Analysed | 198 |
| HIGH Risk Sessions | 40 |
| MEDIUM Risk Sessions | 59 |
| LOW Risk Sessions | 99 |
| Sessions with Echo Bubble Signal | 18 |
| Heavy Doom Scroll Sessions | 5 |
| Double Risk Sessions | 9 |
| Avg Combined Risk Score | 39.90 |

## Tech Stack
- Python: pandas, numpy
- Power BI: 3-page interactive dashboard
- Algorithms: Shannon Entropy, Min-Max Normalization, Weighted Scoring

## How to Run
```bash
pip install pandas numpy
python 01_data_exploration.py
python 02_data_cleaning.py
python 3.1_echo_chamber_analysis.py
python 3.2_doomscrolling_analysis.py
python 04_final_output.py
```

## Author
Neha Verma  
MSc Data Science — University of Strathclyde, United Kingdom
