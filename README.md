# Harmful Digital Behavior Detection System

## Overview
An end-to-end data analytics project that detects two harmful 
digital behaviors — Echo Chambers and Doom Scrolling — using 
Reddit Mental Health data, Python, and Power BI.

## Problem Statement
- **Echo Chambers**: Users consume content from only one category, narrowing their perspective
- **Doom Scrolling**: Users compulsively consume high-negativity content, damaging mental health

## Dataset
- Source: Reddit Mental Health Dataset
- Size: 5,957 posts across 5 mental health subreddits
- Categories: Stress, Depression, Bipolar, Social Anxiety, General Anxiety

## Project Files
| File | Description | Notebooks
|------|-------------|

| 01_data_exploration.py | Dataset profiling and EDA |
| 02_data_cleaning.py | Text cleaning and feature engineering |
| 3.1_echo_chamber_analysis.py | Shannon Entropy diversity scoring |
| 3.2_doomscrolling_analysis.py | Weighted doom score per post and user |
| 04_final_output.py | Combined risk engine and Power BI exports |

## Key Results
| Metric | Value |
|--------|-------|
| Total Users Analysed | 100 |
| HIGH Risk Users | 20 |
| Users in Echo Bubble | 30 |
| Heavy Doom Scroll Users | 10 |
| Double Risk Users | 6 |
| Avg Combined Risk Score | 43.60 |

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
MSc Data Science — University of Strathclyde
