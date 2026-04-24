# Team Communication Analyzer

Analyzes Reddit comment data to classify users into communication roles
and map them to Big Five personality traits.

## Pipeline
1. `extract_data.py` — pulls 63k comments from Reddit May 2015 dataset
2. `clean_data.py` — removes bots, normalizes, converts timestamps
3. `feature_extraction.py` — computes 9 behavioral features per user
4. `role_classification.py` — assigns 6 roles + Big Five scores
5. `dashboard.py` — interactive Streamlit dashboard

## Setup
```bash
pip install pandas numpy scikit-learn nltk textblob networkx streamlit matplotlib seaborn
streamlit run dashboard.py
```

## Data
Reddit May 2015 dataset from Kaggle — r/programming + r/learnprogramming  
63,150 messages · 15,819 users

## Roles
Leader · Coordinator · Contributor · Reactive · Passive · Isolated

## Big Five Mapping
Extraversion → message volume  
Agreeableness → reply frequency  
Openness → message length  
Conscientiousness → posting consistency  
Neuroticism → sentiment variability