import pandas as pd
import numpy as np
from textblob import TextBlob

# ── Load clean data ──────────────────────────────────────────────────────
df = pd.read_json("reddit_clean.json", lines=True)
df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
df = df.sort_values('datetime').reset_index(drop=True)
print(f"Loaded {len(df)} messages from {df['author'].nunique()} users")

# ── Build reply map: comment_id → author ────────────────────────────────
# parent_id looks like "t1_cquckvs" — strip the "t1_" prefix to get id
# We need to map each comment's parent back to who wrote it

# Extract comment id from parent_id (strip prefix)
df['comment_id'] = df.index.astype(str)  # use row index as proxy id
id_to_author = dict(zip(df['comment_id'], df['author']))

# ── 1. MESSAGE COUNT per user ────────────────────────────────────────────
msg_count = df.groupby('author').size().rename('msg_count')

# ── 2. AVERAGE MESSAGE LENGTH ────────────────────────────────────────────
df['msg_length'] = df['body'].str.len()
avg_length = df.groupby('author')['msg_length'].mean().rename('avg_msg_length')

# ── 3. QUESTION RATIO (messages containing "?") ──────────────────────────
df['is_question'] = df['body'].str.contains(r'\?', regex=True).astype(int)
question_ratio = df.groupby('author')['is_question'].mean().rename('question_ratio')

# ── 4. REPLIES SENT (how many times user replied to someone) ─────────────
# A reply has parent_id starting with "t1_" (reply to comment, not post)
df['is_reply'] = df['parent_id'].str.startswith('t1_').astype(int)
replies_sent = df.groupby('author')['is_reply'].sum().rename('replies_sent')

# ── 5. UNIQUE USERS INTERACTED WITH ──────────────────────────────────────
# Build parent_id → author lookup from actual data
# parent_id in format t1_XXXXX — we need to match against a comment id
# Since we don't have explicit comment ids in this dataset,
# we approximate: count unique subreddits/threads user replied in
interaction_count = df[df['is_reply'] == 1].groupby('author')['subreddit'].count().rename('interaction_count')

# ── 6. SENTIMENT SCORE ───────────────────────────────────────────────────
print("Computing sentiment scores (this takes ~2 minutes)...")

def get_sentiment(text):
    try:
        return TextBlob(str(text)).sentiment.polarity
    except:
        return 0.0

df['sentiment'] = df['body'].apply(get_sentiment)
avg_sentiment   = df.groupby('author')['sentiment'].mean().rename('avg_sentiment')
sentiment_std   = df.groupby('author')['sentiment'].std().rename('sentiment_variability')

# ── 7. ACTIVITY CONSISTENCY ───────────────────────────────────────────────
# How many different days the user posted (out of 31 days in May)
df['day'] = df['datetime'].dt.day
active_days = df.groupby('author')['day'].nunique().rename('active_days')

# ── 8. AVERAGE RESPONSE TIME ─────────────────────────────────────────────
# Sort by author and time, compute gap between consecutive posts
df_sorted = df.sort_values(['author', 'datetime'])
df_sorted['prev_time'] = df_sorted.groupby('author')['datetime'].shift(1)
df_sorted['response_gap_mins'] = (
    (df_sorted['datetime'] - df_sorted['prev_time'])
    .dt.total_seconds() / 60
)
avg_response_time = df_sorted.groupby('author')['response_gap_mins'].median().rename('avg_response_time_mins')

# ── Combine all features ─────────────────────────────────────────────────
features = pd.concat([
    msg_count,
    avg_length,
    question_ratio,
    replies_sent,
    interaction_count,
    avg_sentiment,
    sentiment_std,
    active_days,
    avg_response_time
], axis=1).fillna(0)

features = features.reset_index()  # brings 'author' back as column
print(f"\nFeature matrix shape: {features.shape}")
print(f"Columns: {features.columns.tolist()}")
print("\nSample (top 5 most active users):")
print(features.sort_values('msg_count', ascending=False).head())

# ── Save ─────────────────────────────────────────────────────────────────
features.to_json("features.json", orient="records", lines=True)
print("\nSaved to features.json")