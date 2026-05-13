import pandas as pd
import numpy as np
from textblob import TextBlob
import re

df = pd.read_json("whatsapp_clean.json", lines=True)
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)
print(f"Loaded {len(df)} messages from {df['author'].nunique()} users")

# ── 1. Message count ──────────────────────────────────────────────────────
msg_count = df.groupby('author').size().rename('msg_count')

# ── 2. Avg message length ─────────────────────────────────────────────────
avg_length = df.groupby('author')['body'].apply(
    lambda x: x.str.len().mean()).rename('avg_msg_length')

# ── 3. Question ratio (Bee asks many questions / explores ideas) ──────────
df['is_question'] = df['body'].str.contains(r'\?', regex=True).astype(int)
question_ratio = df.groupby('author')['is_question'].mean().rename('question_ratio')

# ── 4. @mentions — how many different people someone addresses (Bee signal)
df['mention_count'] = df['body'].str.count(r'@\w+')
avg_mentions = df.groupby('author')['mention_count'].mean().rename('avg_mentions')

# ── 5. Reply speed — how fast someone responds (Ant = fast, Leech = slow) ─
df['prev_time']   = df['datetime'].shift(1)
df['prev_author'] = df['author'].shift(1)
df['gap_mins']    = (df['datetime'] - df['prev_time']).dt.total_seconds() / 60
df['is_reply']    = ((df['gap_mins'] <= 10) &
                     (df['prev_author'] != df['author'])).astype(int)
reply_speed = df[df['is_reply']==1].groupby('author')['gap_mins'].median().rename('avg_reply_speed_mins')
replies_sent = df.groupby('author')['is_reply'].sum().rename('replies_sent')

# ── 6. Task-focused language (Ant signal) ────────────────────────────────
task_words = ['done', 'finished', 'completed', 'sent', 'here', 'attached',
              'will do', 'ok', 'okay', 'sure', 'deadline', 'file', 'link',
              'submitted', 'ready', 'push', 'commit', 'fixed', 'update']
df['task_score'] = df['body'].str.lower().apply(
    lambda x: sum(w in x for w in task_words)
)
avg_task_score = df.groupby('author')['task_score'].mean().rename('task_focus_score')

# ── 7. Summarizing / reformulating language (Butterfly signal) ────────────
butterfly_words = ['so basically', 'in summary', 'to summarize', 'in other words',
                   'what i mean', 'let me explain', 'to clarify', 'in short',
                   'the idea is', 'what we mean', 'so what']
df['butterfly_score'] = df['body'].str.lower().apply(
    lambda x: sum(w in x for w in butterfly_words)
)
avg_butterfly = df.groupby('author')['butterfly_score'].mean().rename('butterfly_score')

# ── 8. Positive/supportive sentiment (Capybara signal) ───────────────────
print("Computing sentiment (takes ~1 min for small groups)...")
df['sentiment'] = df['body'].apply(
    lambda x: TextBlob(str(x)).sentiment.polarity
)
avg_sentiment = df.groupby('author')['sentiment'].mean().rename('avg_sentiment')

capybara_words = ['great', 'well done', 'good job', 'thanks', 'thank you',
                  'appreciate', 'agree', 'exactly', 'love', 'perfect',
                  'awesome', 'nice', 'good point', 'helpful', 'support']
df['capybara_score'] = df['body'].str.lower().apply(
    lambda x: sum(w in x for w in capybara_words)
)
avg_capybara = df.groupby('author')['capybara_score'].mean().rename('capybara_score')

# ── 9. Active days (consistency — Leech is inconsistent) ─────────────────
df['date']  = df['datetime'].dt.date
active_days = df.groupby('author')['date'].nunique().rename('active_days')

# ── 10. New topic ratio (Bee introduces new ideas) ───────────────────────
# Proxy: messages NOT replying (i.e., sent >10 min after last message)
df['is_new_topic'] = (df['gap_mins'] > 10).astype(int)
new_topic_ratio = df.groupby('author')['is_new_topic'].mean().rename('new_topic_ratio')

# ── Combine ───────────────────────────────────────────────────────────────
features = pd.concat([
    msg_count, avg_length, question_ratio, avg_mentions,
    reply_speed, replies_sent, avg_task_score, avg_butterfly,
    avg_sentiment, avg_capybara, active_days, new_topic_ratio
], axis=1).fillna(0).reset_index()

features.to_json("wa_features.json", orient="records", lines=True)
print(f"\nFeature matrix: {features.shape}")
print(features.sort_values('msg_count', ascending=False).to_string())