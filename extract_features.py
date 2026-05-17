"""
extract_features.py
───────────────────
Extracts 16 behavioral + linguistic features per user from WhatsApp data.

New features added from Classification_Archetypes.pdf:
  - MATTR (vocabulary diversity) → Bee signal
  - Response latency variance    → Ant signal (consistency)
  - In/Out ratio                 → Leech signal
  - Acknowledgment rate          → Capybara signal
  - Emoji ratio                  → Butterfly signal
  - Initiation rate              → Bee signal

Runs on whatsapp_clean.json → wa_features.json
"""

import pandas as pd
import numpy as np
import re
from textblob import TextBlob

INPUT_FILE  = "whatsapp_clean.json"
OUTPUT_FILE = "wa_features.json"

# ── Load ──────────────────────────────────────────────────────────────────
df = pd.read_json(INPUT_FILE, lines=True)
df['datetime'] = pd.to_datetime(df['datetime'], unit='ms', errors='coerce')
df = df.dropna(subset=['datetime']).sort_values('datetime').reset_index(drop=True)
print(f"Loaded {len(df):,} messages from {df['author'].nunique()} users")

# ── Reply detection ───────────────────────────────────────────────────────
df['prev_author'] = df['author'].shift(1)
df['prev_time']   = df['datetime'].shift(1)
df['gap_mins']    = (df['datetime'] - df['prev_time']).dt.total_seconds() / 60
df['is_reply']    = (
    (df['gap_mins'] <= 10) & (df['prev_author'] != df['author'])
).astype(int)
df['date'] = df['datetime'].dt.date

# ════════════════════════════════════════════════════════════════════════
# ORIGINAL 11 FEATURES
# ════════════════════════════════════════════════════════════════════════

msg_count  = df.groupby('author').size().rename('msg_count')
avg_length = df.groupby('author')['body'].apply(
    lambda x: x.str.len().mean()).rename('avg_msg_length')

df['is_question']   = df['body'].str.contains(r'\?', regex=True).astype(int)
question_ratio      = df.groupby('author')['is_question'].mean().rename('question_ratio')

df['mention_count'] = df['body'].str.count(r'@\w+')
avg_mentions        = df.groupby('author')['mention_count'].mean().rename('avg_mentions')

replies_sent = df.groupby('author')['is_reply'].sum().rename('replies_sent')

task_words = ['done', 'finished', 'completed', 'sent', 'here', 'attached',
              'will do', 'ok', 'okay', 'sure', 'deadline', 'file', 'link',
              'submitted', 'ready', 'push', 'commit', 'fixed', 'update',
              'works', 'working', 'solved', 'merged', 'deployed', 'closed']
df['task_score'] = df['body'].str.lower().apply(
    lambda x: sum(w in x for w in task_words))
avg_task = df.groupby('author')['task_score'].mean().rename('task_focus_score')

butterfly_words = ['so basically', 'in summary', 'to summarize', 'in other words',
                   'what i mean', 'let me explain', 'to clarify', 'in short',
                   'the idea is', 'essentially', 'in a nutshell', 'meaning that']
df['bfly_score'] = df['body'].str.lower().apply(
    lambda x: sum(w in x for w in butterfly_words))
avg_butterfly = df.groupby('author')['bfly_score'].mean().rename('butterfly_score')

print("Computing sentiment...", end='', flush=True)
df['sentiment'] = df['body'].apply(
    lambda x: TextBlob(str(x)).sentiment.polarity)
print(" done")
avg_sentiment = df.groupby('author')['sentiment'].mean().rename('avg_sentiment')

capybara_words = ['great', 'well done', 'good job', 'thanks', 'thank you',
                  'appreciate', 'agree', 'exactly', 'love', 'perfect',
                  'awesome', 'nice', 'good point', 'helpful', 'support',
                  'welcome', 'brilliant', 'excellent', 'congrats', 'anytime']
df['cap_score'] = df['body'].str.lower().apply(
    lambda x: sum(w in x for w in capybara_words))
avg_capybara = df.groupby('author')['cap_score'].mean().rename('capybara_score')

active_days     = df.groupby('author')['date'].nunique().rename('active_days')

df['is_new_topic'] = (df['gap_mins'] > 10).astype(int)
new_topic_ratio    = df.groupby('author')['is_new_topic'].mean().rename('new_topic_ratio')

# ════════════════════════════════════════════════════════════════════════
# NEW FEATURES (from Classification_Archetypes.pdf)
# ════════════════════════════════════════════════════════════════════════

# ── 1. MATTR — Moving Average Type-Token Ratio (vocabulary diversity) ────
# High MATTR = Bee (uses varied, idea-rich vocabulary)
# Sliding window of 20 tokens; average TTR across windows
def compute_mattr(texts, window=20):
    tokens = ' '.join(str(t) for t in texts).lower().split()
    if len(tokens) < window:
        unique = len(set(tokens))
        return unique / len(tokens) if tokens else 0
    ttrs = []
    for i in range(len(tokens) - window + 1):
        w    = tokens[i:i + window]
        ttrs.append(len(set(w)) / window)
    return round(np.mean(ttrs), 4)

print("Computing MATTR (vocabulary diversity)...", end='', flush=True)
mattr = df.groupby('author')['body'].apply(
    lambda x: compute_mattr(x.tolist())
).rename('mattr')
print(" done")

# ── 2. Response Latency Variance — Ant = low variance (consistent) ────────
# Only count actual reply gaps (≤ 10 mins), ignore long silences
reply_gaps = df[df['is_reply'] == 1].groupby('author')['gap_mins']
latency_mean = reply_gaps.mean().rename('avg_reply_latency_mins')
latency_var  = reply_gaps.std().fillna(0).rename('reply_latency_variance')

# ── 3. In/Out Ratio — Leech = receives many replies, sends few ────────────
# "In" = times others replied TO this person (next message is from someone else
#        within 10 mins after this person's message)
df['next_author'] = df['author'].shift(-1)
df['next_gap']    = df['gap_mins'].shift(-1).fillna(999)
df['got_reply']   = (
    (df['next_gap'] <= 10) &
    (df['next_author'] != df['author'])
).astype(int)

replies_received = df.groupby('author')['got_reply'].sum().rename('replies_received')

# In/Out ratio: received / sent (high = Leech, balanced = healthy)
def safe_ratio(received, sent):
    if sent == 0:
        return float(received) if received > 0 else 1.0
    return round(received / sent, 3)

in_out_ratio = pd.Series({
    author: safe_ratio(
        replies_received.get(author, 0),
        replies_sent.get(author, 0)
    )
    for author in df['author'].unique()
}, name='in_out_ratio')

# ── 4. Acknowledgment Rate — Capybara starts messages with affirmation ────
ack_starters = ['yes', 'yeah', 'yep', 'ok', 'okay', 'sure', 'great', 'nice',
                'good', 'perfect', 'exactly', 'right', 'true', 'agree',
                'thanks', 'thank', 'awesome', 'cool', 'got it', 'noted',
                'of course', 'absolutely', 'definitely', 'makes sense']

def starts_with_ack(text):
    t = str(text).lower().strip()
    return any(t.startswith(w) for w in ack_starters)

df['is_ack']    = df['body'].apply(starts_with_ack).astype(int)
ack_rate        = df.groupby('author')['is_ack'].mean().rename('acknowledgment_rate')

# ── 5. Emoji Ratio — Butterfly uses emotional/visual language ─────────────
emoji_pattern = re.compile(
    "[\U0001F300-\U0001F9FF"
    "\U00002600-\U000027BF"
    "\U0001FA00-\U0001FA9F"
    "\u2600-\u27BF]+",
    flags=re.UNICODE
)
df['emoji_count'] = df['body'].apply(lambda x: len(emoji_pattern.findall(str(x))))
emoji_ratio       = df.groupby('author')['emoji_count'].mean().rename('emoji_ratio')

# ── Emotion Density (NRC-style word list) — Butterfly/Capybara signal ────
# Based on Mohammad & Turney (2013) emotion lexicon concept
# Using a curated English emotion word list as proxy
emotion_words = [
    # Joy / positive
    'happy','joy','love','great','wonderful','excited','glad','enjoy',
    'pleasure','delight','fantastic','amazing','excellent','brilliant',
    # Surprise
    'wow','surprised','unexpected','incredible','unbelievable',
    # Trust / support
    'trust','reliable','honest','support','believe','confident',
    # Fear / anticipation
    'worried','afraid','nervous','anxious','concerned','hope','expect',
    # Anger / negative
    'angry','frustrated','annoyed','upset','hate','terrible','awful',
    # Sadness
    'sad','sorry','unfortunate','disappointed','miss','regret'
]
df['emotion_count'] = df['body'].str.lower().apply(
    lambda x: sum(w in x.split() for w in emotion_words)
)
emotion_density = df.groupby('author')['emotion_count'].apply(
    lambda x: (x > 0).mean()
).rename('emotion_density')

# ── Betweenness Centrality — Bee signal (Freeman 1977) ────────────────────
import networkx as nx
G_cent = nx.DiGraph()
for author in df['author'].unique():
    G_cent.add_node(author)
for i in range(1, len(df)):
    a    = df.iloc[i]['author']
    prev = df.iloc[i-1]['author']
    gap  = df.iloc[i]['gap_mins'] if 'gap_mins' in df.columns else 999
    if a != prev and gap <= 10:
        if G_cent.has_edge(a, prev):
            G_cent[a][prev]['weight'] += 1
        else:
            G_cent.add_edge(a, prev, weight=1)

if G_cent.number_of_edges() > 0:
    centrality = nx.betweenness_centrality(G_cent, normalized=True)
else:
    centrality = {a: 0.0 for a in df['author'].unique()}
betweenness = pd.Series(centrality, name='betweenness_centrality')

# ── 6. Initiation Rate — Bee starts new conversations ────────────────────
# Already have new_topic_ratio but let's also compute pure initiation
# (first message in a gap > 30 mins = true new conversation starter)
df['is_initiator'] = (df['gap_mins'] > 30).astype(int)
initiation_rate    = df.groupby('author')['is_initiator'].mean().rename('initiation_rate')

# ════════════════════════════════════════════════════════════════════════
# COMBINE ALL FEATURES
# ════════════════════════════════════════════════════════════════════════

features = pd.concat([
    # Original 11
    msg_count, avg_length, question_ratio, avg_mentions,
    replies_sent, avg_task, avg_butterfly,
    avg_sentiment, avg_capybara, active_days, new_topic_ratio,
    # New 6
    mattr, latency_mean, latency_var,
    replies_received, in_out_ratio,
    ack_rate, emoji_ratio, initiation_rate,
    # From document: emotion density + betweenness centrality
    emotion_density, betweenness
], axis=1).fillna(0).reset_index()

# Ensure author is a regular column
if features.index.name == 'author':
    features = features.reset_index()
elif 'author' not in features.columns and 'index' in features.columns:
    features = features.rename(columns={'index': 'author'})

features.to_json(OUTPUT_FILE, orient="records", lines=True)

print(f"\n✅ Feature matrix: {features.shape}")
print(f"   Columns: {features.columns.tolist()}")
print(f"\nFull feature table:")
print(features.sort_values('msg_count', ascending=False).to_string())
print(f"\n✅ Saved to {OUTPUT_FILE}")