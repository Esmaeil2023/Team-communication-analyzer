import pandas as pd

# ── Load from JSON instead of CSV ───────────────────────────────────────
df = pd.read_json("reddit_raw.json", lines=True)
print(f"Raw shape: {df.shape}")

# ── Drop bots and system accounts ───────────────────────────────────────
bots = ['AutoModerator', 'reddit_tos', 'BotTerminator', 'sneakpeekbot']
df = df[~df['author'].isin(bots)]

# ── Drop deleted/removed content ────────────────────────────────────────
df = df[~df['body'].isin(['[deleted]', '[removed]', ''])]
df = df.dropna(subset=['author', 'body', 'created_utc', 'parent_id'])

# ── Normalize author names ───────────────────────────────────────────────
df['author'] = df['author'].str.lower().str.strip()

# ── Convert timestamp to readable datetime ──────────────────────────────
df['datetime'] = pd.to_datetime(df['created_utc'], unit='s')

# ── Remove very short messages (noise) ──────────────────────────────────
df = df[df['body'].str.len() >= 10]

# ── Reset index ─────────────────────────────────────────────────────────
df = df.reset_index(drop=True)

# ── Save clean version also as JSON ─────────────────────────────────────
df.to_json("reddit_clean.json", orient="records", lines=True)

print(f"Clean shape: {df.shape}")
print(f"Unique users: {df['author'].nunique()}")
print(f"Date range: {df['datetime'].min()} → {df['datetime'].max()}")
print(f"Subreddits: {df['subreddit'].value_counts().to_dict()}")
print("\nSample:")
print(df[['author', 'datetime', 'body', 'parent_id']].head(3))