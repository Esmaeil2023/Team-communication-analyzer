import pandas as pd
import numpy as np

# ── Load features ────────────────────────────────────────────────────────
features = pd.read_json("features.json", lines=True)
print(f"Loaded features for {len(features)} users")

# ── Print key distribution stats to understand the data ──────────────────
print("\nData distribution:")
for col in ['msg_count', 'replies_sent', 'interaction_count', 'avg_msg_length']:
    print(f"  {col}: "
          f"median={features[col].median():.1f}  "
          f"p70={features[col].quantile(0.70):.1f}  "
          f"p85={features[col].quantile(0.85):.1f}  "
          f"p95={features[col].quantile(0.95):.1f}")

# ── Thresholds ────────────────────────────────────────────────────────────
def pct(col, p): return features[col].quantile(p)

high_msgs         = pct('msg_count', 0.90)
med_msgs          = pct('msg_count', 0.60)
low_msgs          = pct('msg_count', 0.25)

high_replies      = pct('replies_sent', 0.90)   # raised — top 10% repliers only
high_interactions = pct('interaction_count', 0.85)
low_interactions  = pct('interaction_count', 0.25)

high_length       = pct('avg_msg_length', 0.75)
med_length        = pct('avg_msg_length', 0.50)

print(f"\nThresholds used:")
print(f"  high_msgs={high_msgs:.0f}  med_msgs={med_msgs:.0f}  low_msgs={low_msgs:.0f}")
print(f"  high_replies={high_replies:.0f}")
print(f"  high_interactions={high_interactions:.0f}  low_interactions={low_interactions:.0f}")
print(f"  high_length={high_length:.0f}  med_length={med_length:.0f}")

# ── Role assignment ───────────────────────────────────────────────────────
def assign_role(row):
    msgs    = row['msg_count']
    replies = row['replies_sent']
    inter   = row['interaction_count']
    length  = row['avg_msg_length']

    # Leader: top 10% activity + top 15% interaction breadth
    if msgs >= high_msgs and inter >= high_interactions:
        return 'Leader'

    # Coordinator: heavy replier + broad interactions (not necessarily most posts)
    if replies >= high_replies and inter >= high_interactions:
        return 'Coordinator'

    # Contributor: writes long messages + above average volume
    if length >= high_length and msgs >= med_msgs:
        return 'Contributor'

    # Reactive: high replies but only to a narrow set (not broad coordinator)
    if replies >= high_replies and inter < high_interactions:
        return 'Reactive'

    # Isolated: very few messages + very few interactions
    if msgs <= low_msgs and inter <= low_interactions:
        return 'Isolated'

    # Passive: everything else — present but unremarkable
    return 'Passive'

features['role'] = features.apply(assign_role, axis=1)

# ── Distribution ──────────────────────────────────────────────────────────
print("\nRole distribution:")
dist = features['role'].value_counts()
print(dist)
print("\nPercentages:")
print((dist / len(features) * 100).round(1))

# ── Big Five mapping ──────────────────────────────────────────────────────
def map_big_five(row):
    def norm(val, col):
        mn = features[col].min()
        mx = features[col].max()
        if mx == mn: return 50.0
        return round((val - mn) / (mx - mn) * 100, 1)

    return pd.Series({
        'extraversion'     : norm(row['msg_count'],             'msg_count'),
        'agreeableness'    : norm(row['replies_sent'],          'replies_sent'),
        'openness'         : norm(row['avg_msg_length'],        'avg_msg_length'),
        'conscientiousness': norm(row['active_days'],           'active_days'),
        'neuroticism'      : norm(row['sentiment_variability'], 'sentiment_variability')
    })

print("\nComputing Big Five scores...")
big_five = features.apply(map_big_five, axis=1)
features = pd.concat([features, big_five], axis=1)

# ── Save ──────────────────────────────────────────────────────────────────
features.to_json("results.json", orient="records", lines=True)
print("\nSaved to results.json")

print("\nSample — top 10 most active users:")
cols = ['author', 'role', 'msg_count', 'replies_sent',
        'extraversion', 'agreeableness', 'openness',
        'conscientiousness', 'neuroticism']
print(features.sort_values('msg_count', ascending=False)[cols].head(10).to_string())