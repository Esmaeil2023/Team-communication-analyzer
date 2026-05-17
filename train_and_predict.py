"""
train_and_predict.py
────────────────────
Hybrid COIN Archetype Classifier — Multi-Dataset, Weighted Scores

Key upgrades from Classification_Archetypes.pdf:
  1. 17 features (added MATTR, latency variance, in/out ratio,
     acknowledgment rate, emoji ratio, initiation rate)
  2. Weighted archetype score formula (not simple average)
  3. Output = percentage DISTRIBUTION across all 5 archetypes
     (not a single hard label) — "the dose makes the poison"
  4. Hard label = highest scoring archetype in the distribution

Training: Slack (Chatterjee 2020) + Nankani 2020 CODERS WhatsApp
Test:      Your team's WhatsApp (5 members)
"""

import pandas as pd
import numpy as np
import re
from textblob import TextBlob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ── All feature columns ───────────────────────────────────────────────────
FEATURE_COLS = [
    # Original
    'msg_count', 'avg_msg_length', 'question_ratio', 'avg_mentions',
    'replies_sent', 'task_focus_score', 'butterfly_score',
    'avg_sentiment', 'capybara_score', 'active_days', 'new_topic_ratio',
    # New
    'mattr', 'reply_latency_variance', 'in_out_ratio',
    'acknowledgment_rate', 'emoji_ratio', 'initiation_rate'
]

ARCHETYPE_ORDER = ['🐝 Bee', '🐜 Ant', '🦋 Butterfly', '🦫 Capybara', '🔴 Leech']

# ════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION (for training datasets — handles large scale)
# ════════════════════════════════════════════════════════════════════════

def compute_mattr(texts, window=20):
    tokens = ' '.join(str(t) for t in texts).lower().split()
    if len(tokens) < window:
        return len(set(tokens)) / len(tokens) if tokens else 0
    ttrs = [len(set(tokens[i:i+window])) / window
            for i in range(len(tokens) - window + 1)]
    return round(np.mean(ttrs), 4)

def extract_features(json_path, label="dataset"):
    print(f"\n{'='*55}")
    print(f"  Extracting features: {label}")
    print(f"{'='*55}")

    df = pd.read_json(json_path, lines=True)
    df['datetime'] = pd.to_datetime(df['datetime'], unit='ms', errors='coerce')
    df = df.dropna(subset=['datetime']).sort_values('datetime').reset_index(drop=True)
    print(f"  Messages: {len(df):,}  |  Users: {df['author'].nunique():,}")

    df['prev_author'] = df['author'].shift(1)
    df['prev_time']   = df['datetime'].shift(1)
    df['gap_mins']    = (df['datetime'] - df['prev_time']).dt.total_seconds() / 60
    df['is_reply']    = (
        (df['gap_mins'] <= 10) & (df['prev_author'] != df['author'])
    ).astype(int)
    df['date'] = df['datetime'].dt.date

    # ── Original features ─────────────────────────────────────────────────
    msg_count  = df.groupby('author').size().rename('msg_count')
    avg_length = df.groupby('author')['body'].apply(
        lambda x: x.str.len().mean()).rename('avg_msg_length')

    df['is_question']   = df['body'].str.contains(r'\?', regex=True).astype(int)
    question_ratio      = df.groupby('author')['is_question'].mean().rename('question_ratio')
    df['mention_count'] = df['body'].str.count(r'@\w+')
    avg_mentions        = df.groupby('author')['mention_count'].mean().rename('avg_mentions')
    replies_sent        = df.groupby('author')['is_reply'].sum().rename('replies_sent')

    task_words = ['done', 'finished', 'completed', 'sent', 'here', 'attached',
                  'will do', 'ok', 'okay', 'sure', 'deadline', 'file', 'link',
                  'submitted', 'ready', 'push', 'commit', 'fixed', 'update',
                  'works', 'working', 'solved', 'merged', 'deployed', 'closed',
                  'kiya', 'ho gaya', 'kar diya']
    df['task_score'] = df['body'].str.lower().apply(
        lambda x: sum(w in x for w in task_words))
    avg_task = df.groupby('author')['task_score'].mean().rename('task_focus_score')

    butterfly_words = ['so basically', 'in summary', 'to summarize', 'in other words',
                       'what i mean', 'let me explain', 'to clarify', 'in short',
                       'the idea is', 'essentially', 'in a nutshell']
    df['bfly_score'] = df['body'].str.lower().apply(
        lambda x: sum(w in x for w in butterfly_words))
    avg_butterfly = df.groupby('author')['bfly_score'].mean().rename('butterfly_score')

    print(f"  Computing sentiment...", end='', flush=True)
    if len(df) > 20000:
        sample_idx = df.groupby('author').head(30).index
        df.loc[sample_idx, 'sentiment'] = df.loc[sample_idx, 'body'].apply(
            lambda x: TextBlob(str(x)).sentiment.polarity)
        df['sentiment'] = df['sentiment'].fillna(0)
    else:
        df['sentiment'] = df['body'].apply(
            lambda x: TextBlob(str(x)).sentiment.polarity)
    print(" done")
    avg_sentiment = df.groupby('author')['sentiment'].mean().rename('avg_sentiment')

    capybara_words = ['great', 'well done', 'good job', 'thanks', 'thank you',
                      'appreciate', 'agree', 'exactly', 'love', 'perfect',
                      'awesome', 'nice', 'good point', 'helpful', 'support',
                      'welcome', 'brilliant', 'excellent', 'congrats', 'anytime',
                      'thanks alot', 'relax', 'no problem']
    df['cap_score'] = df['body'].str.lower().apply(
        lambda x: sum(w in x for w in capybara_words))
    avg_capybara  = df.groupby('author')['cap_score'].mean().rename('capybara_score')
    active_days   = df.groupby('author')['date'].nunique().rename('active_days')
    df['is_new']  = (df['gap_mins'] > 10).astype(int)
    new_topic_ratio = df.groupby('author')['is_new'].mean().rename('new_topic_ratio')

    # ── New features ──────────────────────────────────────────────────────
    print(f"  Computing MATTR...", end='', flush=True)
    if len(df) > 20000:
        sample = df.groupby('author').head(50)
        mattr  = sample.groupby('author')['body'].apply(
            lambda x: compute_mattr(x.tolist())).rename('mattr')
    else:
        mattr = df.groupby('author')['body'].apply(
            lambda x: compute_mattr(x.tolist())).rename('mattr')
    print(" done")

    reply_df        = df[df['is_reply'] == 1]
    latency_var     = reply_df.groupby('author')['gap_mins'].std().fillna(0).rename('reply_latency_variance')

    df['next_author'] = df['author'].shift(-1)
    df['next_gap']    = df['gap_mins'].shift(-1).fillna(999)
    df['got_reply']   = (
        (df['next_gap'] <= 10) & (df['next_author'] != df['author'])
    ).astype(int)
    replies_received = df.groupby('author')['got_reply'].sum().rename('replies_received')

    def safe_ratio(r, s): return round(r / s, 3) if s > 0 else float(r) if r > 0 else 1.0
    in_out_ratio = pd.Series({
        a: safe_ratio(replies_received.get(a, 0), replies_sent.get(a, 0))
        for a in df['author'].unique()
    }, name='in_out_ratio')

    ack_starters = ['yes', 'yeah', 'yep', 'ok', 'okay', 'sure', 'great', 'nice',
                    'good', 'perfect', 'exactly', 'right', 'true', 'agree',
                    'thanks', 'thank', 'awesome', 'cool', 'got it', 'noted',
                    'of course', 'absolutely', 'definitely', 'makes sense']
    df['is_ack'] = df['body'].apply(
        lambda x: any(str(x).lower().strip().startswith(w) for w in ack_starters)
    ).astype(int)
    ack_rate = df.groupby('author')['is_ack'].mean().rename('acknowledgment_rate')

    emoji_pat = re.compile(
        "[\U0001F300-\U0001F9FF\U00002600-\U000027BF\U0001FA00-\U0001FA9F]+",
        flags=re.UNICODE)
    df['emoji_count'] = df['body'].apply(lambda x: len(emoji_pat.findall(str(x))))
    emoji_ratio       = df.groupby('author')['emoji_count'].mean().rename('emoji_ratio')

    df['is_init']   = (df['gap_mins'] > 30).astype(int)
    initiation_rate = df.groupby('author')['is_init'].mean().rename('initiation_rate')

    features = pd.concat([
        msg_count, avg_length, question_ratio, avg_mentions,
        replies_sent, avg_task, avg_butterfly, avg_sentiment,
        avg_capybara, active_days, new_topic_ratio,
        mattr, latency_var, in_out_ratio,
        ack_rate, emoji_ratio, initiation_rate
    ], axis=1).fillna(0)

    # Always make author a regular column
    features.index.name = 'author'
    features = features.reset_index()

    print(f"  Feature matrix: {features.shape}")
    return features


# ════════════════════════════════════════════════════════════════════════
# RANK-PERCENTILE NORMALIZATION
# ════════════════════════════════════════════════════════════════════════

def rank_normalize(features, cols):
    out = features.copy()
    for col in cols:
        vals    = out[col].values.astype(float)
        out[col] = pd.Series(vals).rank(pct=True).values
    return out


# ════════════════════════════════════════════════════════════════════════
# WEIGHTED ARCHETYPE SCORE FORMULA (from Classification_Archetypes.pdf)
# ════════════════════════════════════════════════════════════════════════

def compute_weighted_scores(features):
    """
    Weighted combination of features per archetype.
    Based on the score formula from Classification_Archetypes.pdf.
    Output: percentage distribution summing to 100% per user.
    """
    def norm(series):
        mn, mx = series.min(), series.max()
        if mx == mn: return pd.Series([0.5] * len(series), index=series.index)
        return (series - mn) / (mx - mn)

    N = norm  # shorthand
    f = features.set_index('author') if 'author' in features.columns else features

    # ── Bee: vocabulary diversity + topic jumping + initiation ─────────────
    bee = (
        0.30 * N(f['mattr'])             +  # vocabulary diversity (MATTR)
        0.25 * N(f['msg_count'])         +  # high volume
        0.25 * N(f['initiation_rate'])   +  # starts conversations
        0.10 * N(f['avg_mentions'])      +  # connects people
        0.10 * N(f['question_ratio'])       # curious, exploratory
    )

    # ── Ant: task focus + reply consistency + active presence ─────────────
    ant = (
        0.30 * N(f['task_focus_score'])          +  # task language
        0.25 * N(f['replies_sent'])              +  # responsive
        0.20 * N(f['active_days'])               +  # consistent presence
        0.15 * (1 - N(f['reply_latency_variance'])) +  # low variance = consistent
        0.10 * N(f['avg_msg_length'])               # structured messages
    )

    # ── Butterfly: emotional language + emoji ────────────────────────────
    butterfly = (
        0.30 * N(f['avg_msg_length'])    +  # writes in depth
        0.25 * N(f['butterfly_score'])   +  # summarizing language
        0.25 * N(f['emoji_ratio'])       +  # emotional/visual
        0.20 * N(f['avg_sentiment'])        # positive tone
    )

    # ── Capybara: acknowledgment + positivity + harmony ───────────────────
    capybara = (
        0.35 * N(f['acknowledgment_rate']) +  # affirms others first
        0.30 * N(f['capybara_score'])      +  # supportive words
        0.20 * N(f['avg_sentiment'])       +  # positive sentiment
        0.15 * N(f['replies_sent'])            # responds to others
    )

    # ── Leech: receives much, contributes little ──────────────────────────
    leech = (
        0.40 * N(f['in_out_ratio'])      +   # receives >> sends
        0.30 * (1 - N(f['msg_count']))   +   # low message count
        0.20 * (1 - N(f['active_days'])) +   # irregular presence
        0.10 * (1 - N(f['replies_sent']))    # doesn't reply
    )

    scores = pd.DataFrame({
        '🐝 Bee'      : bee,
        '🐜 Ant'      : ant,
        '🦋 Butterfly': butterfly,
        '🦫 Capybara' : capybara,
        '🔴 Leech'    : leech
    })

    # Normalize rows so they sum to 1.0 (percentage distribution)
    row_sums = scores.sum(axis=1)
    scores   = scores.div(row_sums, axis=0)

    # Convert to 0-100 percentages
    scores = (scores * 100).round(1)

    # Hard label = highest scoring archetype
    scores['archetype'] = scores[ARCHETYPE_ORDER].idxmax(axis=1)

    # Rename for dashboard
    scores = scores.rename(columns={
        '🐝 Bee'      : 'bee_pct',
        '🐜 Ant'      : 'ant_pct',
        '🦋 Butterfly': 'butterfly_pct',
        '🦫 Capybara' : 'capybara_pct',
        '🔴 Leech'    : 'leech_pct'
    })

    # Always attach author column
    scores = scores.reset_index(drop=True)
    if 'author' in features.columns:
        scores.insert(0, 'author', features['author'].values)

    return scores


# ════════════════════════════════════════════════════════════════════════
# RULE-BASED LABELS (for ML training pseudo ground truth)
# ════════════════════════════════════════════════════════════════════════

def assign_archetype_rules(features):
    def pct(col, p): return features[col].quantile(p)

    def assign(row):
        scores = {
            '🐝 Bee': (
                (row['msg_count']          >= pct('msg_count',          0.75)) * 2 +
                (row['avg_mentions']       >= pct('avg_mentions',       0.50)) * 2 +
                (row['mattr']              >= pct('mattr',              0.60))     +
                (row['initiation_rate']    >= pct('initiation_rate',    0.50))     +
                (row['question_ratio']     >= pct('question_ratio',     0.50))
            ),
            '🐜 Ant': (
                (row['task_focus_score']         >= pct('task_focus_score',         0.50)) * 2 +
                (row['replies_sent']             >= pct('replies_sent',             0.50)) * 2 +
                (row['active_days']              >= pct('active_days',              0.50))     +
                (row['reply_latency_variance']   <= pct('reply_latency_variance',   0.40))
            ),
            '🦋 Butterfly': (
                (row['butterfly_score'] >= pct('butterfly_score', 0.50)) * 2 +
                (row['avg_msg_length']  >= pct('avg_msg_length',  0.75)) * 2 +
                (row['emoji_ratio']     >= pct('emoji_ratio',     0.50))     +
                (row['avg_sentiment']   >= pct('avg_sentiment',   0.50))
            ),
            '🦫 Capybara': (
                (row['acknowledgment_rate'] >= pct('acknowledgment_rate', 0.50)) * 2 +
                (row['capybara_score']      >= pct('capybara_score',      0.50)) * 2 +
                (row['avg_sentiment']       >= pct('avg_sentiment',       0.75))     +
                (row['replies_sent']        >= pct('replies_sent',        0.50))
            ),
            '🔴 Leech': (
                (row['in_out_ratio']  >= pct('in_out_ratio',  0.75)) * 2 +
                (row['msg_count']     <= pct('msg_count',     0.25)) * 2 +
                (row['active_days']   <= pct('active_days',   0.25))     +
                (row['replies_sent']  <= pct('replies_sent',  0.25))
            )
        }
        return max(scores, key=scores.get), scores

    results = features.apply(lambda r: assign(r), axis=1)
    return results.apply(lambda x: x[0]), results.apply(lambda x: x[1])


# ════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════════════════════════════════════════════

print("\n🐝  COIN Hybrid Archetype Classifier — v2 (Weighted + Distribution)")
print("=" * 65)

# ── STEP 1: Extract features ──────────────────────────────────────────────
slack_raw   = extract_features("slack_clean.json",    "Slack (Chatterjee 2020)")
nankani_raw = extract_features("nankani_clean.json",  "CODERS WhatsApp (Nankani 2020)")
wa_raw      = extract_features("whatsapp_clean.json", "Your Team WhatsApp")
# Ensure author is a regular column, not the index
for _df in [slack_raw, nankani_raw, wa_raw]:
    if _df.index.name == 'author' or (hasattr(_df.index, 'name') and _df.index.name):
        _df.reset_index(inplace=True)
    if 'author' not in _df.columns and 'index' in _df.columns:
        _df.rename(columns={'index': 'author'}, inplace=True)

# ── STEP 2: Rank-normalize within each dataset ────────────────────────────
print("\n📐  Rank-normalizing features within each dataset...")
slack_norm   = rank_normalize(slack_raw,   FEATURE_COLS)
nankani_norm = rank_normalize(nankani_raw, FEATURE_COLS)
wa_norm      = rank_normalize(wa_raw,      FEATURE_COLS)
print("  ✅ Done")

# ── STEP 3: Rule-based labels on training data ────────────────────────────
print("\n📋  Applying COIN rules to training datasets...")
slack_norm['archetype'],   slack_rs   = assign_archetype_rules(slack_norm)
nankani_norm['archetype'], nankani_rs = assign_archetype_rules(nankani_norm)

for name, data in [("Slack", slack_norm), ("Nankani", nankani_norm)]:
    print(f"\n  {name} distribution:")
    for arch, cnt in data['archetype'].value_counts().items():
        print(f"    {arch:<20} {cnt:>5}  ({cnt/len(data)*100:.1f}%)")

# ── STEP 4: Combine training data ─────────────────────────────────────────
print("\n🔗  Combining training datasets...")
slack_norm['dataset']   = 'slack'
nankani_norm['dataset'] = 'nankani'
combined = pd.concat([
    slack_norm[FEATURE_COLS + ['archetype', 'dataset']],
    nankani_norm[FEATURE_COLS + ['archetype', 'dataset']]
], ignore_index=True)
print(f"  Combined: {len(combined):,} users  "
      f"({len(slack_norm):,} Slack + {len(nankani_norm):,} Nankani)")

# ── STEP 5: Train Random Forest ───────────────────────────────────────────
print("\n🤖  Training Random Forest...")
le = LabelEncoder()
X  = combined[FEATURE_COLS].values
y  = le.fit_transform(combined['archetype'])

cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    RandomForestClassifier(n_estimators=300, max_depth=12,
                           min_samples_leaf=3, class_weight='balanced',
                           random_state=42),
    X, y, cv=cv, scoring='f1_macro'
)
print(f"  CV F1 (5-fold): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

rf = RandomForestClassifier(n_estimators=300, max_depth=12,
                             min_samples_leaf=3, class_weight='balanced',
                             random_state=42)
rf.fit(X, y)
print(f"  Train accuracy: {rf.score(X, y)*100:.1f}%")

print("\n  Feature importance (top 8):")
for feat, imp in sorted(zip(FEATURE_COLS, rf.feature_importances_),
                         key=lambda x: x[1], reverse=True)[:8]:
    print(f"    {feat:<30} {imp:.3f}  {'█' * int(imp*60)}")

# ── STEP 6: Predict for WhatsApp team ────────────────────────────────────
print("\n🔮  Predicting archetypes for WhatsApp team...")
# Save author list before any transformation loses it
wa_authors  = wa_raw['author'].tolist() if 'author' in wa_raw.columns else list(range(len(wa_raw)))
X_wa        = wa_norm[FEATURE_COLS].values
y_pred      = rf.predict(X_wa)
y_pred_prob = rf.predict_proba(X_wa)

wa_raw['archetype_ml'] = le.inverse_transform(y_pred)
wa_raw['ml_confidence']= [round(p.max()*100,1) for p in y_pred_prob]

# ── STEP 7: Rule-based for WhatsApp ──────────────────────────────────────
wa_rule_labels, wa_rule_scores = assign_archetype_rules(wa_raw)
wa_raw['archetype_rule'] = wa_rule_labels

# ── STEP 8: Weighted percentage distribution ──────────────────────────────
print("\n📊  Computing weighted archetype percentage distribution...")
wa_raw_for_scores = wa_raw.copy()
wa_raw_for_scores['author'] = wa_authors
wa_scores = compute_weighted_scores(wa_raw_for_scores)
wa_scores['author'] = wa_authors

# ── STEP 9: Hybrid vote (50% ML + 50% rules) → final archetype ───────────
print("🗳️   Hybrid voting...")

def hybrid_vote(rule_scores_dict, ml_probs, le):
    ml_dict    = dict(zip(le.classes_, ml_probs))
    rule_total = sum(rule_scores_dict.values())
    rule_dict  = ({k: v/rule_total for k, v in rule_scores_dict.items()}
                  if rule_total > 0 else {k: 0.2 for k in rule_scores_dict})
    hybrid     = {a: 0.5*ml_dict.get(a,0) + 0.5*rule_dict.get(a,0)
                  for a in ARCHETYPE_ORDER}
    winner     = max(hybrid, key=hybrid.get)
    return winner, round(hybrid[winner]*100, 1)

final_archetypes   = []
final_confidences  = []
for i, (_, row) in enumerate(wa_raw.iterrows()):
    idx    = wa_raw.index.get_loc(row.name)
    winner, conf = hybrid_vote(wa_rule_scores.iloc[idx], y_pred_prob[idx], le)
    final_archetypes.append(winner)
    final_confidences.append(conf)

wa_raw['archetype']  = final_archetypes
wa_raw['confidence'] = final_confidences

# ── STEP 10: Build final results directly ────────────────────────────────
wa_results = wa_raw_for_scores.copy()
wa_results['archetype']  = final_archetypes
wa_results['confidence'] = final_confidences
for col in ['bee_pct','ant_pct','butterfly_pct','capybara_pct','leech_pct']:
    wa_results[col] = wa_scores[col].values

# Keep dashboard-compatible score column names too
wa_results['bee_score']         = wa_results['bee_pct']
wa_results['ant_score']         = wa_results['ant_pct']
wa_results['butterfly_score_n'] = wa_results['butterfly_pct']
wa_results['capybara_score_n']  = wa_results['capybara_pct']
wa_results['leech_risk']        = wa_results['leech_pct']

wa_results.to_json("wa_results.json", orient="records", lines=True)

# Save training reference
slack_save   = slack_raw[FEATURE_COLS].copy()
slack_save['author']    = slack_raw.index if 'author' not in slack_raw.columns else slack_raw['author'].values
slack_save['archetype'] = slack_norm['archetype'].values
slack_save['dataset']   = 'slack'

nankani_save = nankani_raw[FEATURE_COLS].copy()
nankani_save['author']    = nankani_raw.index if 'author' not in nankani_raw.columns else nankani_raw['author'].values
nankani_save['archetype'] = nankani_norm['archetype'].values
nankani_save['dataset']   = 'nankani'

combined_with_author = pd.concat([slack_save, nankani_save], ignore_index=True)
combined_with_author.to_json("training_results.json", orient="records", lines=True)

# ── STEP 11: Final summary ────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  ✅  FINAL RESULTS — Archetype Distribution (not just a label!)")
print("=" * 65)
print(f"\n  {'Name':<12} {'Archetype':<22} {'🐝':>6} {'🐜':>6} "
      f"{'🦋':>6} {'🦫':>6} {'🔴':>6}  Conf")
print(f"  {'-'*12} {'-'*22} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6}  {'-'*4}")

for _, row in wa_results.sort_values('msg_count', ascending=False).iterrows():
    match = "✅" if row['archetype_rule'] == row['archetype_ml'] else "〰️"
    print(
        f"  {row['author']:<12} "
        f"{row['archetype']:<22} "
        f"{row['bee_pct']:>5.1f}% "
        f"{row['ant_pct']:>5.1f}% "
        f"{row['butterfly_pct']:>5.1f}% "
        f"{row['capybara_pct']:>5.1f}% "
        f"{row['leech_pct']:>5.1f}%  "
        f"{match} {row['confidence']:.0f}%"
    )

print(f"\n  Training: Slack + Nankani 2020 → {len(combined):,} users")
print(f"  CV F1:    {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print(f"\n  ✅  Saved → wa_results.json")
print(f"  ✅  Saved → training_results.json")
print(f"\n  Run:  streamlit run dashboard.py")