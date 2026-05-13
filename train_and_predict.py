"""
train_and_predict.py
────────────────────
Hybrid COIN Archetype Classifier — Multi-Dataset Training

Training data (combined):
  1. Slack Developer Chats (Chatterjee et al., MSR 2020)
     — 60,000 messages · 1,639 software engineers
     — Python, Clojure, Elm, Racket communities

  2. Nankani 2020 — CODERS WhatsApp Group (TSEC Mumbai)
     — ~20,000 messages · real CS/IT students
     — Discussing coding, GCP, GitHub, web development, hackathons

Test / Demo data:
  — Your team's WhatsApp group (5 members, 324 messages)

Method:
  — Rank-normalize features within each dataset (fixes scale mismatch)
  — Rule-based COIN labels → pseudo ground truth
  — Random Forest (300 trees, balanced classes)
  — 5-fold cross-validation
  — Hybrid vote: 50% ML + 50% rules for final prediction
"""

import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

FEATURE_COLS = [
    'msg_count', 'avg_msg_length', 'question_ratio', 'avg_mentions',
    'replies_sent', 'task_focus_score', 'butterfly_score',
    'avg_sentiment', 'capybara_score', 'active_days', 'new_topic_ratio'
]

# ════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ════════════════════════════════════════════════════════════════════════

def extract_features(json_path, label="dataset"):
    print(f"\n{'='*55}")
    print(f"  Extracting features: {label}")
    print(f"{'='*55}")

    df = pd.read_json(json_path, lines=True)
    df['datetime'] = pd.to_datetime(df['datetime'], unit='ms', errors='coerce')
    df = df.dropna(subset=['datetime']).sort_values('datetime').reset_index(drop=True)
    print(f"  Messages: {len(df):,}  |  Users: {df['author'].nunique():,}")

    msg_count  = df.groupby('author').size().rename('msg_count')
    avg_length = df.groupby('author')['body'].apply(
        lambda x: x.str.len().mean()).rename('avg_msg_length')

    df['is_question']   = df['body'].str.contains(r'\?', regex=True).astype(int)
    question_ratio      = df.groupby('author')['is_question'].mean().rename('question_ratio')

    df['mention_count'] = df['body'].str.count(r'@\w+')
    avg_mentions        = df.groupby('author')['mention_count'].mean().rename('avg_mentions')

    df['prev_author']   = df['author'].shift(1)
    df['prev_time']     = df['datetime'].shift(1)
    df['gap_mins']      = (df['datetime'] - df['prev_time']).dt.total_seconds() / 60
    df['is_reply']      = (
        (df['gap_mins'] <= 10) & (df['prev_author'] != df['author'])
    ).astype(int)
    replies_sent = df.groupby('author')['is_reply'].sum().rename('replies_sent')

    task_words = ['done', 'finished', 'completed', 'sent', 'here', 'attached',
                  'will do', 'ok', 'okay', 'sure', 'deadline', 'file', 'link',
                  'submitted', 'ready', 'push', 'commit', 'fixed', 'update',
                  'works', 'working', 'solved', 'merged', 'deployed', 'closed',
                  'kiya', 'ho gaya', 'kar diya']   # Hindi task words from Nankani
    df['task_score'] = df['body'].str.lower().apply(
        lambda x: sum(w in x for w in task_words))
    avg_task = df.groupby('author')['task_score'].mean().rename('task_focus_score')

    butterfly_words = ['so basically', 'in summary', 'to summarize', 'in other words',
                       'what i mean', 'let me explain', 'to clarify', 'in short',
                       'the idea is', 'essentially', 'in a nutshell', 'meaning that',
                       'if you want', 'you can', 'you should', 'i suggest', 'try this']
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
                      'thanks alot', 'thanks a lot', 'relax', 'no problem', 'sure']
    df['cap_score'] = df['body'].str.lower().apply(
        lambda x: sum(w in x for w in capybara_words))
    avg_capybara = df.groupby('author')['cap_score'].mean().rename('capybara_score')

    df['date']         = df['datetime'].dt.date
    active_days        = df.groupby('author')['date'].nunique().rename('active_days')

    df['is_new_topic'] = (df['gap_mins'] > 10).astype(int)
    new_topic_ratio    = df.groupby('author')['is_new_topic'].mean().rename('new_topic_ratio')

    features = pd.concat([
        msg_count, avg_length, question_ratio, avg_mentions,
        replies_sent, avg_task, avg_butterfly,
        avg_sentiment, avg_capybara, active_days, new_topic_ratio
    ], axis=1).fillna(0).reset_index()

    print(f"  Feature matrix: {features.shape}")
    return features


# ════════════════════════════════════════════════════════════════════════
# RANK-PERCENTILE NORMALIZATION
# ════════════════════════════════════════════════════════════════════════

def rank_normalize(features, cols):
    """
    Convert each feature to within-group percentile rank (0–1).
    Removes absolute scale differences between datasets.
    Celina's 197 messages → 1.0 (top of her 5-person group)
    Joey's 2196 messages → 1.0 (top of 1639-person Slack group)
    Both correctly identified as high-activity for their context.
    """
    out = features.copy()
    for col in cols:
        vals     = out[col].values.astype(float)
        out[col] = pd.Series(vals).rank(pct=True).values
    return out


# ════════════════════════════════════════════════════════════════════════
# RULE-BASED ARCHETYPE LABELING  (COIN theory — Gloor 2006)
# ════════════════════════════════════════════════════════════════════════

def assign_archetype_rules(features):
    def pct(col, p): return features[col].quantile(p)

    def assign(row):
        scores = {
            '🐝 Bee': (
                (row['msg_count']        >= pct('msg_count',        0.75)) * 2 +
                (row['avg_mentions']     >= pct('avg_mentions',     0.50)) * 2 +
                (row['question_ratio']   >= pct('question_ratio',   0.50))     +
                (row['new_topic_ratio']  >= pct('new_topic_ratio',  0.50))
            ),
            '🐜 Ant': (
                (row['task_focus_score'] >= pct('task_focus_score', 0.50)) * 2 +
                (row['replies_sent']     >= pct('replies_sent',     0.50)) * 2 +
                (row['active_days']      >= pct('active_days',      0.50))     +
                (row['msg_count']        >= pct('msg_count',        0.50))
            ),
            '🦋 Butterfly': (
                (row['butterfly_score']  >= pct('butterfly_score',  0.50)) * 2 +
                (row['avg_msg_length']   >= pct('avg_msg_length',   0.75)) * 2 +
                (row['avg_mentions']     >= pct('avg_mentions',     0.50))     +
                (row['avg_sentiment']    >= pct('avg_sentiment',    0.50))
            ),
            '🦫 Capybara': (
                (row['capybara_score']   >= pct('capybara_score',   0.50)) * 2 +
                (row['avg_sentiment']    >= pct('avg_sentiment',    0.75)) * 2 +
                (row['replies_sent']     >= pct('replies_sent',     0.50))     +
                (row['msg_count']        >= pct('msg_count',        0.33))
            ),
            '🔴 Leech': (
                (row['msg_count']        <= pct('msg_count',        0.25)) * 2 +
                (row['active_days']      <= pct('active_days',      0.25)) * 2 +
                (row['replies_sent']     <= pct('replies_sent',     0.25))     +
                (row['avg_mentions']     <= pct('avg_mentions',     0.25))
            )
        }
        return max(scores, key=scores.get), scores

    results = features.apply(lambda r: assign(r), axis=1)
    labels  = results.apply(lambda x: x[0])
    scores  = results.apply(lambda x: x[1])
    return labels, scores


# ════════════════════════════════════════════════════════════════════════
# ARCHETYPE DIMENSION SCORES (0-100 for dashboard)
# ════════════════════════════════════════════════════════════════════════

def compute_archetype_scores(features):
    def norm(val, col):
        mn = features[col].min()
        mx = features[col].max()
        if mx == mn: return 50.0
        return round(float(np.clip((val - mn) / (mx - mn) * 100, 0, 100)), 1)

    def score_row(row):
        return pd.Series({
            'bee_score'        : round((norm(row['msg_count'],       'msg_count') +
                                        norm(row['avg_mentions'],     'avg_mentions') +
                                        norm(row['question_ratio'],   'question_ratio')) / 3, 1),
            'ant_score'        : round((norm(row['task_focus_score'], 'task_focus_score') +
                                        norm(row['replies_sent'],     'replies_sent') +
                                        norm(row['active_days'],      'active_days')) / 3, 1),
            'butterfly_score_n': round((norm(row['butterfly_score'],  'butterfly_score') +
                                        norm(row['avg_msg_length'],   'avg_msg_length')) / 2, 1),
            'capybara_score_n' : round((norm(row['capybara_score'],   'capybara_score') +
                                        norm(row['avg_sentiment'],    'avg_sentiment')) / 2, 1),
            'leech_risk'       : round(100 - (norm(row['msg_count'],  'msg_count') +
                                              norm(row['active_days'],'active_days') +
                                              norm(row['replies_sent'],'replies_sent')) / 3, 1)
        })
    return features.apply(score_row, axis=1)


# ════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════════════════════════════════════════════

print("\n🐝  COIN Hybrid Archetype Classifier — Multi-Dataset Training")
print("=" * 60)

# ── STEP 1: Extract features from all three datasets ─────────────────────
slack_raw    = extract_features("slack_clean.json",    "Slack Developer Chat (Chatterjee 2020)")
nankani_raw  = extract_features("nankani_clean.json",  "CODERS WhatsApp — TSEC Mumbai (Nankani 2020)")
wa_raw       = extract_features("whatsapp_clean.json", "Your Team WhatsApp")

# ── STEP 2: Rank-normalize WITHIN each dataset ───────────────────────────
print("\n📐  Rank-normalizing features within each dataset...")
slack_norm   = rank_normalize(slack_raw,   FEATURE_COLS)
nankani_norm = rank_normalize(nankani_raw, FEATURE_COLS)
wa_norm      = rank_normalize(wa_raw,      FEATURE_COLS)
print("  ✅ Each feature = within-group percentile rank (0–1)")

# ── STEP 3: Rule-based COIN labels on both training datasets ─────────────
print("\n📋  Applying COIN rules to training datasets...")
slack_norm['archetype'],   slack_rule_scores   = assign_archetype_rules(slack_norm)
nankani_norm['archetype'], nankani_rule_scores = assign_archetype_rules(nankani_norm)

print("\n  Slack archetype distribution:")
slack_dist = slack_norm['archetype'].value_counts()
for arch, count in slack_dist.items():
    print(f"    {arch:<20} {count:>5}  ({count/len(slack_norm)*100:.1f}%)")

print("\n  Nankani 2020 archetype distribution:")
nankani_dist = nankani_norm['archetype'].value_counts()
for arch, count in nankani_dist.items():
    print(f"    {arch:<20} {count:>5}  ({count/len(nankani_norm)*100:.1f}%)")

# ── STEP 4: Combine training datasets ────────────────────────────────────
print("\n🔗  Combining Slack + Nankani 2020 training data...")
slack_norm['dataset']   = 'slack'
nankani_norm['dataset'] = 'nankani'

combined = pd.concat([
    slack_norm[FEATURE_COLS + ['archetype', 'dataset']],
    nankani_norm[FEATURE_COLS + ['archetype', 'dataset']]
], ignore_index=True)

print(f"  Combined training users: {len(combined):,}")
print(f"    Slack:    {len(slack_norm):,} users")
print(f"    Nankani:  {len(nankani_norm):,} users")
print(f"\n  Combined archetype distribution:")
combined_dist = combined['archetype'].value_counts()
for arch, count in combined_dist.items():
    bar = '█' * int(count / len(combined) * 40)
    print(f"    {arch:<20} {count:>5}  ({count/len(combined)*100:.1f}%)  {bar}")

# ── STEP 5: Train Random Forest on combined data ──────────────────────────
print("\n🤖  Training Random Forest on combined dataset...")

le = LabelEncoder()
X  = combined[FEATURE_COLS].values
y  = le.fit_transform(combined['archetype'])

cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf_cv  = RandomForestClassifier(
    n_estimators=300, max_depth=12,
    min_samples_leaf=3, class_weight='balanced', random_state=42
)
cv_scores = cross_val_score(rf_cv, X, y, cv=cv, scoring='f1_macro')
print(f"  Cross-validation F1 (5-fold): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

rf = RandomForestClassifier(
    n_estimators=300, max_depth=12,
    min_samples_leaf=3, class_weight='balanced', random_state=42
)
rf.fit(X, y)
print(f"  Training accuracy: {rf.score(X, y)*100:.1f}%")

print("\n  Feature importance (top 6):")
importances = sorted(
    zip(FEATURE_COLS, rf.feature_importances_), key=lambda x: x[1], reverse=True
)
for feat, imp in importances[:6]:
    bar = '█' * int(imp * 60)
    print(f"    {feat:<28} {imp:.3f}  {bar}")

# ── STEP 6: Rule-based labels on WhatsApp ────────────────────────────────
print("\n📋  Applying COIN rules to WhatsApp team (within-group)...")
wa_rule_labels, wa_rule_scores = assign_archetype_rules(wa_raw)
wa_raw['archetype_rule'] = wa_rule_labels

print("\n  Rule-based predictions:")
for _, row in wa_raw.sort_values('msg_count', ascending=False).iterrows():
    print(f"    {row['author']:<12} → {row['archetype_rule']}")

# ── STEP 7: ML predictions on normalized WhatsApp ────────────────────────
print("\n🔮  ML predictions on WhatsApp (rank-normalized)...")
X_wa        = wa_norm[FEATURE_COLS].values
y_pred      = rf.predict(X_wa)
y_pred_prob = rf.predict_proba(X_wa)

wa_raw['archetype_ml']  = le.inverse_transform(y_pred)
wa_raw['ml_confidence'] = [round(p.max() * 100, 1) for p in y_pred_prob]

print("\n  ML predictions:")
for _, row in wa_raw.sort_values('msg_count', ascending=False).iterrows():
    print(f"    {row['author']:<12} → {row['archetype_ml']}  "
          f"(conf: {row['ml_confidence']:.0f}%)")

# ── STEP 8: Hybrid voting (50% ML + 50% Rules) ───────────────────────────
print("\n🗳️   Hybrid voting (50% ML + 50% Rules)...")

archetype_order = ['🐝 Bee', '🐜 Ant', '🦋 Butterfly', '🦫 Capybara', '🔴 Leech']

def hybrid_vote(rule_scores_dict, ml_probs, le):
    ml_prob_dict   = dict(zip(le.classes_, ml_probs))
    rule_total     = sum(rule_scores_dict.values())
    rule_prob_dict = ({k: v / rule_total for k, v in rule_scores_dict.items()}
                     if rule_total > 0 else {k: 0.2 for k in rule_scores_dict})
    hybrid = {
        arch: 0.50 * ml_prob_dict.get(arch, 0) + 0.50 * rule_prob_dict.get(arch, 0)
        for arch in archetype_order
    }
    winner = max(hybrid, key=hybrid.get)
    return winner, round(hybrid[winner] * 100, 1)

wa_final_archetypes  = []
wa_final_confidences = []

for i, (_, row) in enumerate(wa_raw.iterrows()):
    idx    = wa_raw.index.get_loc(row.name)
    rule_sc = wa_rule_scores.iloc[idx]
    winner, conf = hybrid_vote(rule_sc, y_pred_prob[idx], le)
    wa_final_archetypes.append(winner)
    wa_final_confidences.append(conf)

wa_raw['archetype']  = wa_final_archetypes
wa_raw['confidence'] = wa_final_confidences

# ── STEP 9: Archetype dimension scores ───────────────────────────────────
scores_df  = compute_archetype_scores(wa_raw)
wa_results = pd.concat([wa_raw, scores_df], axis=1)

# Ensure all columns dashboard needs are present
for col in ['avg_reply_speed_mins', 'question_ratio', 'task_focus_score']:
    if col not in wa_results.columns:
        wa_results[col] = 0

# ── STEP 10: Save all outputs ─────────────────────────────────────────────
wa_results.to_json("wa_results.json", orient="records", lines=True)

# Save combined training reference
combined_scores = compute_archetype_scores(
    pd.concat([slack_raw[FEATURE_COLS + ['author']].assign(dataset='slack'),
               nankani_raw[FEATURE_COLS + ['author']].assign(dataset='nankani')],
              ignore_index=True)
)
combined_full = pd.concat([
    slack_raw[FEATURE_COLS + ['author']].assign(
        archetype=slack_norm['archetype'].values, dataset='slack'),
    nankani_raw[FEATURE_COLS + ['author']].assign(
        archetype=nankani_norm['archetype'].values, dataset='nankani')
], ignore_index=True)
combined_full.to_json("training_results.json", orient="records", lines=True)

# ── STEP 11: Final summary ────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  ✅  FINAL RESULTS — Your WhatsApp Team (Multi-Dataset Hybrid)")
print("=" * 65)
print(f"  {'Name':<12} {'Hybrid':<22} {'Rule':<22} {'ML':<22} {'Conf':>5}")
print(f"  {'-'*12} {'-'*22} {'-'*22} {'-'*22} {'-'*5}")

for _, row in wa_results.sort_values('msg_count', ascending=False).iterrows():
    match = "✅" if row['archetype_rule'] == row['archetype_ml'] else "〰️"
    print(
        f"  {row['author']:<12} "
        f"{row['archetype']:<22} "
        f"{row['archetype_rule']:<22} "
        f"{row['archetype_ml']:<22} "
        f"{match} {row['confidence']:.0f}%"
    )

print(f"\n  Final archetype distribution:")
for arch, count in wa_results['archetype'].value_counts().items():
    print(f"    {arch}  →  {count} member{'s' if count > 1 else ''}")

print(f"\n  Training summary:")
print(f"    Datasets used:     Slack (Chatterjee 2020) + Nankani 2020")
print(f"    Training users:    {len(combined):,} ({len(slack_norm):,} Slack + {len(nankani_norm):,} Nankani)")
print(f"    CV F1 score:       {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print(f"\n  ✅  Saved → wa_results.json")
print(f"  ✅  Saved → training_results.json")
print(f"\n  Run:  streamlit run dashboard.py")