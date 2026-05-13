import pandas as pd
import numpy as np

features = pd.read_json("wa_features.json", lines=True)
n = len(features)
print(f"Classifying {n} team members into COIN archetypes")

def pct(col, p):
    return features[col].quantile(p)

# Adaptive thresholds for small groups
high = 0.67 if n < 15 else 0.75
med  = 0.40
low  = 0.33

def assign_archetype(row):
    msgs        = row['msg_count']
    mentions    = row['avg_mentions']
    q_ratio     = row['question_ratio']
    new_topics  = row['new_topic_ratio']
    task        = row['task_focus_score']
    replies     = row['replies_sent']
    reply_speed = row['avg_reply_speed_mins']
    butterfly   = row['butterfly_score']
    capybara    = row['capybara_score']
    sentiment   = row['avg_sentiment']
    active      = row['active_days']
    length      = row['avg_msg_length']

    # Score each archetype
    scores = {
        '🐝 Bee': (
            (msgs >= pct('msg_count', high)) * 2 +
            (mentions >= pct('avg_mentions', med)) * 2 +
            (q_ratio >= pct('question_ratio', med)) +
            (new_topics >= pct('new_topic_ratio', med))
        ),
        '🐜 Ant': (
            (task >= pct('task_focus_score', med)) * 2 +
            (replies >= pct('replies_sent', med)) * 2 +
            (reply_speed <= pct('avg_reply_speed_mins', high) and reply_speed > 0) +
            (active >= pct('active_days', med))
        ),
        '🦋 Butterfly': (
            (butterfly >= pct('butterfly_score', med)) * 2 +
            (length >= pct('avg_msg_length', high)) * 2 +
            (mentions >= pct('avg_mentions', med)) +
            (sentiment >= pct('avg_sentiment', med))
        ),
        '🦫 Capybara': (
            (capybara >= pct('capybara_score', med)) * 2 +
            (sentiment >= pct('avg_sentiment', high)) * 2 +
            (replies >= pct('replies_sent', med)) +
            (msgs >= pct('msg_count', low))
        ),
        '🔴 Leech': (
            (msgs <= pct('msg_count', low)) * 2 +
            (active <= pct('active_days', low)) * 2 +
            (replies <= pct('replies_sent', low)) +
            (mentions <= pct('avg_mentions', low))
        )
    }

    return max(scores, key=scores.get)

def map_scores(row):
    """Return normalized 0-100 score for each archetype dimension."""
    def norm(val, col):
        mn = features[col].min()
        mx = features[col].max()
        if mx == mn: return 50.0
        return round((val - mn) / (mx - mn) * 100, 1)

    return pd.Series({
        'bee_score'       : round((norm(row['msg_count'], 'msg_count') +
                                   norm(row['avg_mentions'], 'avg_mentions') +
                                   norm(row['question_ratio'], 'question_ratio')) / 3, 1),
        'ant_score'       : round((norm(row['task_focus_score'], 'task_focus_score') +
                                   norm(row['replies_sent'], 'replies_sent') +
                                   norm(row['active_days'], 'active_days')) / 3, 1),
        'butterfly_score_n': round((norm(row['butterfly_score'], 'butterfly_score') +
                                    norm(row['avg_msg_length'], 'avg_msg_length')) / 2, 1),
        'capybara_score_n': round((norm(row['capybara_score'], 'capybara_score') +
                                   norm(row['avg_sentiment'], 'avg_sentiment')) / 2, 1),
        'leech_risk'      : round(100 - (norm(row['msg_count'], 'msg_count') +
                                         norm(row['active_days'], 'active_days') +
                                         norm(row['replies_sent'], 'replies_sent')) / 3, 1)
    })

features['archetype'] = features.apply(assign_archetype, axis=1)
scores_df = features.apply(map_scores, axis=1)
features  = pd.concat([features, scores_df], axis=1)

features.to_json("wa_results.json", orient="records", lines=True)

print("\n✅ Archetype distribution:")
print(features['archetype'].value_counts())
print("\nFull results:")
cols = ['author','archetype','msg_count','active_days',
        'bee_score','ant_score','butterfly_score_n',
        'capybara_score_n','leech_risk']
print(features[cols].sort_values('msg_count', ascending=False).to_string())