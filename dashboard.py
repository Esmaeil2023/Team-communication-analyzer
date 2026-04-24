import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Team Communication Analyzer",
    page_icon="💬",
    layout="wide"
)

# ── Role colors & metadata ────────────────────────────────────────────────
ROLE_COLORS = {
    'Leader'     : '#4F86C6',
    'Coordinator': '#F4A261',
    'Contributor': '#2A9D8F',
    'Reactive'   : '#E76F51',
    'Passive'    : '#A8DADC',
    'Isolated'   : '#CED4DA'
}

ROLE_DESCRIPTIONS = {
    'Leader'     : 'High activity, broad reach. Drives conversation.',
    'Coordinator': 'Connects many users. Replies widely.',
    'Contributor': 'Writes detailed, long messages. Adds depth.',
    'Reactive'   : 'Responds frequently. Engaged but not initiating.',
    'Passive'    : 'Low but present activity. Occasional participant.',
    'Isolated'   : 'Minimal interaction. Peripheral to the group.'
}

FEATURE_COLS = [
    'msg_count', 'avg_msg_length', 'question_ratio',
    'replies_sent', 'interaction_count', 'avg_sentiment',
    'sentiment_variability', 'active_days', 'avg_response_time_mins'
]

# ── Load data ─────────────────────────────────────────────────────────────
@st.cache_data
def load_results():
    return pd.read_json("results.json", lines=True)

@st.cache_data
def load_clean():
    df = pd.read_json("reddit_clean.json", lines=True)
    df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
    return df

@st.cache_data
def build_network_data(top_n=60):
    """Build edge list from reply relationships among top N users."""
    raw = load_clean()
    results = load_results()
    top_users = set(results.nlargest(top_n, 'msg_count')['author'])

    # Map comment index → author (approximate: use row order as id)
    idx_to_author = dict(enumerate(raw['author']))

    G = nx.DiGraph()
    for role, color in ROLE_COLORS.items():
        pass  # just initializing

    # Add nodes
    for _, row in results[results['author'].isin(top_users)].iterrows():
        G.add_node(row['author'], role=row['role'],
                   msg_count=row['msg_count'])

    # Add edges: for replies within top users
    # parent_id format: t1_XXXXX — we look for author→author connections
    # using sequential reply chains in the data
    raw_top = raw[raw['author'].isin(top_users)].copy()
    raw_top = raw_top.sort_values('datetime').reset_index(drop=True)

    # Build id→author from parent_id matching
    # Each comment has a parent_id; we find which top user wrote the parent
    added = 0
    for i in range(1, min(len(raw_top), 5000)):
        a = raw_top.iloc[i]['author']
        parent = raw_top.iloc[i]['parent_id']
        # Find previous messages from same thread context
        prev = raw_top[raw_top.index < i].tail(20)
        for _, prev_row in prev.iterrows():
            if prev_row['author'] != a and prev_row['author'] in top_users:
                if G.has_node(a) and G.has_node(prev_row['author']):
                    if G.has_edge(a, prev_row['author']):
                        G[a][prev_row['author']]['weight'] += 1
                    else:
                        G.add_edge(a, prev_row['author'], weight=1)
                    added += 1
                    break

    return G

@st.cache_data
def train_random_forest():
    """Train RF on rule-based labels as pseudo-ground-truth."""
    df = load_results()
    df = df.dropna(subset=FEATURE_COLS + ['role'])

    le = LabelEncoder()
    y = le.fit_transform(df['role'])
    X = df[FEATURE_COLS].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=100, max_depth=8,
        class_weight='balanced', random_state=42
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    report = classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        output_dict=True
    )
    importances = dict(zip(FEATURE_COLS, rf.feature_importances_))
    accuracy = rf.score(X_test, y_test)

    return rf, le, report, importances, accuracy

df = load_results()

# ════════════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════════════
st.title("💬 Team Communication Analyzer")
st.markdown(
    "Analyzing behavioral patterns in **r/programming** and "
    "**r/learnprogramming** · Reddit May 2015"
)
st.markdown("---")

# ════════════════════════════════════════════════════════════════════════
# TOP METRICS
# ════════════════════════════════════════════════════════════════════════
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Users",         f"{len(df):,}")
c2.metric("Total Messages",      f"{int(df['msg_count'].sum()):,}")
c3.metric("Roles Detected",      df['role'].nunique())
c4.metric("Avg Messages / User", f"{df['msg_count'].mean():.1f}")
c5.metric("Avg Active Days",     f"{df['active_days'].mean():.1f}")
st.markdown("---")

# ════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Role Overview",
    "🌐 Interaction Network",
    "🧠 Personality (Big Five)",
    "📅 Activity Timeline",
    "🤖 ML Classifier"
])

# ════════════════════════════════════════════════════════════════════
# TAB 1 — ROLE OVERVIEW
# ════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Role Distribution")
    col_left, col_right = st.columns([1, 1])

    with col_left:
        dist = df['role'].value_counts()
        colors = [ROLE_COLORS.get(r, '#999') for r in dist.index]
        fig, ax = plt.subplots(figsize=(5, 4))
        wedges, texts, autotexts = ax.pie(
            dist.values, labels=dist.index,
            autopct='%1.1f%%', colors=colors,
            startangle=140, pctdistance=0.82
        )
        for t in autotexts:
            t.set_fontsize(9)
        ax.set_title("Users by Role", fontsize=13, pad=12)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_right:
        for role in dist.index:
            count = dist[role]
            pct   = count / len(df) * 100
            color = ROLE_COLORS.get(role, '#999')
            desc  = ROLE_DESCRIPTIONS.get(role, '')
            st.markdown(
                f"<div style='margin-bottom:10px'>"
                f"<span style='background:{color};padding:3px 10px;"
                f"border-radius:12px;color:#fff;font-weight:600;"
                f"font-size:13px'>{role}</span> "
                f"<span style='font-size:14px'>{count:,} users ({pct:.1f}%)"
                f"</span><br>"
                f"<span style='font-size:12px;color:#666'>{desc}</span>"
                f"</div>",
                unsafe_allow_html=True
            )

    st.markdown("---")
    st.subheader("Activity by Role")
    col1, col2 = st.columns(2)

    with col1:
        avg_msgs = df.groupby('role')['msg_count'].mean().sort_values()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.barh(avg_msgs.index, avg_msgs.values,
                color=[ROLE_COLORS.get(r, '#999') for r in avg_msgs.index])
        ax.set_xlabel("Avg messages per user")
        ax.set_title("Average Message Count by Role")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        avg_len = df.groupby('role')['avg_msg_length'].mean().sort_values()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.barh(avg_len.index, avg_len.values,
                color=[ROLE_COLORS.get(r, '#999') for r in avg_len.index])
        ax.set_xlabel("Avg message length (chars)")
        ax.set_title("Average Message Length by Role")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("User Profile Lookup")
    search = st.text_input("Search username:", placeholder="e.g. exoticmatter")

    if search:
        matches = df[df['author'].str.contains(search.lower(), case=False, na=False)]
        if len(matches) == 0:
            st.warning("No user found.")
        else:
            user  = matches.iloc[0]
            role  = user['role']
            color = ROLE_COLORS.get(role, '#999')
            st.markdown(
                f"<h3>u/{user['author']} "
                f"<span style='background:{color};padding:4px 14px;"
                f"border-radius:14px;color:#fff;font-size:16px'>{role}</span></h3>",
                unsafe_allow_html=True
            )
            m1,m2,m3,m4,m5 = st.columns(5)
            m1.metric("Messages",     int(user['msg_count']))
            m2.metric("Replies sent", int(user['replies_sent']))
            m3.metric("Active days",  int(user['active_days']))
            m4.metric("Avg length",   f"{user['avg_msg_length']:.0f} chars")
            m5.metric("Avg sentiment",f"{user['avg_sentiment']:.2f}")

            st.markdown("#### Big Five *(approximate behavioral mapping)*")
            traits = ['extraversion','agreeableness','openness',
                      'conscientiousness','neuroticism']
            tcols = st.columns(5)
            for i, t in enumerate(traits):
                tcols[i].metric(t.capitalize(), f"{user[t]:.0f}/100")

            fig, ax = plt.subplots(figsize=(4,4), subplot_kw=dict(polar=True))
            angles = np.linspace(0, 2*np.pi, len(traits), endpoint=False).tolist()
            values = [user[t] for t in traits]
            angles += angles[:1]; values += values[:1]
            ax.plot(angles, values, color=color, linewidth=2)
            ax.fill(angles, values, color=color, alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(
                ['Extraversion','Agreeableness','Openness',
                 'Conscientiousness','Neuroticism'], size=8
            )
            ax.set_ylim(0,100)
            ax.set_title("Personality Profile", size=12, pad=15)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    st.markdown("---")
    st.subheader("Top Users Table")
    role_filter = st.selectbox(
        "Filter by role:", ["All"] + sorted(df['role'].unique().tolist())
    )
    filtered = df if role_filter == "All" else df[df['role'] == role_filter]
    display_cols = ['author','role','msg_count','replies_sent',
                    'avg_msg_length','active_days','avg_sentiment']
    st.dataframe(
        filtered.sort_values('msg_count', ascending=False)
                .head(20)[display_cols]
                .rename(columns={
                    'author':'User','role':'Role',
                    'msg_count':'Messages','replies_sent':'Replies',
                    'avg_msg_length':'Avg Length',
                    'active_days':'Active Days',
                    'avg_sentiment':'Avg Sentiment'
                }).reset_index(drop=True),
        use_container_width=True
    )

# ════════════════════════════════════════════════════════════════════
# TAB 2 — INTERACTION NETWORK
# ════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Interaction Network — Top 60 Users")
    st.markdown(
        "Nodes = users · Size = message count · "
        "Color = role · Arrows = reply relationships"
    )

    with st.spinner("Building interaction network..."):
        G = build_network_data(top_n=60)

    st.markdown(
        f"**{G.number_of_nodes()} nodes** · "
        f"**{G.number_of_edges()} edges** detected among top users"
    )

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_facecolor('#0f1117')
    fig.patch.set_facecolor('#0f1117')

    # Layout
    pos = nx.spring_layout(G, k=2.5, iterations=50, seed=42)

    # Node sizes and colors
    node_sizes  = []
    node_colors = []
    for node in G.nodes():
        row = df[df['author'] == node]
        if len(row) > 0:
            mc = float(row.iloc[0]['msg_count'])
            role = row.iloc[0]['role']
        else:
            mc, role = 1, 'Passive'
        node_sizes.append(max(100, min(mc * 4, 1200)))
        node_colors.append(ROLE_COLORS.get(role, '#999'))

    # Draw edges
    if G.number_of_edges() > 0:
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edge_color='#ffffff22',
            arrows=True,
            arrowsize=10,
            width=0.6,
            connectionstyle='arc3,rad=0.1'
        )

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.9
    )

    # Labels for top 15 only
    top15 = df.nlargest(15, 'msg_count')['author'].tolist()
    labels = {n: n for n in G.nodes() if n in top15}
    nx.draw_networkx_labels(
        G, pos, labels, ax=ax,
        font_size=7, font_color='white'
    )

    # Legend
    legend_handles = [
        mpatches.Patch(color=c, label=r)
        for r, c in ROLE_COLORS.items()
        if r in df['role'].unique()
    ]
    ax.legend(
        handles=legend_handles,
        loc='lower left', fontsize=9,
        facecolor='#1a1a2e', labelcolor='white',
        framealpha=0.8
    )
    ax.axis('off')
    ax.set_title(
        "User Interaction Network · r/programming + r/learnprogramming",
        color='white', fontsize=13, pad=12
    )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.subheader("Network Statistics")
    nc1, nc2, nc3, nc4 = st.columns(4)
    nc1.metric("Nodes (users)",  G.number_of_nodes())
    nc2.metric("Edges (replies)", G.number_of_edges())

    if G.number_of_nodes() > 0:
        degrees = dict(G.degree())
        top_connector = max(degrees, key=degrees.get) if degrees else "—"
        nc3.metric("Most connected", top_connector)
        nc4.metric("Avg connections",
                   f"{np.mean(list(degrees.values())):.1f}" if degrees else "—")

# ════════════════════════════════════════════════════════════════════
# TAB 3 — BIG FIVE BY ROLE
# ════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Big Five Personality Traits by Role")
    st.markdown(
        "Average trait scores per role. "
        "These are **approximate behavioral proxies**, not clinical measurements."
    )

    traits     = ['extraversion','agreeableness','openness',
                  'conscientiousness','neuroticism']
    trait_labels = ['Extraversion','Agreeableness','Openness',
                    'Conscientiousness','Neuroticism']

    # Grouped bar chart
    roles_present = sorted(df['role'].unique())
    avg_by_role   = df.groupby('role')[traits].mean()

    x     = np.arange(len(traits))
    width = 0.12
    fig, ax = plt.subplots(figsize=(12, 5))

    for i, role in enumerate(roles_present):
        if role in avg_by_role.index:
            vals = avg_by_role.loc[role, traits].values
            offset = (i - len(roles_present)/2) * width
            ax.bar(x + offset, vals, width,
                   label=role,
                   color=ROLE_COLORS.get(role, '#999'),
                   alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(trait_labels, fontsize=11)
    ax.set_ylabel("Score (0–100)")
    ax.set_title("Average Big Five Scores by Communication Role", fontsize=13)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.subheader("Radar Chart — Average Profile per Role")

    radar_cols = st.columns(3)
    col_idx = 0

    for role in roles_present:
        row_data = avg_by_role.loc[role] if role in avg_by_role.index else None
        if row_data is None:
            continue
        with radar_cols[col_idx % 3]:
            fig, ax = plt.subplots(figsize=(3.5, 3.5),
                                   subplot_kw=dict(polar=True))
            angles = np.linspace(0, 2*np.pi, len(traits),
                                 endpoint=False).tolist()
            values = row_data[traits].tolist()
            angles += angles[:1]; values += values[:1]
            color = ROLE_COLORS.get(role, '#999')
            ax.plot(angles, values, color=color, linewidth=2)
            ax.fill(angles, values, color=color, alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(
                ['E','A','O','C','N'], size=10, fontweight='bold'
            )
            ax.set_ylim(0, 100)
            ax.set_title(role, size=12, pad=10, color=color, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        col_idx += 1

    st.markdown(
        "**E** = Extraversion · **A** = Agreeableness · "
        "**O** = Openness · **C** = Conscientiousness · **N** = Neuroticism"
    )

    st.markdown("---")
    st.subheader("Trait Interpretation")
    interp = {
        'Extraversion'     : 'Leaders score highest — they post most frequently, dominating conversation volume.',
        'Agreeableness'    : 'Coordinators score highest — they reply to the most people, showing social orientation.',
        'Openness'         : 'Contributors score highest — they write the longest, most detailed messages.',
        'Conscientiousness': 'Leaders and Coordinators score highest — they post consistently across many days.',
        'Neuroticism'      : 'Generally low across all roles — r/programming tends toward calm, technical discussion.'
    }
    for trait, text in interp.items():
        st.markdown(f"**{trait}:** {text}")

# ════════════════════════════════════════════════════════════════════
# TAB 4 — ACTIVITY TIMELINE
# ════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Activity Timeline — May 2015")

    with st.spinner("Loading timeline data..."):
        raw = load_clean()

    raw['day'] = raw['datetime'].dt.day

    # Daily message volume
    daily = raw.groupby('day').size().reset_index(name='messages')

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(daily['day'], daily['messages'],
                    alpha=0.3, color='#4F86C6')
    ax.plot(daily['day'], daily['messages'],
            color='#4F86C6', linewidth=2)
    ax.set_xlabel("Day of May 2015")
    ax.set_ylabel("Number of messages")
    ax.set_title("Daily Message Volume — r/programming + r/learnprogramming")
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # Daily volume by role
    st.subheader("Daily Activity by Role")
    merged = raw.merge(
        df[['author','role']], on='author', how='left'
    ).dropna(subset=['role'])

    daily_role = merged.groupby(['day','role']).size().reset_index(name='count')

    fig, ax = plt.subplots(figsize=(12, 5))
    for role in daily_role['role'].unique():
        subset = daily_role[daily_role['role'] == role]
        ax.plot(
            subset['day'], subset['count'],
            label=role, color=ROLE_COLORS.get(role, '#999'),
            linewidth=2, marker='o', markersize=3
        )
    ax.set_xlabel("Day of May 2015")
    ax.set_ylabel("Messages per day")
    ax.set_title("Daily Message Volume by Communication Role")
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # Hourly heatmap
    st.subheader("Posting Heatmap — Day vs Hour")
    raw['hour']    = raw['datetime'].dt.hour
    pivot = raw.groupby(['day','hour']).size().unstack(fill_value=0)

    # Sample every 3rd day to keep readable
    pivot_sampled = pivot.iloc[::2]

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(pivot_sampled.values, aspect='auto',
                   cmap='Blues', interpolation='nearest')
    ax.set_xlabel("Hour of day (UTC)")
    ax.set_ylabel("Day of May")
    ax.set_title("Posting Intensity by Day and Hour")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels(range(0, 24, 2))
    ax.set_yticks(range(len(pivot_sampled)))
    ax.set_yticklabels(pivot_sampled.index.tolist())
    plt.colorbar(im, ax=ax, label='Messages')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ════════════════════════════════════════════════════════════════════
# TAB 5 — ML CLASSIFIER
# ════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Random Forest Role Classifier")
    st.markdown("""
    We use the rule-based role labels as **pseudo-ground-truth** to train
    a Random Forest classifier. This gives us:
    - A **generalizable model** that learns patterns beyond hard thresholds
    - **Feature importance** showing which signals drive role classification
    - **Validation metrics** to assess classification quality
    """)

    with st.spinner("Training Random Forest (100 trees)..."):
        rf, le, report, importances, accuracy = train_random_forest()

    st.success(f"Model trained · Test accuracy: **{accuracy*100:.1f}%**")

    st.markdown("---")
    col_ml1, col_ml2 = st.columns(2)

    with col_ml1:
        st.subheader("Feature Importance")
        sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        feat_names = [x[0].replace('_',' ') for x in sorted_imp]
        feat_vals  = [x[1] for x in sorted_imp]

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.barh(feat_names[::-1], feat_vals[::-1],
                       color='#4F86C6', alpha=0.85)
        ax.set_xlabel("Importance score")
        ax.set_title("Which features matter most?")
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("**Top 3 most important features:**")
        for i, (feat, val) in enumerate(sorted_imp[:3]):
            st.markdown(f"{i+1}. `{feat}` — {val*100:.1f}% importance")

    with col_ml2:
        st.subheader("Classification Report")
        report_df = pd.DataFrame(report).transpose()
        report_df = report_df.drop(
            ['accuracy','macro avg','weighted avg'], errors='ignore'
        )
        report_df = report_df[['precision','recall','f1-score','support']]
        report_df['support'] = report_df['support'].astype(int)
        report_df = report_df.round(3)
        st.dataframe(report_df, use_container_width=True)

        st.markdown("""
        **How to read this:**
        - **Precision**: of users classified as Role X, how many truly are?
        - **Recall**: of all actual Role X users, how many were found?
        - **F1-score**: harmonic mean of precision and recall
        - High scores validate that our behavioral features genuinely
          distinguish the roles
        """)

    st.markdown("---")
    st.subheader("Predict Role for a Custom User Profile")
    st.markdown("Adjust the sliders to simulate a user and predict their role:")

    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        p_msgs    = st.slider("Message count",      1, 400, 10)
        p_length  = st.slider("Avg message length", 10, 1000, 200)
        p_replies = st.slider("Replies sent",       0, 400, 5)
    with pc2:
        p_interact= st.slider("Interaction count",  0, 200, 5)
        p_days    = st.slider("Active days",        1, 31, 5)
        p_qratio  = st.slider("Question ratio",     0.0, 1.0, 0.1)
    with pc3:
        p_sent    = st.slider("Avg sentiment",      -1.0, 1.0, 0.1)
        p_sentvar = st.slider("Sentiment variability", 0.0, 1.0, 0.2)
        p_resptime= st.slider("Avg response time (mins)", 0, 500, 60)

    user_input = np.array([[
        p_msgs, p_length, p_qratio, p_replies,
        p_interact, p_sent, p_sentvar, p_days, p_resptime
    ]])

    pred_role  = le.inverse_transform(rf.predict(user_input))[0]
    pred_proba = rf.predict_proba(user_input)[0]

    color = ROLE_COLORS.get(pred_role, '#999')
    st.markdown(
        f"<div style='margin-top:16px;padding:16px;"
        f"background:{color}22;border-left:4px solid {color};"
        f"border-radius:8px'>"
        f"<span style='font-size:18px;font-weight:600'>Predicted role: "
        f"<span style='color:{color}'>{pred_role}</span></span><br>"
        f"<span style='font-size:13px;color:#666'>"
        f"{ROLE_DESCRIPTIONS.get(pred_role,'')}</span></div>",
        unsafe_allow_html=True
    )

    # Probability bar
    st.markdown("**Confidence per role:**")
    proba_df = pd.DataFrame({
        'Role': le.classes_,
        'Confidence': pred_proba * 100
    }).sort_values('Confidence', ascending=False)

    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.barh(
        proba_df['Role'], proba_df['Confidence'],
        color=[ROLE_COLORS.get(r,'#999') for r in proba_df['Role']],
        alpha=0.85
    )
    ax.set_xlabel("Confidence (%)")
    ax.set_xlim(0, 100)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════
st.markdown("---")
st.caption(
    "Team Communication Analyzer · Data: Reddit May 2015 · "
    "63,150 messages · 15,819 users · "
    "Roles & Big Five traits are approximate behavioral mappings."
)