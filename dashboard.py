import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="COIN Team Analyzer", page_icon="🐝", layout="wide"
)

# ── Archetype metadata ────────────────────────────────────────────────────
ARCHETYPE_COLORS = {
    '🐝 Bee'      : '#F4C430',
    '🐜 Ant'      : '#8B4513',
    '🦋 Butterfly': '#9B59B6',
    '🦫 Capybara' : '#2ECC71',
    '🔴 Leech'    : '#E74C3C'
}

ARCHETYPE_DESC = {
    '🐝 Bee'      : 'Creates ideas, connects people, cross-pollinates between groups.',
    '🐜 Ant'      : 'Builds, executes, ensures quality. Gets things done.',
    '🦋 Butterfly': 'Transforms complexity into clarity. Summarizes and reformulates.',
    '🦫 Capybara' : 'Harmonizes the team, creates psychological safety, mediates.',
    '🔴 Leech'    : 'Low contribution. Usually disengaged or mismatched — needs support.'
}

ARCHETYPE_SIGNALS = {
    '🐝 Bee'      : 'High message volume · Many @mentions · Introduces new topics · Asks questions',
    '🐜 Ant'      : 'Task-focused language · Fast replies · Consistent daily activity',
    '🦋 Butterfly': 'Long messages · Summarizing language · Reformulates others\' ideas',
    '🦫 Capybara' : 'Positive sentiment · Supportive words · Acknowledges contributions',
    '🔴 Leech'    : 'Few messages · Rarely replies · Low active days · No @mentions'
}

SCORE_COLS   = ['bee_score', 'ant_score', 'butterfly_score_n',
                'capybara_score_n', 'leech_risk']
SCORE_LABELS = ['🐝 Bee', '🐜 Ant', '🦋 Butterfly', '🦫 Capybara', '🔴 Leech Risk']
SCORE_EMOJIS = ['🐝', '🐜', '🦋', '🦫', '🔴']

# ── Load data ─────────────────────────────────────────────────────────────
@st.cache_data
def load_results():
    return pd.read_json("wa_results.json", lines=True)

@st.cache_data
def load_clean():
    df = pd.read_json("whatsapp_clean.json", lines=True)
    df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
    return df

df = load_results()

# ── Header ────────────────────────────────────────────────────────────────
st.title("🐝 COIN Team Communication Analyzer")
st.markdown(
    "**Virtual Mirroring Dashboard** · Analyzes your team's WhatsApp chat "
    "to reveal COIN archetypes and communication patterns · "
    "Based on COINs research by Prof. Peter Gloor"
)
st.markdown("---")

# ── Top metrics ───────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Team Members",     len(df))
c2.metric("Total Messages",   int(df['msg_count'].sum()))
c3.metric("Archetypes Found", df['archetype'].nunique())
c4.metric("Avg Active Days",  f"{df['active_days'].mean():.1f}")
c5.metric("Avg Sentiment",    f"{df['avg_sentiment'].mean():.2f}")
st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🦁 Archetypes",
    "🌐 Interaction Network",
    "📅 Timeline & Heatmap",
    "👤 Member Profiles"
])

# ════════════════════════════════════════════════════════════════════
# TAB 1 — ARCHETYPES
# ════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Team Archetype Distribution")
    col1, col2 = st.columns([1, 1.3])

    with col1:
        dist   = df['archetype'].value_counts()
        colors = [ARCHETYPE_COLORS.get(r, '#999') for r in dist.index]
        fig, ax = plt.subplots(figsize=(5, 5))
        wedges, texts, autotexts = ax.pie(
            dist.values,
            labels=[r.split(' ', 1)[1] for r in dist.index],
            autopct='%1.0f%%', colors=colors,
            startangle=140, pctdistance=0.78
        )
        for t in autotexts:
            t.set_fontsize(11); t.set_fontweight('bold')
        ax.set_title("Team Archetypes", fontsize=13, pad=12)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        for archetype in dist.index:
            count   = dist[archetype]
            color   = ARCHETYPE_COLORS.get(archetype, '#999')
            desc    = ARCHETYPE_DESC.get(archetype, '')
            signals = ARCHETYPE_SIGNALS.get(archetype, '')
            members = df[df['archetype'] == archetype]['author'].tolist()
            st.markdown(
                f"<div style='margin-bottom:14px;padding:12px;"
                f"border-left:5px solid {color};border-radius:6px;"
                f"background:{color}18'>"
                f"<span style='font-size:20px'>{archetype}</span> "
                f"<span style='font-size:13px;color:#666'>"
                f"({count} member{'s' if count > 1 else ''})</span><br>"
                f"<span style='font-size:13px'>{desc}</span><br>"
                f"<span style='font-size:11px;color:#888'>📡 {signals}</span><br>"
                f"<span style='font-size:13px;font-weight:700;color:#333'>"
                f"👤 {' · '.join(members)}</span>"
                f"</div>",
                unsafe_allow_html=True
            )

    st.markdown("---")
    st.subheader("Archetype Score Comparison")
    st.markdown("How strongly each member scores on each archetype dimension (0–100).")

    fig, axes = plt.subplots(1, len(SCORE_COLS), figsize=(14, 4))
    df_sorted = df.sort_values('msg_count', ascending=False)

    for idx, (col, label) in enumerate(zip(SCORE_COLS, SCORE_LABELS)):
        ax     = axes[idx]
        vals   = df_sorted[col].values
        names  = df_sorted['author'].values
        colors = [ARCHETYPE_COLORS.get(a, '#999') for a in df_sorted['archetype']]
        bars   = ax.barh(names, vals, color=colors, alpha=0.85)
        ax.set_xlim(0, 100)
        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.set_xlabel("Score")
        ax.grid(axis='x', alpha=0.3)
        for bar, val in zip(bars, vals):
            ax.text(min(val + 2, 92), bar.get_y() + bar.get_height() / 2,
                    f'{val:.0f}', va='center', fontsize=9)

    plt.suptitle("Per-Member Archetype Scores", fontsize=13, y=1.02)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("🏥 COIN Health Check")

    archetypes_present = set(df['archetype'].unique())
    needed  = {'🐝 Bee', '🐜 Ant', '🦋 Butterfly', '🦫 Capybara'}
    missing = needed - archetypes_present

    if '🔴 Leech' in archetypes_present:
        leech_members = df[df['archetype'] == '🔴 Leech']['author'].tolist()
        st.warning(
            f"⚠️ **Leech behavior detected** in: **{', '.join(leech_members)}**. "
            "This signals disengagement or role mismatch — not a personality flaw. "
            "Consider a conversation about workload or motivation."
        )
    if missing:
        st.warning(
            f"⚠️ **Missing archetypes:** {', '.join(missing)}. "
            "A healthy COIN needs all four productive types."
        )
    if not missing and '🔴 Leech' not in archetypes_present:
        st.success("✅ **Healthy COIN!** All four productive archetypes are present.")

    h1, h2 = st.columns(2)
    with h1:
        st.markdown("""
        **What a healthy COIN looks like:**
        - 🐝 **Bee** — at least one idea generator and connector
        - 🐜 **Ant** — multiple executors delivering work
        - 🦋 **Butterfly** — someone making communication clear
        - 🦫 **Capybara** — a harmonizer keeping the team cohesive
        - 🔴 **No Leech** behavior (or it's being actively addressed)
        """)
    with h2:
        st.markdown("""
        **Your team's specific insight:**
        - Two Bees = strong idea generation, may need more execution focus
        - No Capybara = no dedicated harmonizer, watch for unresolved tension
        - Esmaeil has high Capybara score (79) despite Ant classification — hybrid role
        - Kasra's Leech signal = re-engagement needed, not a judgment
        """)

# ════════════════════════════════════════════════════════════════════
# TAB 2 — INTERACTION NETWORK
# ════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Team Interaction Network")
    st.markdown(
        "**Nodes** = team members · **Size** = message count · "
        "**Color** = archetype · **Edges** = reply interactions · "
        "**Edge thickness** = interaction frequency"
    )

    with st.spinner("Building network..."):
        raw     = load_clean()
        raw     = raw.sort_values('datetime').reset_index(drop=True)
        members = set(df['author'].unique())

        G = nx.DiGraph()
        for _, row in df.iterrows():
            G.add_node(row['author'], archetype=row['archetype'])

        for i in range(1, len(raw)):
            a        = raw.iloc[i]['author']
            prev     = raw.iloc[i - 1]['author']
            gap_mins = (
                raw.iloc[i]['datetime'] - raw.iloc[i - 1]['datetime']
            ).total_seconds() / 60
            if a != prev and a in members and prev in members and gap_mins <= 10:
                if G.has_edge(a, prev):
                    G[a][prev]['weight'] += 1
                else:
                    G.add_edge(a, prev, weight=1)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_facecolor('#0f1117')
    fig.patch.set_facecolor('#0f1117')

    pos = nx.spring_layout(G, k=4, seed=42)

    node_colors = []
    node_sizes  = []
    for node in G.nodes():
        row       = df[df['author'] == node]
        archetype = row.iloc[0]['archetype'] if len(row) > 0 else '🔴 Leech'
        msg_count = float(row.iloc[0]['msg_count']) if len(row) > 0 else 1
        node_colors.append(ARCHETYPE_COLORS.get(archetype, '#999'))
        node_sizes.append(max(800, min(msg_count * 25, 5000)))

    edges   = list(G.edges())
    weights = [G[u][v]['weight'] for u, v in edges]
    max_w   = max(weights) if weights else 1

    if edges:
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            width=[w / max_w * 5 for w in weights],
            edge_color='#ffffff55', arrows=True,
            arrowsize=20, connectionstyle='arc3,rad=0.2'
        )

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors, node_size=node_sizes, alpha=0.95
    )
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_color='white', font_size=10, font_weight='bold'
    )

    legend_handles = [
        mpatches.Patch(color=c, label=a.split(' ', 1)[1])
        for a, c in ARCHETYPE_COLORS.items()
        if a in df['archetype'].unique()
    ]
    ax.legend(
        handles=legend_handles, loc='lower left', fontsize=10,
        facecolor='#1a1a2e', labelcolor='white', framealpha=0.9
    )
    ax.axis('off')
    ax.set_title("Team Interaction Network", color='white', fontsize=14, pad=15)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown("---")
    nc1, nc2, nc3, nc4 = st.columns(4)
    nc1.metric("Total interactions", G.number_of_edges())
    if G.number_of_nodes() > 0:
        in_deg  = dict(G.in_degree())
        out_deg = dict(G.out_degree())
        all_deg = dict(G.degree())
        nc2.metric("Most replied-to",
                   max(in_deg,  key=in_deg.get)  if in_deg  else "—")
        nc3.metric("Most replies sent",
                   max(out_deg, key=out_deg.get) if out_deg else "—")
        nc4.metric("Most connected",
                   max(all_deg, key=all_deg.get) if all_deg else "—")

# ════════════════════════════════════════════════════════════════════
# TAB 3 — TIMELINE & HEATMAP
# ════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Activity Timeline")

    with st.spinner("Loading timeline..."):
        raw = load_clean()

    raw['date'] = raw['datetime'].dt.date
    raw['hour'] = raw['datetime'].dt.hour
    daily       = raw.groupby('date').size().reset_index(name='messages')

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(range(len(daily)), daily['messages'],
                    alpha=0.3, color='#F4C430')
    ax.plot(range(len(daily)), daily['messages'],
            color='#F4C430', linewidth=2, marker='o', markersize=5)
    ax.set_xticks(range(len(daily)))
    ax.set_xticklabels(
        [str(d) for d in daily['date']], rotation=45, ha='right', fontsize=9
    )
    ax.set_ylabel("Messages")
    ax.set_title("Daily Message Volume — Team WhatsApp Group")
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("Activity by Archetype Over Time")
    merged     = raw.merge(df[['author', 'archetype']], on='author', how='left')
    merged     = merged.dropna(subset=['archetype'])
    daily_arch = merged.groupby(['date', 'archetype']).size().reset_index(name='count')
    date_list  = list(daily['date'])

    fig, ax = plt.subplots(figsize=(12, 5))
    for arch in daily_arch['archetype'].unique():
        sub = daily_arch[daily_arch['archetype'] == arch]
        xs  = [date_list.index(d) for d in sub['date'] if d in date_list]
        ys  = sub['count'].tolist()[:len(xs)]
        if xs:
            ax.plot(xs, ys, label=arch,
                    color=ARCHETYPE_COLORS.get(arch, '#999'),
                    linewidth=2, marker='o', markersize=5)
    ax.set_xticks(range(len(date_list)))
    ax.set_xticklabels(
        [str(d) for d in date_list], rotation=45, ha='right', fontsize=9
    )
    ax.set_ylabel("Messages per day")
    ax.set_title("Daily Activity by Archetype")
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("When Is Each Member Most Active?")
    hourly = raw.groupby(['author', 'hour']).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(14, max(3, len(hourly) * 0.8)))
    im = ax.imshow(hourly.values, aspect='auto', cmap='YlOrBr')
    ax.set_xticks(range(24))
    ax.set_xticklabels(
        [f"{h}:00" for h in range(24)], rotation=45, ha='right', fontsize=8
    )
    ax.set_yticks(range(len(hourly)))
    ax.set_yticklabels(hourly.index.tolist(), fontsize=11)
    ax.set_xlabel("Hour of day")
    ax.set_title("Message Activity Heatmap — Hour of Day per Member")
    plt.colorbar(im, ax=ax, label='Messages')
    plt.tight_layout()
    st.pyplot(fig); plt.close()

# ════════════════════════════════════════════════════════════════════
# TAB 4 — MEMBER PROFILES
# ════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Individual Member Profiles")

    selected = st.selectbox(
        "Select team member:",
        df.sort_values('msg_count', ascending=False)['author'].tolist()
    )

    if selected:
        user      = df[df['author'] == selected].iloc[0]
        archetype = user['archetype']
        color     = ARCHETYPE_COLORS.get(archetype, '#999')

        st.markdown(
            f"<div style='padding:16px;background:{color}22;"
            f"border-left:6px solid {color};border-radius:8px;"
            f"margin-bottom:20px'>"
            f"<h2 style='margin:0'>{user['author']} &nbsp; {archetype}</h2>"
            f"<p style='margin:6px 0 0 0;font-size:15px;color:#555'>"
            f"{ARCHETYPE_DESC.get(archetype, '')}</p>"
            f"<p style='margin:4px 0 0 0;font-size:12px;color:#888'>"
            f"📡 {ARCHETYPE_SIGNALS.get(archetype, '')}</p>"
            f"</div>",
            unsafe_allow_html=True
        )

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Messages",       int(user['msg_count']))
        m2.metric("Replies sent",   int(user['replies_sent']))
        m3.metric("Active days",    int(user['active_days']))
        m4.metric("Avg msg length", f"{user['avg_msg_length']:.0f} chars")
        m5.metric("Avg sentiment",  f"{user['avg_sentiment']:.2f}")

        st.markdown("---")
        st.markdown("#### Archetype Scores")
        sc = st.columns(5)
        for i, (col, label) in enumerate(zip(SCORE_COLS, SCORE_LABELS)):
            sc[i].metric(label, f"{user[col]:.0f}/100")

        col_r, col_i = st.columns([1, 1])
        with col_r:
            fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
            vals   = [user[c] for c in SCORE_COLS]
            angles = np.linspace(0, 2 * np.pi, len(vals), endpoint=False).tolist()
            vals  += vals[:1]; angles += angles[:1]
            ax.plot(angles, vals, color=color, linewidth=2.5)
            ax.fill(angles, vals, color=color, alpha=0.2)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(SCORE_EMOJIS, size=18)
            ax.set_ylim(0, 100)
            ax.set_yticks([25, 50, 75, 100])
            ax.set_yticklabels(['25', '50', '75', '100'], size=8)
            ax.set_title(
                f"{user['author']} — Archetype Radar",
                pad=20, size=12, fontweight='bold'
            )
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        with col_i:
            st.markdown("#### Behavioral Insights")
            insights = []
            if user['msg_count'] == df['msg_count'].max():
                insights.append("🏆 **Most active** member in the group")
            if user['avg_msg_length'] == df['avg_msg_length'].max():
                insights.append("📝 Writes the **longest messages** — detail-oriented")
            if user['avg_sentiment'] == df['avg_sentiment'].max():
                insights.append("😊 **Most positive** communicator")
            if user['capybara_score_n'] >= 60:
                insights.append("🦫 Strong **harmonizing** tendency")
            if user['bee_score'] >= 60:
                insights.append("🐝 Strong **idea-generating** tendency")
            if user['leech_risk'] >= 80:
                insights.append("⚠️ High **disengagement risk** — needs attention")
            if user['question_ratio'] >= df['question_ratio'].quantile(0.75):
                insights.append("❓ Asks many questions — curious and exploratory")
            if user['task_focus_score'] >= df['task_focus_score'].quantile(0.75):
                insights.append("✅ High **task-focus** — action-oriented language")
            if not insights:
                insights.append("📊 Balanced contributor — no dominant signal")
            for insight in insights:
                st.markdown(f"- {insight}")

        st.markdown("---")
        st.markdown("#### Sample Messages")
        raw_data  = load_clean()
        user_msgs = raw_data[raw_data['author'] == selected][['datetime', 'body']].head(8)
        user_msgs = user_msgs.copy()
        user_msgs['datetime'] = user_msgs['datetime'].dt.strftime('%Y-%m-%d %H:%M')
        st.dataframe(user_msgs.reset_index(drop=True), use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "🐝 COIN Team Analyzer · Virtual Mirroring Dashboard · "
    "Archetypes: Bee · Ant · Butterfly · Capybara · Leech · "
    "Based on COINs research (Gloor, 2006) · "
    f"Analyzing {int(df['msg_count'].sum())} messages from {len(df)} team members"
)