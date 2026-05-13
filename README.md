# 🐝 COIN Team Communication Analyzer
### Virtual Mirroring Dashboard for Software Teams

> *"We combine behavioral communication signals with lightweight semantic features to infer COIN archetypes and approximate psychological roles from real team chat data — without requiring any surveys, annotations, or prior knowledge of the users."*

---

## 📋 Table of Contents
1. [What This Project Does](#what-this-project-does)
2. [Background — COINs Theory](#background--coins-theory)
3. [The 5 Archetypes](#the-5-archetypes)
4. [Data Sources](#data-sources)
5. [Project Architecture](#project-architecture)
6. [How to Run It](#how-to-run-it)
7. [File Structure](#file-structure)
8. [How Each Script Works](#how-each-script-works)
9. [The Hybrid ML Model](#the-hybrid-ml-model)
10. [Dashboard Features](#dashboard-features)
11. [How to Add Your WhatsApp Chat](#how-to-add-your-whatsapp-chat)
12. [Results Interpretation](#results-interpretation)
13. [Limitations & Honest Disclaimers](#limitations--honest-disclaimers)
14. [References](#references)

---

## What This Project Does

This system analyzes **real team chat messages** (WhatsApp, Slack) and automatically:

1. **Extracts behavioral features** from each team member — how often they write, how long their messages are, how quickly they reply, what kind of language they use
2. **Classifies each person** into one of 5 COIN archetypes (Bee, Ant, Butterfly, Capybara, Leech)
3. **Shows a virtual mirror** — a dashboard that lets the team see their own communication patterns

The goal is not surveillance. It is **self-awareness**. Teams that see their own patterns can change them.

---

## Background — COINs Theory

A **COIN (Collaborative Innovation Network)** is defined by Prof. Peter Gloor (MIT) as:

> *"A cyberteam of self-motivated people with a collective vision, enabled by technology, to collaborate in achieving a common goal."*

Every major technology revolution was built by a COIN:
- **Unix** — a small group at Bell Labs sharing code freely
- **The Web** — Tim Berners-Lee and collaborators, open protocols
- **Linux** — thousands of volunteers, no central control
- **The Transformer** — 8 researchers, one lunch conversation, the architecture behind ChatGPT

COINs succeed through five principles: learning networks, ethical code, trust & self-organization, knowledge accessibility, and honest feedback.

**Virtual Mirroring** is the practice of using AI to analyze a team's actual communication and show them patterns they cannot see themselves. This project is a Virtual Mirror.

---

## The 5 Archetypes

These are **behavioral patterns**, not personality types. People shift between them. Healthy COINs need all four productive archetypes.

| Archetype | Role | Communication Signals |
|---|---|---|
| 🐝 **Bee** | Creates ideas, connects people, cross-pollinates between groups | High message volume, many @mentions, introduces new topics, asks many questions |
| 🐜 **Ant** | Builds, executes, ensures quality, gets things done | Task-focused language ("done", "fixed", "submitted"), fast replies, consistent daily activity |
| 🦋 **Butterfly** | Transforms complexity into clarity, summarizes, reformulates | Long detailed messages, summarizing language ("in summary", "to clarify"), reformulates others' ideas |
| 🦫 **Capybara** | Harmonizes, creates psychological safety, bridges differences | Positive sentiment, supportive words ("great job", "thank you"), acknowledges others' contributions |
| 🔴 **Leech** | Takes without contributing *(symptom, not personality)* | Few messages, rarely replies, only appears at deadlines, no @mentions of others |

> ⚠️ **Important**: Leech behavior is usually a signal that someone is **mismatched with their role, disengaged, or struggling**. The fix is not punishment — it is helping them find where they can contribute.

---

## Data Sources

### Training Data — Slack Developer Chats (Chatterjee et al., MSR 2020)
- **What**: Real software developer conversations from public Slack communities
- **Communities**: Python, Clojure, Elm, Racket programming channels
- **Size**: 437,893 messages from 12,171 users (we use 60,000 messages / 1,639 users)
- **Why**: Peer-reviewed academic dataset, published at the International Conference on Mining Software Repositories
- **Citation**: Chatterjee, P., Damevski, K., Kraft, N.A., Pollock, L. (2020). *Software-related Slack Chats with Disentangled Conversations*. MSR 2020.
- **Link**: https://github.com/preethac/Software-related-Slack-Chats-with-Disentangled-Conversations

### Test / Demo Data — Team WhatsApp Group
- **What**: Our own project team's WhatsApp group chat (exported with consent of all members)
- **Size**: 324 messages from 5 team members
- **Period**: April–May 2026
- **Why**: This is the "virtual mirror" — we analyze our own team live

---

## Project Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING PHASE                               │
│                                                                 │
│   Slack Developer Chat (60k messages, 1,639 software engineers) │
│              ↓                                                  │
│   Feature Extraction (11 behavioral features per user)         │
│              ↓                                                  │
│   COIN Rule-Based Labels  ←── Gloor's archetype theory         │
│   (pseudo ground truth)                                         │
│              ↓                                                  │
│   Random Forest Training  (5-fold cross-validation)            │
│   F1 = 0.891 ± 0.010                                           │
└─────────────────────────────────────────────────────────────────┘
                           ↓
              Trained Model (saved in memory)
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                    PREDICTION PHASE                             │
│                                                                 │
│   WhatsApp Team Chat (324 messages, 5 members)                 │
│              ↓                                                  │
│   Same Feature Extraction                                       │
│              ↓                                                  │
│   ML Prediction  +  Rule-Based Prediction (both shown)         │
│              ↓                                                  │
│   Archetype Scores (0-100 per dimension)                       │
│              ↓                                                  │
│   Virtual Mirror Dashboard (Streamlit)                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## How to Run It

### Prerequisites
```bash
pip install pandas numpy scikit-learn nltk textblob networkx streamlit matplotlib seaborn
```

### First time setup
```bash
# 1. Clone the Slack training dataset
git clone https://github.com/preethac/Software-related-Slack-Chats-with-Disentangled-Conversations.git slack_data

# 2. Parse the Slack XML data
python parse_slack.py

# 3. Export your WhatsApp group:
#    Phone → Group → Export Chat → Without Media → send to yourself
#    Rename the file to whatsapp_chat.txt and place it in this folder

# 4. Parse your WhatsApp chat
python parse_whatsapp.py

# 5. Extract features from WhatsApp
python extract_features.py

# 6. Train on Slack + predict for WhatsApp
python train_and_predict.py

# 7. Launch the dashboard
streamlit run dashboard.py
```

### After first setup (re-run anytime)
```bash
python train_and_predict.py
streamlit run dashboard.py
```

---

## File Structure

```
SNA/
├── 📄 parse_whatsapp.py        Parses WhatsApp .txt export → whatsapp_clean.json
├── 📄 parse_slack.py           Parses Slack XML files → slack_clean.json
├── 📄 extract_features.py      Extracts 11 behavioral features from WhatsApp data
├── 📄 classify_archetypes.py   Rule-based archetype classifier (standalone)
├── 📄 train_and_predict.py     ⭐ Main: trains on Slack, predicts for WhatsApp
├── 📄 dashboard.py             Streamlit virtual mirror dashboard
│
├── 📁 slack_data/              Cloned Slack dataset (XML files)
├── 📊 slack_clean.json         Parsed Slack messages (60k rows)
├── 📊 slack_results.json       Slack users with archetype labels (training reference)
├── 📊 whatsapp_chat.txt        Raw WhatsApp export (not committed to git)
├── 📊 whatsapp_clean.json      Parsed WhatsApp messages
├── 📊 wa_features.json         Feature matrix for WhatsApp team
├── 📊 wa_results.json          Final archetype predictions for team (dashboard input)
│
├── 📄 .gitignore               Excludes large files and private chat data
└── 📄 README.md                This file
```

---

## How Each Script Works

### `parse_whatsapp.py`
Reads the raw WhatsApp `.txt` export and converts it into a structured JSON file.
- Handles two date formats (European DD/MM/YYYY and US MM/DD/YYYY)
- Maps phone numbers to real names via `NAME_MAP`
- Detects replies: if a message comes within 10 minutes of another person's message, it is classified as a reply
- Filters out system messages (encryption notices, media omitted, etc.)

### `parse_slack.py`
Reads the XML files from the Slack academic dataset.
- Parses `<ts>`, `<user>`, `<text>` fields from each message
- Extracts channel name from folder structure
- Uses `conversation_id` attribute to detect reply chains
- Samples 60,000 messages for manageable processing

### `extract_features.py`
Computes 11 behavioral features per user from WhatsApp data:

| Feature | What it measures | Archetype signal |
|---|---|---|
| `msg_count` | Total messages sent | Bee (volume) |
| `avg_msg_length` | Average characters per message | Butterfly (depth) |
| `question_ratio` | % of messages containing "?" | Bee (curiosity) |
| `avg_mentions` | Average @mentions per message | Bee (connectivity) |
| `replies_sent` | How many times user replied | Ant (responsiveness) |
| `avg_reply_speed_mins` | Median minutes to reply | Ant (speed) |
| `task_focus_score` | Density of task words ("done", "fixed") | Ant (execution) |
| `butterfly_score` | Density of summarizing phrases | Butterfly (clarification) |
| `avg_sentiment` | Average emotional polarity (-1 to +1) | Capybara (positivity) |
| `capybara_score` | Density of supportive words | Capybara (harmony) |
| `active_days` | Number of distinct days active | Ant/Bee (consistency) |
| `new_topic_ratio` | % of messages starting a new thread | Bee (initiative) |

### `train_and_predict.py`
The core of the hybrid model:

1. **Extracts features** from 1,639 Slack users
2. **Applies COIN rules** to generate archetype labels (pseudo ground truth)
3. **Trains Random Forest** (200 trees, max depth 10, class-balanced)
4. **5-fold cross-validation** → F1 = 0.891 ± 0.010
5. **Predicts** archetypes for your 5 WhatsApp teammates
6. **Shows both** ML prediction and rule-based prediction side by side
7. **Reports confidence** score per prediction

---

## The Hybrid ML Model

### Why hybrid?

A purely rule-based system is transparent and theoretically grounded but rigid — it uses hard thresholds. A purely ML system learns complex patterns but is a black box. We combine both:

- **Rules** → grounded in COIN theory, interpretable, always visible
- **ML** → generalizes beyond hard thresholds, validated at scale, learns from 1,639 real software engineers

### Why Random Forest?

- Handles small prediction sets (5 people) without overfitting
- Provides feature importance — explains *why* someone is classified a certain way
- Works well with class imbalance (we use `class_weight='balanced'`)
- Interpretable compared to neural networks

### Training results
```
Cross-validation F1 (5-fold):  0.891 ± 0.010
Training accuracy:              97.8%

Top features by importance:
  msg_count          0.183  ████████
  replies_sent       0.159  ███████
  avg_mentions       0.121  █████
  avg_msg_length     0.118  █████
  avg_sentiment      0.107  █████
  active_days        0.091  ████
```

### Why rule ↔ ML disagreement happens

With only 5 people, the ML model (trained on 1,639 Slack users) sometimes disagrees with the rules (designed for small groups). This is expected and scientifically honest. We show both predictions and let the team discuss which feels more accurate — this is part of the virtual mirroring process.

---

## Dashboard Features

The Streamlit dashboard has 4 tabs:

### Tab 1 — 🦁 Archetypes
- Pie chart of team archetype distribution
- Colored cards per archetype showing which members belong and why
- Bar charts comparing all members on each archetype dimension (0–100)
- **COIN Health Check** — alerts if archetypes are missing or Leech behavior detected

### Tab 2 — 🌐 Interaction Network
- Force-directed graph showing who replies to whom
- Node size = message count, color = archetype
- Edge thickness = interaction frequency
- Statistics: most replied-to, most replies sent, most connected

### Tab 3 — 📅 Timeline & Heatmap
- Daily message volume over time
- Activity by archetype per day
- Hour-of-day heatmap per member (when is each person most active?)

### Tab 4 — 👤 Member Profiles
- Select any team member
- See their archetype card, all 5 dimension scores, radar chart
- Behavioral insights generated automatically
- Sample of their recent messages

---

## How to Add Your WhatsApp Chat

1. Open the WhatsApp group on your phone
2. Tap the group name → Export Chat → Without Media
3. Send the `.txt` file to yourself (email or AirDrop)
4. Rename it `whatsapp_chat.txt` and put it in the project folder
5. Edit `NAME_MAP` in `parse_whatsapp.py` to replace phone numbers with names
6. Run: `python parse_whatsapp.py`
7. Run: `python train_and_predict.py`
8. Run: `streamlit run dashboard.py`

### Name mapping example
```python
NAME_MAP = {
    '+49 1523 3682176': 'Celina',
    '+49 163 1519856' : 'Faryan',
    '+49 1517 0872047': 'Marie',
    'esmaeil molapour': 'Esmaeil',
    'kasra'           : 'Kasra'
}
```

---

## Results Interpretation

### Our team's results

| Member | ML Archetype | Rule Archetype | Confidence | Messages |
|---|---|---|---|---|
| Celina | 🐜 Ant | 🐝 Bee | 86% | 197 |
| Faryan | 🐜 Ant | 🐝 Bee | 79% | 69 |
| Esmaeil | 🐜 Ant | 🐜 Ant | 49% | 29 |
| Marie | 🐜 Ant | 🦋 Butterfly | 76% | 15 |
| Kasra | 🐜 Ant | 🔴 Leech | 52% | 14 |

### What this means
- **Celina and Faryan** are the most active members. The rule system classifies them as Bees (idea generators) because of volume and consistency. The ML classifies them as Ants because compared to 1,639 Slack software engineers, their absolute message counts are lower — the model sees task-focused execution patterns dominating.
- **Esmaeil** is classified as Ant by both systems. His Capybara score is high (79) — he is a hybrid Ant-Capybara who gets things done AND supports the team.
- **Marie** writes longer messages (rule system → Butterfly) but the ML sees her overall profile as Ant-like.
- **Kasra** has the lowest activity. The rule system flags Leech behavior (disengagement signal). The ML gives 52% confidence — the weakest prediction, meaning Kasra is on the boundary.

### Key insight: No Capybara
The team has no dedicated harmonizer. Esmaeil has the highest Capybara score but is outweighed by his Ant signals. This means the team may be strong at executing but weak at managing interpersonal dynamics and team morale.

---

## Limitations & Honest Disclaimers

1. **Small sample**: 324 messages over 8 days is a small window. Archetypes may shift with more data.
2. **Approximate mapping**: Our features are behavioral proxies, not clinically validated psychological measurements.
3. **No Capybara in training**: With only 9.2% Capybara in the Slack training data, the model is less confident at detecting this archetype.
4. **Cross-platform gap**: Slack (technical Q&A) and WhatsApp (informal team chat) have different communication norms. The model accounts for this but some drift is expected.
5. **Reply detection approximation**: WhatsApp does not expose reply threading via API. We approximate replies using a 10-minute time window.
6. **Sentiment analysis**: TextBlob is a lightweight model trained on general English text, not developer jargon. Sentiment scores are indicative, not precise.

---

## References

1. **Gloor, P.A.** (2006). *Swarm Creativity: Competitive Advantage through Collaborative Innovation Networks*. Oxford University Press.
2. **Chatterjee, P., Damevski, K., Kraft, N.A., Pollock, L.** (2020). Software-related Slack Chats with Disentangled Conversations. *MSR 2020 Data Showcase Track*. https://github.com/preethac/Software-related-Slack-Chats-with-Disentangled-Conversations
3. **Muthu Subash, K. et al.** (2022). DISCO: A Dataset of Discord Chat Conversations for Software Engineering Research. *MSR 2022*.
4. **Breiman, L.** (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
5. **Project website**: https://sites.google.com/view/coinseminar26/creating-coins

---

## Team

| Name | Role in project |
|---|---|
| Celina | 🐝 Bee / 🐜 Ant |
| Faryan | 🐝 Bee / 🐜 Ant |
| Esmaeil | 🐜 Ant / 🦫 Capybara |
| Marie | 🦋 Butterfly / 🐜 Ant |
| Kasra | Re-engaging 🐜 Ant |

---

*Built with Python · Streamlit · scikit-learn · NetworkX · TextBlob*
*Data: Slack Developer Chats (Chatterjee et al., 2020) + Team WhatsApp*
*Based on COINs research by Prof. Peter Gloor (MIT)*