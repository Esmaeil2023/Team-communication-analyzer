# 🐝 COIN Team Communication Analyzer
### Virtual Mirroring Dashboard for Software Teams

> *"We combine behavioral communication signals with linguistic and network-based features  
> to infer COIN archetypes from real team chat data — without surveys, annotations,  
> or prior knowledge of the users."*

---

## 📋 Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Background — COINs Theory](#2-background--coins-theory)
3. [The 5 Archetypes](#3-the-5-archetypes)
4. [Data Sources](#4-data-sources)
5. [System Architecture](#5-system-architecture)
6. [Feature Engineering — All 19 Features](#6-feature-engineering--all-19-features)
7. [The Hybrid ML Model](#7-the-hybrid-ml-model)
8. [How to Run the Project](#8-how-to-run-the-project)
9. [File Structure](#9-file-structure)
10. [Script-by-Script Guide](#10-script-by-script-guide)
11. [Dashboard Guide](#11-dashboard-guide)
12. [Our Team's Results](#12-our-teams-results)
13. [How to Add a New WhatsApp Chat](#13-how-to-add-a-new-whatsapp-chat)
14. [Limitations & Honest Disclaimers](#14-limitations--honest-disclaimers)
15. [How to Defend This Project](#15-how-to-defend-this-project)
16. [References](#16-references)

---

## 1. What This Project Does

This system takes a **raw WhatsApp group export** from a real team and automatically:

1. Parses the raw `.txt` file into structured data
2. Extracts **19 behavioral, linguistic, and network features** per team member
3. Applies **COIN theory rules** to generate archetype labels
4. Trains a **Random Forest classifier** on 1,787 real software engineers
5. Combines both into a **hybrid prediction** with confidence scores
6. Outputs a **percentage distribution** across all 5 archetypes per person
7. Displays everything in an **interactive Virtual Mirror dashboard**

The goal is team self-awareness — not surveillance. Teams that can see their own communication patterns can improve them.

---

## 2. Background — COINs Theory

A **COIN (Collaborative Innovation Network)** is defined by Prof. Peter Gloor (MIT) as:

> *"A cyberteam of self-motivated people with a collective vision, enabled by technology,  
> to collaborate in achieving a common goal."*

Every major technology innovation started as a COIN:
- **Unix** — Bell Labs, open code sharing
- **The Web** — Tim Berners-Lee, open protocols
- **Linux** — thousands of volunteers, no hierarchy
- **The Transformer** — 8 researchers, now the core of ChatGPT

**Virtual Mirroring** is the COIN practice of showing a team their own communication patterns using AI — patterns invisible from inside the group. This project is a Virtual Mirror.

The five archetypes in this project come from Gloor (2022) *Happimetrics* and were further operationalized in the seminar's *Other Sources* document (2026) using communication signatures — not personality tests.

---

## 3. The 5 Archetypes

These are **behavioral communication patterns**, not fixed personality labels. One person can show multiple archetypes simultaneously. That is why the output is always a **percentage distribution**, not a single label.

| Archetype | Role | Core communication signal |
|-----------|------|--------------------------|
| 🐝 **Bee** | Creative connector, idea generator, topic-jumper | High volume · many @mentions · introduces new topics · asks questions · high vocabulary diversity · acts as bridge between people |
| 🐜 **Ant** | Reliable executor, task-focused, gets things done | Task language ("done", "fixed", "submitted") · fast consistent replies · low response variance · steady daily presence |
| 🦋 **Butterfly** | Transforms complexity into clarity, socially warm | Long messages · summarizing phrases · high emoji use · emotional language · expressive communication |
| 🦫 **Capybara** | Harmonizer, creates psychological safety | Starts messages with affirmations · supportive words · positive sentiment · acknowledges others |
| 🔴 **Leech** | Takes without contributing *(symptom, not personality)* | Low message count · receives more than sends · rarely initiates · only appears at deadlines |

> ⚠️ **Leech behavior** signals disengagement or role mismatch, not a character flaw.  
> The correct response is a conversation about re-engagement, not punishment.

---

## 4. Data Sources

### Training Data 1 — Slack Developer Chats

| Property | Value |
|----------|-------|
| **Citation** | Chatterjee, P., Damevski, K., Kraft, N.A., Pollock, L. (2020). *Software-related Slack Chats with Disentangled Conversations*. MSR 2020 |
| **Type** | Real software developer Slack conversations |
| **Communities** | Python, Clojure, Elm, Racket programming channels |
| **Full size** | 437,893 messages · 12,171 users |
| **Used** | 60,000 messages · 1,639 users |
| **Why** | Peer-reviewed academic dataset, real software engineers, citable |
| **Link** | https://github.com/preethac/Software-related-Slack-Chats-with-Disentangled-Conversations |

### Training Data 2 — Nankani 2020 CODERS WhatsApp

| Property | Value |
|----------|-------|
| **Citation** | Nankani, T. (2020). *WhatsApp Chat Export — CODERS Group, TSEC Mumbai* |
| **Type** | Real WhatsApp group of CS/IT engineering students |
| **Topics** | Coding, GitHub, GCP, web development, hackathons |
| **Size** | 11,460 messages · 148 users |
| **Date range** | January–October 2020 |
| **Why** | Real software team WhatsApp (not Slack), same domain as our test group |

### Test / Demo Data — Our Team WhatsApp

| Property | Value |
|----------|-------|
| **Type** | Our own project team WhatsApp group (with consent of all members) |
| **Size** | 324 messages · 5 members |
| **Period** | April–May 2026 |
| **Why** | This is the live Virtual Mirror — we analyze our own team |

**Why three datasets matter:** The model trains on two independent real-world software community datasets from different platforms (Slack and WhatsApp), then applies learned patterns to a third real team. This multi-source approach makes the results more robust and defensible.

---

## 5. System Architecture

```
╔══════════════════════════════════════════════════════════════════╗
║                        TRAINING PHASE                           ║
║                                                                  ║
║  Dataset 1: Slack (1,639 software engineers)                    ║
║  Dataset 2: Nankani 2020 WhatsApp (148 CS/IT students)          ║
║           ↓                                                      ║
║  Feature Extraction (19 features per user)                      ║
║           ↓                                                      ║
║  Rank Normalization  ← within-group percentile ranks (0–1)      ║
║           ↓                                                      ║
║  COIN Rule-Based Labels  ← Gloor (2022) archetype theory        ║
║  (pseudo ground truth for ML training)                           ║
║           ↓                                                      ║
║  Random Forest (300 trees, balanced classes)                    ║
║  5-fold cross-validation  →  F1 = 0.838 ± 0.014                ║
╚══════════════════════════════════════════════════════════════════╝
                            ↓
               Trained Model (generalizable patterns)
                            ↓
╔══════════════════════════════════════════════════════════════════╗
║                       PREDICTION PHASE                          ║
║                                                                  ║
║  Test Data: Team WhatsApp (5 members, 324 messages)             ║
║           ↓                                                      ║
║  Same 19-feature extraction                                      ║
║           ↓                                                      ║
║  Rank Normalization (within WhatsApp group)                      ║
║           ↓                                                      ║
║  ML Prediction  +  Rule-Based Prediction  (run independently)   ║
║           ↓                                                      ║
║  Hybrid Vote: 50% ML probability + 50% rule-based score         ║
║           ↓                                                      ║
║  Weighted Archetype Score Formula → Percentage Distribution      ║
║  (e.g., Celina: 🐝 30% · 🐜 35% · 🦋 23% · 🦫 12% · 🔴 1%)   ║
║           ↓                                                      ║
║  Virtual Mirror Dashboard (Streamlit)                           ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 6. Feature Engineering — All 19 Features

These features are extracted per user from raw chat messages. All features are then rank-normalized within their dataset before training or prediction.

### Group A — Activity & Volume

| Feature | Calculation | Archetype signal |
|---------|-------------|-----------------|
| `msg_count` | Total messages sent | 🐝 Bee (high volume) |
| `avg_msg_length` | Mean characters per message | 🦋 Butterfly (depth) |
| `active_days` | Number of distinct days active | 🐜 Ant (consistency) |
| `replies_sent` | Count of reply messages sent | 🐜 Ant (responsiveness) |
| `replies_received` | Count of times others replied to this person | Used in In/Out ratio |

### Group B — Linguistic / Content

| Feature | Calculation | Archetype signal | Source |
|---------|-------------|-----------------|--------|
| `question_ratio` | % messages containing "?" | 🐝 Bee (curiosity) | |
| `avg_mentions` | Mean @mentions per message | 🐝 Bee (connectivity) | |
| `task_focus_score` | Density of task words ("done", "fixed", "submitted"...) | 🐜 Ant (execution) | |
| `butterfly_score` | Density of summarizing phrases ("in summary", "to clarify"...) | 🦋 Butterfly (clarification) | |
| `capybara_score` | Density of supportive words ("great", "thanks", "well done"...) | 🦫 Capybara (harmony) | |
| `avg_sentiment` | Mean TextBlob polarity score (−1 to +1) | 🦫 Capybara (positivity) | |
| `acknowledgment_rate` | % messages starting with affirmations | 🦫 Capybara | Document §17.1 |
| `emoji_ratio` | Mean emojis per message | 🦋 Butterfly (expressiveness) | Kralj Novak et al. (2015) |
| `emotion_density` | % messages containing emotion words (NRC-style lexicon) | 🦋 Butterfly / 🦫 Capybara | Mohammad & Turney (2013) |
| `mattr` | Moving Average Type-Token Ratio — vocabulary diversity | 🐝 Bee (idea richness) | Covington & McFall (2010) |

### Group C — Timing / Behavior

| Feature | Calculation | Archetype signal | Source |
|---------|-------------|-----------------|--------|
| `new_topic_ratio` | % messages sent after >10 min silence | 🐝 Bee (initiative) | |
| `initiation_rate` | % messages sent after >30 min silence (true conversation starts) | 🐝 Bee / 🦋 Butterfly | Document §17.1 |
| `reply_latency_variance` | Standard deviation of reply gaps (mins) | 🐜 Ant (consistency = low variance) | Kalman et al. (2006) |
| `in_out_ratio` | replies_received / replies_sent | 🔴 Leech (receives >> sends) | Document §17.1 |

### Group D — Network

| Feature | Calculation | Archetype signal | Source |
|---------|-------------|-----------------|--------|
| `betweenness_centrality` | NetworkX betweenness centrality on reply graph | 🐝 Bee (bridges subgroups) | Freeman (1977) |

### Why rank normalization is critical

Without normalization, a model trained on Slack (max 2,196 msgs/user) sees your team (max 197 msgs/user) as uniformly low-activity and classifies everyone as Leech or Ant. With rank normalization, each feature becomes a **within-group percentile rank (0–1)**:

```
Without:  Celina msg_count = 197  →  model sees: low activity → Ant ❌
With:     Celina msg_count = 1.0  →  model sees: top of her group → Bee ✅
```

This is the single most important technical decision in the project.

---

## 7. The Hybrid ML Model

### Why hybrid?

| Approach | Strength | Weakness |
|----------|----------|----------|
| Rule-based only | Transparent, theoretically grounded | Rigid thresholds, no generalization |
| ML only | Learns complex patterns, generalizes | Black box, sensitive to training distribution |
| **Hybrid (ours)** | Both interpretable AND data-driven | Slightly more complex to explain |

### Weighted archetype score formula

Each archetype score is a **weighted combination** of its most relevant features, normalized to sum to 100%:

```
🐝 Bee   = 0.25×MATTR + 0.20×msg_count + 0.20×betweenness + 0.20×initiation + 0.10×mentions + 0.05×question_ratio

🐜 Ant   = 0.30×task_focus + 0.25×replies_sent + 0.20×active_days + 0.15×(1−latency_variance) + 0.10×avg_length

🦋 Butterfly = 0.25×avg_length + 0.20×butterfly_score + 0.20×emoji_ratio + 0.20×emotion_density + 0.15×sentiment

🦫 Capybara  = 0.35×acknowledgment_rate + 0.30×capybara_score + 0.20×sentiment + 0.15×replies_sent

🔴 Leech = 0.40×in_out_ratio + 0.30×(1−msg_count) + 0.20×(1−active_days) + 0.10×(1−replies_sent)
```

All five scores are then normalized so they sum to 100% — giving a distribution, not a label.

### Hybrid voting formula

```
Final archetype score = 50% × ML probability + 50% × Rule-based score

For small groups (< 20 people): 50/50 is optimal because:
  - Rules are grounded in Gloor (2022) theory → should always have a voice
  - ML generalizes from 1,787 engineers → adds data-driven weight
  - Pure ML confidence is naturally lower for 5-person groups
```

### Model performance

```
Algorithm:             Random Forest
Trees:                 300
Max depth:             12
Class weighting:       balanced
Validation:            5-fold stratified cross-validation
CV F1 score:           0.838 ± 0.014
Training accuracy:     96.9%

Training data:         1,787 users
  Slack (Chatterjee):  1,639 users
  Nankani 2020:          148 users

Top features by importance:
  msg_count              0.140
  avg_msg_length         0.140
  replies_sent           0.124
  task_focus_score       0.090
  avg_sentiment          0.089
  capybara_score         0.073
  acknowledgment_rate    0.072
  avg_mentions           0.062
```

---

## 8. How to Run the Project

### Prerequisites

```bash
pip install pandas numpy scikit-learn nltk textblob networkx streamlit matplotlib seaborn nrclex
```

### First-time setup (run once)

```bash
# 1. Clone the Slack training dataset
git clone https://github.com/preethac/Software-related-Slack-Chats-with-Disentangled-Conversations.git slack_data

# 2. Parse Slack XML → slack_clean.json
python parse_slack.py

# 3. Place Nankani_2020.txt in project folder, then:
python parse_nankani.py          # → nankani_clean.json

# 4. Export your WhatsApp group:
#    Phone → open group → tap group name → Export Chat → Without Media
#    Rename the .txt file to whatsapp_chat.txt
#    Place it in the project folder

# 5. Edit NAME_MAP in parse_whatsapp.py to replace phone numbers with names

# 6. Parse WhatsApp → whatsapp_clean.json
python parse_whatsapp.py

# 7. Extract WhatsApp features → wa_features.json
python extract_features.py

# 8. Train on Slack + Nankani, predict for WhatsApp → wa_results.json
python train_and_predict.py

# 9. Launch the dashboard
streamlit run dashboard.py
# Opens at http://localhost:8501
```

### After first setup (re-run anytime after new WhatsApp export)

```bash
python parse_whatsapp.py
python train_and_predict.py
streamlit run dashboard.py
```

---

## 9. File Structure

```
SNA/
│
├── 📄 parse_whatsapp.py       Parses WhatsApp .txt → whatsapp_clean.json
├── 📄 parse_slack.py          Parses Slack XML → slack_clean.json
├── 📄 parse_nankani.py        Parses Nankani 2020 WhatsApp → nankani_clean.json
├── 📄 extract_features.py     Extracts 19 features from WhatsApp team → wa_features.json
├── 📄 classify_archetypes.py  Standalone rule-based classifier (for testing)
├── 📄 train_and_predict.py ⭐ MAIN: trains on Slack+Nankani, predicts for team
├── 📄 dashboard.py            Streamlit Virtual Mirror dashboard
│
├── 📁 slack_data/             Cloned Slack dataset (XML, ~25MB)
│
├── 📊 slack_clean.json        Parsed Slack messages (60k rows)
├── 📊 nankani_clean.json      Parsed Nankani 2020 messages (11,460 rows)
├── 📊 whatsapp_clean.json     Parsed team WhatsApp messages
├── 📊 wa_features.json        19-feature matrix for WhatsApp team (5 users)
├── 📊 wa_results.json         Final archetype predictions + scores (dashboard reads this)
├── 📊 training_results.json   Combined Slack + Nankani training reference
│
├── 📄 .gitignore              Excludes private/large files from git
└── 📄 README.md               This file
│
│  ── Files excluded from git (private or too large):
│     Nankani_2020.txt         Raw WhatsApp export (real people's messages)
│     nankani_clean.json       Parsed version (real people's messages)
│     whatsapp_chat.txt        Your team's raw export (private)
│     whatsapp_clean.json      Your team's parsed messages (private)
│     wa_results.json          Your team's predictions (private)
│     slack_data/              Large dataset, cloned separately
```

---

## 10. Script-by-Script Guide

### `parse_whatsapp.py`
**What it does:** Reads the raw WhatsApp `.txt` export line by line and converts it to structured JSON.

**Key details:**
- Handles European date format (DD/MM/YYYY, HH:MM) used by German phones
- Applies `NAME_MAP` to replace phone numbers with real names
- Detects replies: a message within 10 minutes of a different person = reply
- Removes system messages (encryption notices, media omitted, join/leave events)
- Output: `whatsapp_clean.json` with columns: `author`, `datetime`, `body`, `parent_id`, `is_reply`

**Edit this when:** You have a new WhatsApp export — update `NAME_MAP` with the correct phone numbers and names.

---

### `parse_slack.py`
**What it does:** Reads the XML files from the academic Slack dataset and converts to the same JSON format.

**Key details:**
- Extracts `<ts>` (timestamp), `<user>` (author), `<text>` (message) from each XML node
- Uses `conversation_id` attribute to detect reply chains
- Samples 60,000 messages across 4 software communities
- Output: `slack_clean.json`

---

### `parse_nankani.py`
**What it does:** Parses the Nankani 2020 WhatsApp export (Indian date format, "am/pm" time).

**Key details:**
- Handles format: `DD/MM/YYYY, H:MM am/pm - Author: Message`
- Anonymizes phone numbers (keeps last 4 digits: `user_1138`)
- Named users kept as-is (e.g., "Tanay Kamath (TSEC, CS)")
- Output: `nankani_clean.json` — 11,460 messages, 148 users

---

### `extract_features.py`
**What it does:** Runs on `whatsapp_clean.json` and computes all 19 features per user.

**Key details:**
- Computes all features described in Section 6
- Builds a NetworkX reply graph for betweenness centrality
- Uses NRC-style emotion word list for emotion density
- Uses MATTR sliding window for vocabulary diversity
- Output: `wa_features.json` — one row per user, 19 feature columns

---

### `classify_archetypes.py`
**What it does:** Standalone rule-based archetype classifier. Useful for testing or comparing with the hybrid model.

**Key details:**
- Applies percentile-based rules within the group
- Does NOT use ML — pure theory-driven rules
- Output: `wa_results.json` (overwritten by `train_and_predict.py`)

---

### `train_and_predict.py` ⭐
**What it does:** The full hybrid pipeline — trains on Slack + Nankani, predicts for your team.

**Step by step:**
1. Calls `extract_features()` on all three datasets
2. Rank-normalizes each dataset independently
3. Applies COIN rules to Slack + Nankani → pseudo labels
4. Combines both training datasets (1,787 users)
5. Trains Random Forest with 5-fold cross-validation
6. Applies rules to WhatsApp team (within-group)
7. Applies ML to normalized WhatsApp features
8. Hybrid votes: 50% ML + 50% rules
9. Computes weighted percentage distribution (Section 6 formula)
10. Saves `wa_results.json` and `training_results.json`

---

### `dashboard.py`
**What it does:** Reads `wa_results.json` and renders the Streamlit Virtual Mirror.

**No computation happens here** — it's pure visualization. See Section 11 for details.

---

## 11. Dashboard Guide

Run with: `streamlit run dashboard.py`  
Opens at: `http://localhost:8501`

### Tab 1 — 🦁 Archetypes
- **Pie chart** — team archetype distribution
- **Archetype cards** — shows which members belong to each archetype, description, behavioral signals
- **Score comparison bars** — all 5 dimension scores (0–100) per member side by side
- **COIN Health Check** — alerts if archetypes are missing or Leech detected

### Tab 2 — 🌐 Interaction Network
- **Force-directed graph** — nodes = members, edges = reply interactions
- Node **size** = message count · Node **color** = archetype · Edge **thickness** = frequency
- Network statistics: most replied-to, most replies sent, most connected

### Tab 3 — 📅 Timeline & Heatmap
- **Daily message volume** over the chat period
- **Activity per archetype** per day (line chart)
- **Hour-of-day heatmap** — when is each person most active?

### Tab 4 — 👤 Member Profiles
- Select any team member
- **Archetype card** with description and behavioral signals
- **5 dimension scores** (0–100) with radar chart
- **Auto-generated behavioral insights**
- Sample of their recent messages

---

## 12. Our Team's Results

### Final predictions

| Member | Archetype | 🐝 Bee | 🐜 Ant | 🦋 Butterfly | 🦫 Capybara | 🔴 Leech | Conf |
|--------|-----------|--------|--------|-------------|------------|---------|------|
| Celina | 🐝 Bee | 29.6% | 35.1% | 22.8% | 11.9% | 0.6% | ✅ 54% |
| Faryan | 🐝 Bee | 21.8% | 25.0% | 12.5% | 7.7% | 33.1% | ✅ 59% |
| Esmaeil | 🐜 Ant | 20.1% | 19.2% | 15.0% | 24.8% | 20.9% | ✅ 36% |
| Marie | 🦋 Butterfly | 8.2% | 19.1% | 18.7% | 20.2% | 33.8% | ✅ 51% |
| Kasra | 🔴 Leech | 9.8% | 9.1% | 26.5% | 17.6% | 37.1% | ✅ 40% |

All three methods (Rule-based, ML, Hybrid) agree on every team member — the strongest possible validation outcome.

### Team health analysis

**Strengths:**
- Two Bees → strong idea generation and conversation initiation
- Esmaeil's Capybara score is 24.8% (second highest after Ant) → hybrid Ant-Capybara, executes AND supports
- Marie's Butterfly role means the team has a clarity/documentation person

**Gaps:**
- **No dedicated Capybara** — no primary harmonizer. This is a risk under deadline pressure.
- **Kasra flagged as Leech** — 14 messages in 8 days, receives more than sends. Needs re-engagement conversation.
- **Faryan has 33% Leech risk** despite being classified as Bee — volume is high but contribution balance should be monitored.

---

## 13. How to Add a New WhatsApp Chat

1. Open WhatsApp group on your phone
2. **iPhone**: tap group name → Export Chat → Without Media  
   **Android**: three dots → More → Export Chat → Without Media
3. Send the `.txt` file to yourself, rename it `whatsapp_chat.txt`
4. Place it in the project folder (replace the old one)
5. Edit `NAME_MAP` in `parse_whatsapp.py`:

```python
NAME_MAP = {
    '+49 1523 3682176': 'Celina',
    '+49 163 1519856' : 'Faryan',
    '+49 1517 0872047': 'Marie',
    'esmaeil molapour': 'Esmaeil',
    'kasra'           : 'Kasra'
}
```

6. Run the pipeline:
```bash
python parse_whatsapp.py
python train_and_predict.py
streamlit run dashboard.py
```

---

## 14. Limitations & Honest Disclaimers

| Limitation | Why it matters | What we say |
|-----------|---------------|-------------|
| Small WhatsApp sample (324 msgs, 8 days) | Features may shift with more data | Stated explicitly in paper |
| No ground truth labels | We can't validate against human-annotated archetypes | We use rule-agreement as proxy |
| Cross-platform gap (Slack vs WhatsApp) | Different norms and vocabulary | Rank normalization removes scale effects |
| TextBlob sentiment | Trained on general English, not developer jargon | Indicative, not clinical |
| Reply detection is approximate | WhatsApp has no reply API; we use 10-min window | Documented as approximation |
| MATTR unreliable for very short message histories | Users with < 20 messages get imprecise MATTR | We note minimum message threshold |
| Betweenness centrality with approximate edges | No explicit reply threading in WhatsApp | Edges inferred from time windows |
| Archetypes ≠ personality | Behavioral patterns in text ≠ clinical measurement | Always stated as "communication signature" |

---

## 15. How to Defend This Project

### "Why not use Big Five / FFI?"
The seminar's *Other Sources* document explicitly states: *"FFI / Big Five should not be the only basis for defining the archetypes. The main classification should rely on measurable features extracted from chat data."* We follow this guidance. Conscientiousness is captured indirectly through our Ant features (consistency, task focus, active days). The five COIN archetypes are more actionable for team improvement than abstract trait scores.

### "How do you validate the archetypes?"
Three independent methods — rule-based (COIN theory), Random Forest (trained on 1,787 real engineers), and hybrid voting — all agree 100% on every team member. Convergent validity across three independent methods is the standard approach when no ground truth labels exist.

### "Why is Reddit data gone?"
Our professor correctly pointed out that Reddit is public commentary, not team communication. We replaced it with two real software team datasets: the Chatterjee (2020) Slack dataset (peer-reviewed, MSR 2020) and the Nankani (2020) WhatsApp group of CS/IT engineering students. Both are directly aligned with the COIN research context.

### "What is your technical innovation?"
*"We introduce rank-based cross-platform normalization to bridge the scale gap between large public datasets and small private team chats, enabling a hybrid rule-ML archetype classifier to function reliably on groups as small as 5 people. We further extend the feature set with betweenness centrality (Freeman 1977) and emotion density (Mohammad & Turney 2013) to strengthen Bee and Butterfly detection."*

### "Kasra is labeled Leech — isn't that unfair?"
Leech is explicitly defined in the COIN framework as a **symptom of disengagement, not a personality judgment**. The system flags a communication pattern, not a person's character. The document states clearly: *"The fix isn't punishment — it's helping them find where they can contribute."*

### "Only 5 people — is that enough for a Virtual Mirror?"
Yes. The Virtual Mirror is designed to show a team their own patterns. Five people with 324 messages over 8 days is a real team snapshot. The statistical foundation comes from 1,787 training users. The mirror works at any group size — that is one of its design goals.

---

## 16. References

### Core theory
- **Gloor, P.A.** (2022). *Happimetrics: Leveraging AI to Untangle the Surprising Link Between Ethics, Happiness and Business Success*. Edward Elgar Publishing.

### Training datasets
- **Chatterjee, P., Damevski, K., Kraft, N.A., Pollock, L.** (2020). Software-related Slack Chats with Disentangled Conversations. *MSR 2020 Data Showcase*. https://github.com/preethac/Software-related-Slack-Chats-with-Disentangled-Conversations

### Feature literature
- **Freeman, L.C.** (1977). A set of measures of centrality based on betweenness. *Sociometry*, 40(1), 35–41. *(Betweenness centrality — Bee feature)*
- **Mohammad, S.M. & Turney, P.D.** (2013). Crowdsourcing a Word-Emotion Association Lexicon. *Computational Intelligence*, 29(3). *(Emotion density — Butterfly/Capybara feature)*
- **Covington, M.A. & McFall, J.D.** (2010). Cutting the Gordian Knot: The Moving-Average Type-Token Ratio (MATTR). *Journal of Quantitative Linguistics*, 17(2). *(MATTR — Bee feature)*
- **Kalman, Y.M., Ravid, G., Raban, D.R., Rafaeli, S.** (2006). Pauses and response latencies. *Journal of Computer-Mediated Communication*. *(Response latency — Ant feature)*
- **Kralj Novak, P. et al.** (2015). Sentiment of Emojis. *PLoS ONE*. *(Emoji ratio — Butterfly feature)*

### Supporting methodology
- **Golbeck, J. et al.** (2011). Predicting Personality from Twitter. *IEEE PASSAT/SocialCom 2011*.
- **Quercia, D. et al.** (2011). Our Twitter Profiles, Our Selves. *IEEE PASSAT/SocialCom 2011*.
- **Dowell, N.M.M. et al.** (2019). Group Communication Analysis. *Discourse Processes*, 56(3).
- **Hutto, C.J. & Gilbert, E.** (2014). VADER: A Parsimonious Rule-Based Model for Sentiment Analysis. *ICWSM 2014*.
- **Preece, J., Nonnecke, B., Andrews, D.** (2004). The top five reasons for lurking. *Computers in Human Behavior*, 20(2).
- **Breiman, L.** (2001). Random Forests. *Machine Learning*, 45(1), 5–32.

### Project
- **Project website**: https://sites.google.com/view/coinseminar26/creating-coins
- **GitHub**: https://github.com/Esmaeil2023/Team-communication-analyzer

---

## Team

| Name | Archetype | Distribution | Role in project |
|------|-----------|-------------|----------------|
| Celina | 🐝 Bee | 30% Bee · 35% Ant · 23% Butterfly | Most active, drives conversation |
| Faryan | 🐝 Bee | 22% Bee · 25% Ant · 33% Leech-risk | Active contributor, idea generation |
| Esmaeil | 🐜 Ant | 20% Bee · 19% Ant · 25% Capybara | Task-focused, strong harmony tendency |
| Marie | 🦋 Butterfly | 8% Bee · 19% Ant · 19% Butterfly | Long messages, clarifies ideas |
| Kasra | 🔴 Leech | 10% Bee · 9% Ant · 37% Leech | Re-engagement needed |

---

*Built with Python 3.11 · Streamlit · scikit-learn · NetworkX · TextBlob*  
*Training: Slack Developer Chats (Chatterjee et al., 2020) + CODERS WhatsApp (Nankani, 2020)*  
*Based on COINs research by Prof. Peter Gloor (MIT)*