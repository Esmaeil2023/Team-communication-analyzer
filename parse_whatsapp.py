import pandas as pd
import re

# ── Load your exported WhatsApp .txt file ────────────────────────────────
WHATSAPP_FILE = "whatsapp_chat.txt"

# ── Map phone numbers to real names ──────────────────────────────────────
NAME_MAP = {
    '+49 1523 3682176': 'Celina',
    '+49 163 1519856' : 'Faryan',
    '+49 1517 0872047': 'Marie',
    'esmaeil molapour': 'Esmaeil',
    'kasra'           : 'Kasra'
}

def parse_whatsapp(filepath):
    """Parse WhatsApp export into a clean DataFrame."""

    pattern = re.compile(
        r'(\d{1,2}[\/\.]\d{1,2}[\/\.]\d{2,4}),?\s'
        r'(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][Mm])?)\s?[-–]\s'
        r'([^:]+?):\s(.+)'
    )

    messages    = []
    current_msg = None

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line  = line.strip()
        match = pattern.match(line)

        if match:
            # Save previous message before starting new one
            if current_msg:
                messages.append(current_msg)

            date_str, time_str, author, body = match.groups()

            # ── Apply name mapping ────────────────────────────────────────
            author_raw   = author.strip()
            author_clean = NAME_MAP.get(
                author_raw,
                NAME_MAP.get(author_raw.lower(), author_raw)
            )

            current_msg = {
                'date_str': date_str,
                'time_str': time_str,
                'author'  : author_clean,
                'body'    : body.strip()
            }
        else:
            # Continuation of a multi-line message
            if current_msg and line:
                current_msg['body'] += ' ' + line

    # Don't forget the last message
    if current_msg:
        messages.append(current_msg)

    df = pd.DataFrame(messages)

    if len(df) == 0:
        print("❌ No messages parsed. Check your file format.")
        return df

    # ── Parse datetime ────────────────────────────────────────────────────
    df['datetime'] = pd.to_datetime(
        df['date_str'] + ' ' + df['time_str'],
        dayfirst=True,
        errors='coerce'
    )

    # ── Drop system messages ──────────────────────────────────────────────
    system_phrases = [
        'messages and calls are end-to-end encrypted',
        'changed the subject', 'added you', 'left',
        'changed this group', 'joined using this group',
        'you were added', 'image omitted', 'video omitted',
        'audio omitted', 'document omitted', 'sticker omitted',
        '<media omitted>', 'null'
    ]
    mask = df['body'].str.lower().apply(
        lambda x: not any(p in x for p in system_phrases)
    )
    df = df[mask]

    # ── Drop very short messages and bad rows ─────────────────────────────
    df = df[df['body'].str.len() >= 2]
    df = df.dropna(subset=['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    # ── Detect replies (within 10 mins from a different person) ──────────
    df['prev_time']   = df['datetime'].shift(1)
    df['prev_author'] = df['author'].shift(1)
    df['gap_mins']    = (
        df['datetime'] - df['prev_time']
    ).dt.total_seconds() / 60

    df['is_reply'] = (
        (df['gap_mins'] <= 10) &
        (df['prev_author'] != df['author'])
    ).astype(int)

    df['parent_id'] = df.apply(
        lambda r: f"t1_approx_{r.name - 1}"
                  if r['is_reply'] == 1
                  else f"t3_post_{r.name}",
        axis=1
    )

    # ── Final columns ─────────────────────────────────────────────────────
    df['subreddit'] = 'whatsapp_group'
    df['score']     = 1

    df = df[['author', 'datetime', 'body', 'parent_id',
             'is_reply', 'subreddit', 'score']]

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"✅ Parsed {len(df)} messages")
    print(f"👥 Unique users: {df['author'].nunique()}")
    print(f"📅 Date range: {df['datetime'].min()} → {df['datetime'].max()}")
    print(f"↩️  Reply rate: {df['is_reply'].mean() * 100:.1f}%")
    print(f"\nUsers found:")
    print(df['author'].value_counts())
    print(f"\nSample:")
    print(df[['author', 'datetime', 'body', 'is_reply']].head(5))

    return df


df = parse_whatsapp(WHATSAPP_FILE)

if len(df) > 0:
    df.to_json("whatsapp_clean.json", orient="records", lines=True)
    print(f"\n✅ Saved to whatsapp_clean.json")