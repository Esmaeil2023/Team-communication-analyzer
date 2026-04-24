import sqlite3
import pandas as pd

DB_PATH = "/Users/radio/Desktop/archive/database.sqlite"
conn = sqlite3.connect(DB_PATH)

print("Extracting r/programming...")
df1 = pd.read_sql_query("""
    SELECT author, created_utc, body, parent_id, subreddit, score
    FROM May2015
    WHERE subreddit = 'programming'
      AND author != '[deleted]'
      AND body NOT IN ('[deleted]', '[removed]')
    LIMIT 60000
""", conn)

print("Extracting r/learnprogramming...")
df2 = pd.read_sql_query("""
    SELECT author, created_utc, body, parent_id, subreddit, score
    FROM May2015
    WHERE subreddit = 'learnprogramming'
      AND author != '[deleted]'
      AND body NOT IN ('[deleted]', '[removed]')
    LIMIT 20000
""", conn)

conn.close()

df = pd.concat([df1, df2], ignore_index=True)

df.to_json("reddit_raw.json", orient="records", lines=True)
print(f"Total shape: {df.shape}")
print("Saved to reddit_raw.json")