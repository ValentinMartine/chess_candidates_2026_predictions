"""
Initialize the Women's Candidates 2026 database and populate all matches
(rounds 1-10 completed, rounds 11-14 upcoming).
"""

import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

DB_PATH = str(PROJECT_ROOT / "data" / "chess_matches_women.db")
TOURNAMENT = "Women's Candidates 2026"

PLAYERS = [
    ("Aleksandra Goryachkina", 4147103, "FID", 2560),
    ("Divya Deshmukh",         35006916, "IND", 2548),
    ("Kateryna Lagno",         14109336, "FID", 2530),
    ("Zhu Jiner",              8608059,  "CHN", 2528),
    ("Vaishali Rameshbabu",    5091756,  "IND", 2522),
    ("Anna Muzychuk",          14111330, "UKR", 2518),
    ("Tan Zhongyi",            8603642,  "CHN", 2510),
    ("Bibisara Assaubayeva",   13708694, "KAZ", 2478),
]

# (white, black, result_or_None, date, round)
# result: 1.0=white wins, 0.5=draw, 0.0=black wins, None=upcoming
MATCHES = [
    # Round 1 - 2026-04-01
    ("Divya Deshmukh",       "Anna Muzychuk",          0.5, "2026-04-01", 1),
    ("Vaishali Rameshbabu",  "Bibisara Assaubayeva",   0.5, "2026-04-01", 1),
    ("Aleksandra Goryachkina","Kateryna Lagno",         0.5, "2026-04-01", 1),
    ("Zhu Jiner",            "Tan Zhongyi",            0.5, "2026-04-01", 1),
    # Round 2 - 2026-04-02
    ("Anna Muzychuk",        "Tan Zhongyi",            0.5, "2026-04-02", 2),
    ("Kateryna Lagno",       "Zhu Jiner",              0.5, "2026-04-02", 2),
    ("Bibisara Assaubayeva", "Aleksandra Goryachkina", 0.5, "2026-04-02", 2),
    ("Divya Deshmukh",       "Vaishali Rameshbabu",    0.5, "2026-04-02", 2),
    # Round 3 - 2026-04-03
    ("Vaishali Rameshbabu",  "Anna Muzychuk",          0.5, "2026-04-03", 3),
    ("Aleksandra Goryachkina","Divya Deshmukh",        0.5, "2026-04-03", 3),
    ("Zhu Jiner",            "Bibisara Assaubayeva",   0.0, "2026-04-03", 3),
    ("Tan Zhongyi",          "Kateryna Lagno",         0.0, "2026-04-03", 3),
    # Round 4 - 2026-04-04
    ("Anna Muzychuk",        "Kateryna Lagno",         1.0, "2026-04-04", 4),
    ("Bibisara Assaubayeva", "Tan Zhongyi",            0.5, "2026-04-04", 4),
    ("Divya Deshmukh",       "Zhu Jiner",              0.0, "2026-04-04", 4),
    ("Vaishali Rameshbabu",  "Aleksandra Goryachkina", 0.5, "2026-04-04", 4),
    # Round 5 - 2026-04-05
    ("Aleksandra Goryachkina","Anna Muzychuk",         0.5, "2026-04-05", 5),
    ("Zhu Jiner",            "Vaishali Rameshbabu",    1.0, "2026-04-05", 5),
    ("Tan Zhongyi",          "Divya Deshmukh",         0.5, "2026-04-05", 5),
    ("Kateryna Lagno",       "Bibisara Assaubayeva",   1.0, "2026-04-05", 5),
    # Round 6 - 2026-04-07
    ("Zhu Jiner",            "Anna Muzychuk",          0.0, "2026-04-07", 6),
    ("Tan Zhongyi",          "Aleksandra Goryachkina", 0.5, "2026-04-07", 6),
    ("Kateryna Lagno",       "Vaishali Rameshbabu",    0.0, "2026-04-07", 6),
    ("Bibisara Assaubayeva", "Divya Deshmukh",         0.0, "2026-04-07", 6),
    # Round 7 - 2026-04-08
    ("Anna Muzychuk",        "Bibisara Assaubayeva",   0.5, "2026-04-08", 7),
    ("Divya Deshmukh",       "Kateryna Lagno",         0.5, "2026-04-08", 7),
    ("Vaishali Rameshbabu",  "Tan Zhongyi",            1.0, "2026-04-08", 7),
    ("Aleksandra Goryachkina","Zhu Jiner",             0.5, "2026-04-08", 7),
    # Round 8 - 2026-04-09
    ("Anna Muzychuk",        "Divya Deshmukh",         0.0, "2026-04-09", 8),
    ("Bibisara Assaubayeva", "Vaishali Rameshbabu",    0.5, "2026-04-09", 8),
    ("Kateryna Lagno",       "Aleksandra Goryachkina", 1.0, "2026-04-09", 8),
    ("Tan Zhongyi",          "Zhu Jiner",              0.0, "2026-04-09", 8),
    # Round 9 - 2026-04-10
    ("Tan Zhongyi",          "Anna Muzychuk",          0.5, "2026-04-10", 9),
    ("Zhu Jiner",            "Kateryna Lagno",         1.0, "2026-04-10", 9),
    ("Aleksandra Goryachkina","Bibisara Assaubayeva",  0.5, "2026-04-10", 9),
    ("Vaishali Rameshbabu",  "Divya Deshmukh",         1.0, "2026-04-10", 9),
    # Round 10 - 2026-04-11 (2 completed, 2 upcoming)
    ("Anna Muzychuk",        "Vaishali Rameshbabu",    0.5, "2026-04-11", 10),
    ("Divya Deshmukh",       "Aleksandra Goryachkina", None,"2026-04-11", 10),
    ("Bibisara Assaubayeva", "Zhu Jiner",              None,"2026-04-11", 10),
    ("Kateryna Lagno",       "Tan Zhongyi",            0.5, "2026-04-11", 10),
    # Round 11 - 2026-04-12 (upcoming)
    ("Kateryna Lagno",       "Anna Muzychuk",          None,"2026-04-12", 11),
    ("Tan Zhongyi",          "Bibisara Assaubayeva",   None,"2026-04-12", 11),
    ("Zhu Jiner",            "Divya Deshmukh",         None,"2026-04-12", 11),
    ("Aleksandra Goryachkina","Vaishali Rameshbabu",   None,"2026-04-12", 11),
    # Round 12 - 2026-04-14 (upcoming)
    ("Anna Muzychuk",        "Aleksandra Goryachkina", None,"2026-04-14", 12),
    ("Vaishali Rameshbabu",  "Zhu Jiner",              None,"2026-04-14", 12),
    ("Divya Deshmukh",       "Tan Zhongyi",            None,"2026-04-14", 12),
    ("Bibisara Assaubayeva", "Kateryna Lagno",         None,"2026-04-14", 12),
    # Round 13 - 2026-04-15 (upcoming)
    ("Bibisara Assaubayeva", "Anna Muzychuk",          None,"2026-04-15", 13),
    ("Kateryna Lagno",       "Divya Deshmukh",         None,"2026-04-15", 13),
    ("Tan Zhongyi",          "Vaishali Rameshbabu",    None,"2026-04-15", 13),
    ("Zhu Jiner",            "Aleksandra Goryachkina", None,"2026-04-15", 13),
    # Round 14 - 2026-04-16 (upcoming)
    ("Anna Muzychuk",        "Zhu Jiner",              None,"2026-04-16", 14),
    ("Aleksandra Goryachkina","Tan Zhongyi",           None,"2026-04-16", 14),
    ("Vaishali Rameshbabu",  "Kateryna Lagno",         None,"2026-04-16", 14),
    ("Divya Deshmukh",       "Bibisara Assaubayeva",   None,"2026-04-16", 14),
]


def init_db(conn):
    c = conn.cursor()
    # Same schema as men's DB: id IS the FIDE ID for candidates, auto-increment for opponents
    c.execute("""
    CREATE TABLE IF NOT EXISTS players (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        country TEXT,
        rating_initial INTEGER
    )""")
    c.execute("""
    CREATE TABLE IF NOT EXISTS matches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        white_id INTEGER NOT NULL,
        black_id INTEGER NOT NULL,
        result REAL,
        played_at DATE,
        tournament TEXT,
        round INTEGER,
        is_candidates INTEGER DEFAULT 0,
        FOREIGN KEY (white_id) REFERENCES players(id),
        FOREIGN KEY (black_id) REFERENCES players(id)
    )""")
    conn.commit()


def insert_players(conn):
    c = conn.cursor()
    for name, fide_id, country, rating in PLAYERS:
        # Use FIDE ID as primary key (mirrors men's DB)
        c.execute("""
            INSERT OR IGNORE INTO players (id, name, country, rating_initial)
            VALUES (?, ?, ?, ?)
        """, (fide_id, name, country, rating))
    conn.commit()


def get_id(conn, name):
    # Look up by name → returns the FIDE ID (which is the primary key)
    row = conn.execute("SELECT id FROM players WHERE name = ?", (name,)).fetchone()
    if not row:
        raise ValueError(f"Player not found: {name}")
    return row[0]


def insert_matches(conn):
    c = conn.cursor()
    count = 0
    for white, black, result, date, rnd in MATCHES:
        w_id = get_id(conn, white)
        b_id = get_id(conn, black)
        c.execute("""
            INSERT INTO matches (white_id, black_id, result, played_at, tournament, round, is_candidates)
            VALUES (?, ?, ?, ?, ?, ?, 1)
        """, (w_id, b_id, result, date, TOURNAMENT, rnd))
        count += 1
    conn.commit()
    print(f"Inserted {count} matches.")


if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)
    insert_players(conn)
    insert_matches(conn)
    conn.close()
    print(f"Women's DB ready at {DB_PATH}")
