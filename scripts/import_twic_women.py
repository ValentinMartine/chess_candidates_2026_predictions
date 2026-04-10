"""
Import classical games involving the 8 Women's Candidates 2026 players from TWIC PGN files.
Uses the same regex-based header parsing as import_twic_candidates.py.
"""

import re
import sqlite3
import sys
from pathlib import Path
from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

DB_PATH = str(PROJECT_ROOT / "data" / "chess_matches_women.db")
TWIC_DIR = PROJECT_ROOT / "data" / "raw" / "twic"

# TWIC name variants → (canonical DB name, fide_id)
# TWIC uses "Lastname,F" or "Lastname,Firstname" format
CANDIDATE_TWIC_NAMES = {
    # Aleksandra Goryachkina
    "Goryachkina,Aleksandra": ("Aleksandra Goryachkina", 4147103),
    "Goryachkina,A":          ("Aleksandra Goryachkina", 4147103),
    # Divya Deshmukh
    "Deshmukh,Divya":         ("Divya Deshmukh", 35006916),
    "Deshmukh,D":             ("Divya Deshmukh", 35006916),
    # Kateryna Lagno
    "Lagno,Kateryna":         ("Kateryna Lagno", 14109336),
    "Lagno,K":                ("Kateryna Lagno", 14109336),
    # Zhu Jiner
    "Zhu Jiner":              ("Zhu Jiner", 8608059),
    "Zhu,Jiner":              ("Zhu Jiner", 8608059),
    # Vaishali Rameshbabu
    "Vaishali,R":             ("Vaishali Rameshbabu", 5091756),
    "Vaishali Rameshbabu":    ("Vaishali Rameshbabu", 5091756),
    "Vaishali,Rameshbabu":    ("Vaishali Rameshbabu", 5091756),
    # Anna Muzychuk
    "Muzychuk,Anna":          ("Anna Muzychuk", 14111330),
    "Muzychuk,A":             ("Anna Muzychuk", 14111330),
    # Tan Zhongyi
    "Tan Zhongyi":            ("Tan Zhongyi", 8603642),
    "Tan,Zhongyi":            ("Tan Zhongyi", 8603642),
    # Bibisara Assaubayeva
    "Assaubayeva,Bibisara":   ("Bibisara Assaubayeva", 13708694),
    "Assaubayeva,B":          ("Bibisara Assaubayeva", 13708694),
}

SKIP_KEYWORDS = ["blitz", "rapid", "bullet", "armageddon", "960", "fischer"]
RESULT_MAP = {"1-0": 1.0, "1/2-1/2": 0.5, "0-1": 0.0}
MIN_OPPONENT_ELO = 2200  # Lower threshold for women's games

_TAG = re.compile(r'\[(\w+)\s+"([^"]+)"\]')


def parse_headers(game_text: str) -> dict:
    return {m.group(1): m.group(2) for m in _TAG.finditer(game_text)}


def get_or_create_player(cursor, name: str, rating: int | None) -> int:
    row = cursor.execute("SELECT id FROM players WHERE name = ?", (name,)).fetchone()
    if row:
        return row[0]
    cursor.execute(
        "INSERT INTO players (name, rating_initial) VALUES (?, ?)",
        (name, rating),
    )
    return cursor.lastrowid


def already_exists(cursor, white_id, black_id, played_at, tournament) -> bool:
    return cursor.execute(
        "SELECT 1 FROM matches WHERE white_id=? AND black_id=? AND played_at=? AND tournament=?",
        (white_id, black_id, played_at, tournament),
    ).fetchone() is not None


def import_file(pgn_path: Path, conn: sqlite3.Connection) -> tuple[int, int]:
    cursor = conn.cursor()
    content = pgn_path.read_text(encoding="utf-8", errors="replace")
    games = re.split(r"\n(?=\[Event )", content)

    imported = skipped = 0
    candidate_names = set(CANDIDATE_TWIC_NAMES.keys())

    for raw in games:
        if not any(name in raw for name in candidate_names):
            skipped += 1
            continue

        h = parse_headers(raw)
        white_twic = h.get("White", "")
        black_twic = h.get("Black", "")

        w_is_cand = white_twic in CANDIDATE_TWIC_NAMES
        b_is_cand = black_twic in CANDIDATE_TWIC_NAMES
        if not (w_is_cand or b_is_cand):
            skipped += 1
            continue

        result = RESULT_MAP.get(h.get("Result", ""), None)
        if result is None:
            skipped += 1
            continue

        event = h.get("Event", "Unknown")
        if any(kw in event.lower() for kw in SKIP_KEYWORDS):
            skipped += 1
            continue

        # Skip Women's Candidates 2026 — already in DB
        if "Candidates" in event and "2026" in event:
            skipped += 1
            continue

        try:
            white_elo = int(h.get("WhiteElo", 0)) or None
            black_elo = int(h.get("BlackElo", 0)) or None
        except ValueError:
            white_elo = black_elo = None

        opp_elo = black_elo if w_is_cand else white_elo
        if opp_elo and opp_elo < MIN_OPPONENT_ELO:
            skipped += 1
            continue

        raw_date = h.get("Date", "").replace(".", "-")
        if "?" in raw_date or not raw_date:
            skipped += 1
            continue

        round_num = h.get("Round", "1")
        try:
            round_num = int(float(round_num))
        except (ValueError, TypeError):
            round_num = 1

        if w_is_cand:
            _, w_id = CANDIDATE_TWIC_NAMES[white_twic]
            b_name = black_twic
            b_id = get_or_create_player(cursor, b_name, black_elo)
        else:
            w_name = white_twic
            w_id = get_or_create_player(cursor, w_name, white_elo)
            _, b_id = CANDIDATE_TWIC_NAMES[black_twic]

        if already_exists(cursor, w_id, b_id, raw_date, event):
            skipped += 1
            continue

        cursor.execute(
            "INSERT INTO matches (white_id, black_id, result, played_at, tournament, round) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (w_id, b_id, result, raw_date, event, round_num),
        )
        imported += 1

    conn.commit()
    return imported, skipped


def main():
    pgn_files = sorted(TWIC_DIR.glob("twic*.pgn"))
    if not pgn_files:
        logger.error(f"No TWIC PGN files found in {TWIC_DIR}")
        return

    logger.info(f"Found {len(pgn_files)} TWIC files. Starting import...")
    conn = sqlite3.connect(DB_PATH)
    total_imported = total_skipped = 0

    for pgn in pgn_files:
        imp, skp = import_file(pgn, conn)
        if imp > 0:
            logger.info(f"{pgn.name}: +{imp} imported, {skp} skipped")
        total_imported += imp
        total_skipped += skp

    conn.close()
    logger.success(f"Done — total imported: {total_imported}, skipped: {total_skipped}")

    # Summary by tournament
    conn2 = sqlite3.connect(DB_PATH)
    import pandas as pd
    df = pd.read_sql_query(
        "SELECT tournament, COUNT(*) cnt FROM matches "
        "WHERE result IS NOT NULL AND tournament != \"Women's Candidates 2026\" "
        "GROUP BY tournament ORDER BY cnt DESC LIMIT 20",
        conn2,
    )
    conn2.close()
    logger.info(f"\nTop tournaments in DB:\n{df.to_string(index=False)}")


if __name__ == "__main__":
    main()
