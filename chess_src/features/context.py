import math
import pandas as pd


def _color_streak(history: list) -> int:
    """Returns length of current color streak: positive = whites, negative = blacks."""
    if not history:
        return 0
    last = history[-1]
    streak = 0
    for c in reversed(history):
        if c == last:
            streak += 1
        else:
            break
    return streak if last == "W" else -streak


def intra_tpr(history: list, default_elo: float = 2700.0) -> float:
    """TPR from (score, opp_elo) pairs accumulated within the current tournament."""
    if not history:
        return default_elo
    scores = [s for s, _ in history]
    opp_elos = [e for _, e in history]
    avg_score = sum(scores) / len(scores)
    avg_opp = sum(opp_elos) / len(opp_elos)
    if avg_score >= 1.0:
        return avg_opp + 400
    if avg_score <= 0.0:
        return avg_opp - 400
    return avg_opp + 400 * math.log10(avg_score / (1 - avg_score))


class ChessContextCalculator:
    def __init__(self, total_rounds: int = 14):
        self.total_rounds = total_rounds

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["round"] = (
            pd.to_numeric(df["round"], errors="coerce").fillna(1).astype(float)
        )
        df = df.sort_values(["tournament", "round"])

        for col in [
            "white_tournament_points",
            "black_tournament_points",
            "white_last2_score",
            "black_last2_score",
            "white_last3_score",
            "black_last3_score",
            "white_color_balance",
            "black_color_balance",
            "white_gap_to_leader",
            "black_gap_to_leader",
            "white_color_streak",
            "black_color_streak",
            "white_intra_tpr",
            "black_intra_tpr",
            "intra_tpr_diff",
        ]:
            df[col] = 0.0

        for tourney in df["tournament"].unique():
            t_mask = df["tournament"] == tourney
            t_df = df[t_mask].sort_values(["round", "played_at"])

            points = {}          # player_id -> cumulative points
            rounds_history = {}  # player_id -> [(round_num, score)]
            colors = {}          # player_id -> (whites_count, blacks_count)
            color_history = {}   # player_id -> list of 'W'/'B'
            intra_history = {}   # player_id -> [(score, opp_elo)]

            for idx, row in t_df.iterrows():
                w_id = row["white_id"]
                b_id = row["black_id"]
                round_num = row["round"]
                w_elo = float(row.get("white_elo", 2700.0))
                b_elo = float(row.get("black_elo", 2700.0))

                # ── Snapshot BEFORE this match ─────────────────────────────
                df.at[idx, "white_tournament_points"] = points.get(w_id, 0.0)
                df.at[idx, "black_tournament_points"] = points.get(b_id, 0.0)

                w_hist = sorted(rounds_history.get(w_id, []), key=lambda x: x[0])
                b_hist = sorted(rounds_history.get(b_id, []), key=lambda x: x[0])
                df.at[idx, "white_last2_score"] = sum(s for _, s in w_hist[-2:])
                df.at[idx, "black_last2_score"] = sum(s for _, s in b_hist[-2:])
                df.at[idx, "white_last3_score"] = sum(s for _, s in w_hist[-3:])
                df.at[idx, "black_last3_score"] = sum(s for _, s in b_hist[-3:])

                w_col = colors.get(w_id, (0, 0))
                b_col = colors.get(b_id, (0, 0))
                df.at[idx, "white_color_balance"] = w_col[0] - w_col[1]
                df.at[idx, "black_color_balance"] = b_col[0] - b_col[1]

                leader_score = max(points.values()) if points else 0.0
                df.at[idx, "white_gap_to_leader"] = leader_score - points.get(w_id, 0.0)
                df.at[idx, "black_gap_to_leader"] = leader_score - points.get(b_id, 0.0)

                df.at[idx, "white_color_streak"] = _color_streak(
                    color_history.get(w_id, [])
                )
                df.at[idx, "black_color_streak"] = _color_streak(
                    color_history.get(b_id, [])
                )

                w_itpr = intra_tpr(intra_history.get(w_id, []), default_elo=w_elo)
                b_itpr = intra_tpr(intra_history.get(b_id, []), default_elo=b_elo)
                df.at[idx, "white_intra_tpr"] = w_itpr
                df.at[idx, "black_intra_tpr"] = b_itpr
                df.at[idx, "intra_tpr_diff"] = w_itpr - b_itpr

                # ── Update state AFTER this match ──────────────────────────
                res = row["result"]
                if pd.notna(res):
                    res = float(res)
                    points[w_id] = points.get(w_id, 0.0) + res
                    points[b_id] = points.get(b_id, 0.0) + (1.0 - res)
                    rounds_history.setdefault(w_id, []).append((round_num, res))
                    rounds_history.setdefault(b_id, []).append((round_num, 1.0 - res))
                    w_whites, w_blacks = colors.get(w_id, (0, 0))
                    b_whites, b_blacks = colors.get(b_id, (0, 0))
                    colors[w_id] = (w_whites + 1, w_blacks)
                    colors[b_id] = (b_whites, b_blacks + 1)
                    color_history.setdefault(w_id, []).append("W")
                    color_history.setdefault(b_id, []).append("B")
                    intra_history.setdefault(w_id, []).append((res, b_elo))
                    intra_history.setdefault(b_id, []).append((1.0 - res, w_elo))

        df["tournament_points_diff"] = (
            df["white_tournament_points"] - df["black_tournament_points"]
        )
        df["round_norm"] = df["round"] / self.total_rounds
        df["is_closing_stage"] = (df["round"] >= (self.total_rounds - 3)).astype(int)

        return df
