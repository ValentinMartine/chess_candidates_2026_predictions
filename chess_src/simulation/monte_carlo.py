import math
import numpy as np
import pandas as pd
from chess_src.models.lgbm_model import ChessLGBMModel
from chess_src.features.pipeline import ChessFeaturePipeline
from chess_src.features.context import intra_tpr

TOTAL_ROUNDS = 14

# Context features recomputed dynamically per simulation round
DYNAMIC_CONTEXT_COLS = {
    "white_tournament_points",
    "black_tournament_points",
    "tournament_points_diff",
    "white_gap_to_leader",
    "black_gap_to_leader",
    "white_last2_score",
    "black_last2_score",
    "white_last3_score",
    "black_last3_score",
    "white_intra_tpr",
    "black_intra_tpr",
    "intra_tpr_diff",
}


class CandidatesSimulator:
    def __init__(
        self,
        model: ChessLGBMModel,
        pipeline: ChessFeaturePipeline,
        players_config: list,
        num_simulations: int = 1000,
    ):
        self.model = model
        self.pipeline = pipeline
        self.players = players_config
        self.num_simulations = num_simulations
        self.elo_map = {p["fide_id"]: float(p["rating_april_2006"]) for p in players_config}

    def simulate(
        self, current_matches: pd.DataFrame, remaining_matches: pd.DataFrame
    ) -> dict:
        p_ids = [p["fide_id"] for p in self.players]
        win_counts = {pid: 0 for pid in p_ids}

        # ── Build history from completed matches ───────────────────────────
        history_points = {pid: 0.0 for pid in p_ids}
        history_intra = {pid: [] for pid in p_ids}  # [(score, opp_elo)]
        history_rounds = {pid: [] for pid in p_ids}  # [(round_num, score)]

        for _, r in current_matches.iterrows():
            w_id = int(r["white_id"])
            b_id = int(r["black_id"])
            res = float(r["result"])
            rnd = float(r.get("round", 1))
            w_elo = self.elo_map.get(w_id, 2700.0)
            b_elo = self.elo_map.get(b_id, 2700.0)
            history_points[w_id] = history_points.get(w_id, 0.0) + res
            history_points[b_id] = history_points.get(b_id, 0.0) + (1.0 - res)
            history_intra[w_id].append((res, b_elo))
            history_intra[b_id].append((1.0 - res, w_elo))
            history_rounds[w_id].append((rnd, res))
            history_rounds[b_id].append((rnd, 1.0 - res))

        # ── Compute static features once ───────────────────────────────────
        full_df = pd.concat([current_matches, remaining_matches], ignore_index=True)
        full_df["white_id"] = full_df["white_id"].astype(int)
        full_df["black_id"] = full_df["black_id"].astype(int)
        full_df["result"] = pd.to_numeric(full_df["result"], errors="coerce")
        df_proc = self.pipeline.process(full_df)

        rem_df = df_proc[df_proc["result"].isna()].copy()
        rem_df = rem_df.sort_values(["round", "white_id"]).reset_index(drop=True)

        if rem_df.empty:
            winner_id = max(history_points, key=history_points.get)
            return {pid: (1.0 if pid == winner_id else 0.0) for pid in p_ids}

        rounds_grouped = [
            (rnd, grp.reset_index(drop=True)) for rnd, grp in rem_df.groupby("round")
        ]
        outcomes = [0.0, 0.5, 1.0]

        for _ in range(self.num_simulations):
            sim_scores = history_points.copy()
            sim_intra = {pid: list(history_intra[pid]) for pid in p_ids}
            sim_rounds = {pid: list(history_rounds[pid]) for pid in p_ids}

            for rnd, rnd_matches in rounds_grouped:
                batch = rnd_matches[self.model.feature_cols].copy()

                # ── Override dynamic context features ──────────────────────
                w_ids = rnd_matches["white_id"].values
                b_ids = rnd_matches["black_id"].values

                w_pts = np.array([sim_scores.get(int(i), 0.0) for i in w_ids])
                b_pts = np.array([sim_scores.get(int(i), 0.0) for i in b_ids])
                leader = max(sim_scores.values()) if sim_scores else 0.0

                batch["white_tournament_points"] = w_pts
                batch["black_tournament_points"] = b_pts
                batch["tournament_points_diff"] = w_pts - b_pts
                batch["white_gap_to_leader"] = leader - w_pts
                batch["black_gap_to_leader"] = leader - b_pts

                def last_n(pid, n):
                    h = sorted(sim_rounds.get(int(pid), []), key=lambda x: x[0])
                    return sum(s for _, s in h[-n:])

                batch["white_last2_score"] = [last_n(i, 2) for i in w_ids]
                batch["black_last2_score"] = [last_n(i, 2) for i in b_ids]
                batch["white_last3_score"] = [last_n(i, 3) for i in w_ids]
                batch["black_last3_score"] = [last_n(i, 3) for i in b_ids]

                w_itprs = np.array([
                    intra_tpr(sim_intra.get(int(i), []), self.elo_map.get(int(i), 2700.0))
                    for i in w_ids
                ])
                b_itprs = np.array([
                    intra_tpr(sim_intra.get(int(i), []), self.elo_map.get(int(i), 2700.0))
                    for i in b_ids
                ])
                batch["white_intra_tpr"] = w_itprs
                batch["black_intra_tpr"] = b_itprs
                batch["intra_tpr_diff"] = w_itprs - b_itprs

                probs_batch = self.model.predict_proba(batch)

                for i, row in rnd_matches.iterrows():
                    p = np.array(probs_batch[i], dtype=np.float64)
                    p /= p.sum()
                    res = float(np.random.choice(outcomes, p=p))
                    w_id = int(row["white_id"])
                    b_id = int(row["black_id"])
                    w_elo = self.elo_map.get(w_id, 2700.0)
                    b_elo = self.elo_map.get(b_id, 2700.0)
                    sim_scores[w_id] = sim_scores.get(w_id, 0.0) + res
                    sim_scores[b_id] = sim_scores.get(b_id, 0.0) + (1.0 - res)
                    sim_rounds.setdefault(w_id, []).append((rnd, res))
                    sim_rounds.setdefault(b_id, []).append((rnd, 1.0 - res))
                    sim_intra.setdefault(w_id, []).append((res, b_elo))
                    sim_intra.setdefault(b_id, []).append((1.0 - res, w_elo))

            max_s = max(sim_scores[pid] for pid in p_ids)
            winners = [pid for pid in p_ids if sim_scores[pid] == max_s]
            win_counts[np.random.choice(winners)] += 1

        return {pid: win_counts[pid] / self.num_simulations for pid in p_ids}
