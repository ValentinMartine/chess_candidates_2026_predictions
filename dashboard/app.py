import streamlit as st
import pandas as pd
import sqlite3
import yaml
import json
import sys
import os
import numpy as np
from pathlib import Path

# Pathing and namespace mapping for unpickling
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "chess_src"))

import chess_src.models.lgbm_model

sys.modules["src.models.lgbm_model"] = chess_src.models.lgbm_model

from chess_src.models.lgbm_model import ChessLGBMModel
from chess_src.features.pipeline import ChessFeaturePipeline
from chess_src.simulation.monte_carlo import CandidatesSimulator

# ── Paths ──────────────────────────────────────────────────────────────────────
DB_MEN   = str(PROJECT_ROOT / "data" / "chess_matches.db")
DB_WOMEN = str(PROJECT_ROOT / "data" / "chess_matches_women.db")
MODEL_MEN   = str(PROJECT_ROOT / "data" / "chess_lgbm.pkl")
MODEL_WOMEN = str(PROJECT_ROOT / "data" / "chess_lgbm_women.pkl")
CONFIG_MEN   = str(PROJECT_ROOT / "config" / "settings.yaml")
CONFIG_WOMEN = str(PROJECT_ROOT / "config" / "settings_women.yaml")

# FIDE IDs
FIDE_MEN = [2004887, 2020009, 8603405, 24116068, 14205481, 25059650, 4661654, 24175439]
FIDE_WOMEN = [4147103, 35006916, 14109336, 8608059, 5091756, 14111330, 8603642, 13708694]

TOURNAMENT_MEN   = "Candidates 2026"
TOURNAMENT_WOMEN = "Women's Candidates 2026"


# ── Shared helpers ─────────────────────────────────────────────────────────────

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def sql_ids(ids):
    return ",".join(str(i) for i in ids)


def get_standings(db_path, fide_ids, tournament):
    conn = sqlite3.connect(db_path)
    ids_sql = sql_ids(fide_ids)
    query = f"""
        SELECT
            p.name AS Joueur,
            p.country AS Pays,
            CAST(p.rating_initial AS INTEGER) AS Elo,
            ROUND(
                COALESCE(SUM(
                    CASE
                        WHEN m.result = 1.0 AND m.white_id = p.id THEN 1.0
                        WHEN m.result = 0.0 AND m.black_id = p.id THEN 1.0
                        WHEN m.result = 0.5                        THEN 0.5
                        ELSE 0.0
                    END
                ), 0), 1
            ) AS Points
        FROM players p
        LEFT JOIN matches m
            ON (p.id = m.white_id OR p.id = m.black_id)
            AND m.result IS NOT NULL
            AND m.tournament = ?
        WHERE p.id IN ({ids_sql})
        GROUP BY p.id
        ORDER BY Points DESC, Elo DESC
    """
    df = pd.read_sql_query(query, conn, params=(tournament,))
    conn.close()
    return df


def get_completed(db_path, fide_ids, tournament):
    conn = sqlite3.connect(db_path)
    ids_sql = sql_ids(fide_ids)
    query = f"""
        SELECT m.id, m.round, m.white_id, m.black_id, m.result, m.played_at, m.tournament
        FROM matches m
        WHERE m.result IS NOT NULL
          AND m.tournament = ?
          AND m.white_id IN ({ids_sql})
          AND m.black_id IN ({ids_sql})
        ORDER BY m.round ASC, m.id ASC
    """
    df = pd.read_sql_query(query, conn, params=(tournament,))
    conn.close()
    return df


def get_upcoming(db_path, fide_ids, tournament):
    conn = sqlite3.connect(db_path)
    ids_sql = sql_ids(fide_ids)
    query = f"""
        SELECT
            m.id, m.round,
            pw.name AS white, pb.name AS black,
            m.white_id, m.black_id, m.played_at, m.tournament
        FROM matches m
        JOIN players pw ON m.white_id = pw.id
        JOIN players pb ON m.black_id = pb.id
        WHERE m.result IS NULL
          AND m.tournament = ?
          AND m.white_id IN ({ids_sql})
          AND m.black_id IN ({ids_sql})
        ORDER BY m.round ASC, m.id ASC
    """
    df = pd.read_sql_query(query, conn, params=(tournament,))
    conn.close()
    return df


# ── Shared UI block ────────────────────────────────────────────────────────────

def render_tournament_tab(db_path, fide_ids, tournament, model_path, config_path):
    config = load_config(config_path)
    col1, col2 = st.columns([1, 2])

    # Standings
    with col1:
        st.subheader("📊 Classement actuel")
        standings = get_standings(db_path, fide_ids, tournament)
        standings.insert(0, "#", range(1, len(standings) + 1))
        st.dataframe(standings, hide_index=True, use_container_width=True)

    # Match predictions
    with col2:
        st.subheader("🔮 Probabilités - prochains matchs")

        if not os.path.exists(model_path):
            st.warning("Modèle introuvable. Lancer l'entraînement d'abord.")
            return

        try:
            model = ChessLGBMModel.load(model_path)

            # Load feature list from sidecar if available (women's model)
            sidecar = Path(model_path).with_suffix(".features.json")
            if sidecar.exists():
                model.feature_cols = json.loads(sidecar.read_text(encoding="utf-8"))

            pipeline = ChessFeaturePipeline(config, db_path=db_path)
            completed = get_completed(db_path, fide_ids, tournament)
            upcoming  = get_upcoming(db_path, fide_ids, tournament)

            if upcoming.empty:
                st.info("Tournoi terminé !")
                return

            upcoming_for_pipeline = upcoming[
                ["id", "round", "white_id", "black_id", "played_at", "tournament"]
            ].copy()
            upcoming_for_pipeline["result"] = np.nan

            all_matches   = pd.concat([completed, upcoming_for_pipeline], ignore_index=True)
            all_features  = pipeline.process(all_matches)
            upcoming_feat = all_features[all_features["result"].isna()].copy()

            if upcoming_feat.empty:
                st.warning("Aucune prédiction disponible.")
                return

            probs = model.predict_proba(upcoming_feat)
            pred_df = pd.DataFrame(probs, columns=["P_Noir", "P_Nulle", "P_Blanc"])
            pred_df["id"]    = upcoming_feat["id"].values
            pred_df["Ronde"] = upcoming_feat["round"].values.astype(int)

            final_display = pd.merge(
                upcoming[["id", "white", "black"]], pred_df, on="id"
            ).sort_values("Ronde").reset_index(drop=True)

            display_df = final_display[
                ["Ronde", "white", "black", "P_Blanc", "P_Nulle", "P_Noir"]
            ].rename(columns={"white": "Blancs", "black": "Noirs"})
            display_df.columns = ["Ronde", "Blancs", "Noirs", "Victoire Blancs", "Nulle", "Victoire Noirs"]

            rounds = display_df["Ronde"].unique()
            round_parity = {r: i % 2 for i, r in enumerate(sorted(rounds))}
            ROW_ODD  = "background-color: #f0f2f6; color: #000000"
            ROW_EVEN = "background-color: #ffffff; color: #000000"
            GREEN  = "background-color: #c8f0c8; color: #1a5c1a; font-weight: bold"
            ORANGE = "background-color: #ffe5a0; color: #7a4f00; font-weight: bold"
            RED    = "background-color: #f0c8c8; color: #7a1a1a; font-weight: bold"

            def style_row(row):
                base = ROW_ODD if round_parity.get(row["Ronde"], 0) else ROW_EVEN
                pb, pn, pk = row["Victoire Blancs"], row["Nulle"], row["Victoire Noirs"]
                ranked = sorted(
                    [("pb", pb), ("pn", pn), ("pk", pk)],
                    key=lambda x: x[1], reverse=True
                )
                palette = [GREEN, ORANGE, RED]
                color_map = {name: palette[i] for i, (name, _) in enumerate(ranked)}
                return [base, base, base, color_map["pb"], color_map["pn"], color_map["pk"]]

            styled = (
                display_df.style
                .apply(style_row, axis=1)
                .format({
                    "Victoire Blancs": "{:.1%}",
                    "Nulle": "{:.1%}",
                    "Victoire Noirs": "{:.1%}",
                })
            )
            st.dataframe(styled, hide_index=True, use_container_width=True)

        except Exception as e:
            st.error(f"Erreur d'affichage : {e}")
            import traceback
            st.code(traceback.format_exc())

    # Monte Carlo
    st.divider()
    st.subheader("📈 Prévision du tournoi - simulation Monte Carlo")

    if st.button("🚀 Lancer 1 000 simulations", key=f"mc_{tournament}"):
        with st.spinner("Simulation en cours…"):
            try:
                _model = ChessLGBMModel.load(model_path)
                sidecar = Path(model_path).with_suffix(".features.json")
                if sidecar.exists():
                    _model.feature_cols = json.loads(sidecar.read_text(encoding="utf-8"))

                _pipeline  = ChessFeaturePipeline(config, db_path=db_path)
                _players   = config["players"]

                completed_t = get_completed(db_path, fide_ids, tournament)
                upcoming_t  = get_upcoming(db_path, fide_ids, tournament)
                upcoming_t["result"] = np.nan

                n_sims = 1000
                simulator = CandidatesSimulator(
                    _model, _pipeline, _players, num_simulations=n_sims
                )
                raw_results = simulator.simulate(completed_t, upcoming_t)

                id_to_name = {p["fide_id"]: p["name"] for p in _players}
                rows = []
                for pid, prob in raw_results.items():
                    se = np.sqrt(prob * (1 - prob) / n_sims)
                    lo = max(0.0, prob - 1.96 * se)
                    hi = min(1.0, prob + 1.96 * se)
                    rows.append({
                        "Joueur": id_to_name.get(pid, str(pid)),
                        "Estimation": round(prob * 100, 1),
                        "Fourchette (95%)": f"{lo*100:.0f}% – {hi*100:.0f}%",
                    })
                res_df = (
                    pd.DataFrame(rows)
                    .sort_values("Estimation", ascending=False)
                    .reset_index(drop=True)
                )
                res_df.insert(0, "#", range(1, len(res_df) + 1))

                col_a, col_b = st.columns([1, 1])
                with col_a:
                    st.dataframe(res_df, hide_index=True, use_container_width=True)
                with col_b:
                    st.bar_chart(res_df.set_index("Joueur")["Estimation"])

            except Exception as e:
                st.error(f"Erreur de simulation : {e}")
                import traceback
                st.code(traceback.format_exc())


# ── App ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="FIDE Candidates 2026", layout="wide")
st.title("♟️ FIDE Candidates 2026 - Dashboard prédictif")
st.markdown("Classement en temps réel et prédictions pour les rondes restantes.")

tab_men, tab_women = st.tabs(["👨 Tournoi Masculin", "👩 Tournoi Féminin"])

with tab_men:
    render_tournament_tab(
        db_path=DB_MEN,
        fide_ids=FIDE_MEN,
        tournament=TOURNAMENT_MEN,
        model_path=MODEL_MEN,
        config_path=CONFIG_MEN,
    )

with tab_women:
    render_tournament_tab(
        db_path=DB_WOMEN,
        fide_ids=FIDE_WOMEN,
        tournament=TOURNAMENT_WOMEN,
        model_path=MODEL_WOMEN,
        config_path=CONFIG_WOMEN,
    )
