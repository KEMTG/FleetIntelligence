"""
=================================================================================
 FLEET INTELLIGENCE -- STREAMLIT WEB APP
=================================================================================
A file-upload front end for fleet_intelligence_pipeline.py. Anyone with a
correctly formatted CSV/XLSX can drop it in a browser, click Run, and get the
full Module 1-4 analysis (anonymization, cleaning, archetypes, dormancy-risk
model) back as charts, tables, and downloadable files -- no code required on
their end.

RUN LOCALLY
-----------
    pip install streamlit pandas numpy scikit-learn matplotlib openpyxl
    streamlit run streamlit_app.py

DEPLOY (see the deployment guide sent alongside this file for full steps)
---------------------------------------------------------------------------
This file + fleet_intelligence_pipeline.py + requirements.txt must sit in the
same folder / same GitHub repo. Point Streamlit Community Cloud (or Hugging
Face Spaces) at that repo and this file as the entry point.

IMPORTANT: this file imports the pipeline as a MODULE (no CLI parsing), so it
must be run with `streamlit run`, not `python streamlit_app.py`.
=================================================================================
"""

import io
import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

import fleet_intelligence_pipeline as fip  # the pipeline script from earlier, unmodified

st.set_page_config(page_title="Fleet Intelligence & Analytics", layout="wide")

st.title("Electric Vehicle Fleet Intelligence Platform")
st.caption(
    "Upload your fleet's raw 5-minute telematics stream (CSV or XLSX) and get "
    "cleaned data, rider archetypes, and a trained dormancy/breakdown-risk model -- "
    "all in your browser, nothing installed."
)

with st.expander("📋 Required file format", expanded=False):
    st.markdown(
        "Your file needs these columns (any order, flexible naming -- e.g. "
        "`name`, `serial`, or `bike_id` all work for the ID column):\n\n"
        "| date | time | latitude | longitude | bike_id |\n"
        "|---|---|---|---|---|\n"
        "| 2026/09/25 | 22:05 | -1.388147 | 37.678901 | ANONBIKE-40153DD97173EF52 |\n\n"
        "One row per GPS ping. Works best with several weeks of history across "
        "multiple vehicles -- the archetype and risk models need a real fleet "
        "to train on, not just one bike."
    )

uploaded = st.file_uploader("Upload telematics file", type=["csv", "xlsx", "xls"])

if uploaded is not None:
    # ---- Load ----
    ext = Path(uploaded.name).suffix.lower()
    try:
        if ext in (".xlsx", ".xls"):
            raw = pd.read_excel(uploaded)
        else:
            raw = pd.read_csv(uploaded)
        raw = fip.normalize_columns(raw)
    except Exception as e:
        st.error(f"Couldn't read this file: {e}")
        st.stop()

    st.success(f"Loaded {len(raw):,} rows.")

    bike_options = ["(none -- fleet-wide only)"] + sorted(raw["bike_id"].astype(str).unique().tolist())
    demo_bike_choice = st.selectbox(
        "Optional: pick one bike for a detailed demo chart",
        bike_options,
    )
    demo_bike = None if demo_bike_choice.startswith("(none") else demo_bike_choice

    anonymize = st.checkbox("Anonymize bike IDs in the output (HMAC-SHA256)", value=True)

    run = st.button("▶ Run analysis", type="primary")

    if run:
        outdir = Path(tempfile.mkdtemp())
        progress = st.progress(0, text="Cleaning and engineering features...")

        # ---- Module 2 ----
        df = fip.clean_and_engineer(raw)
        df = fip.cluster_stops(df)
        stop_summary = fip.summarize_stop_clusters(df)
        progress.progress(25, text="Building behavioral features and rider archetypes...")

        # ---- Module 3 ----
        daily = fip.build_daily_features(df)
        archetypes, archetype_info = fip.build_rider_archetypes(df, daily)
        progress.progress(55, text="Training the dormancy/breakdown-risk model...")

        # ---- Module 4 ----
        full_daily_reindexed, valid = fip.build_dormancy_training_table(daily)
        clf, model_metrics = fip.train_and_validate_risk_model(valid)
        risk_scores = fip.score_latest_day(clf, full_daily_reindexed) if clf is not None else pd.DataFrame()

        dropout = fip.flag_sustained_dropout(full_daily_reindexed)
        archetypes = archetypes.merge(dropout, on="bike_id", how="left")
        dropout_mask = archetypes["sustained_dropout_flag"] == True
        archetypes.loc[dropout_mask, "archetype_label"] = "Dormant / Dropped-Out"
        progress.progress(75, text="Anonymizing and rendering results...")

        # ---- Module 1 ----
        if anonymize:
            secret = fip.get_hmac_secret()
            df_out = df.copy()
            df_out["bike_id"] = fip.anonymize_series(df_out["bike_id"], secret)
        else:
            df_out = df

        # ---- Plots ----
        fleet_png = outdir / "fleet_overview.png"
        fip.plot_fleet_overview(archetypes, risk_scores, fleet_png)

        demo_png = None
        if demo_bike:
            demo_png = outdir / f"{demo_bike}_demo.png"
            fip.plot_single_bike(df, daily, stop_summary, demo_bike, demo_png)

        progress.progress(100, text="Done.")
        progress.empty()

        # ================= RESULTS =================
        st.header("Results")

        c1, c2, c3 = st.columns(3)
        c1.metric("Bikes analyzed", df["bike_id"].nunique())
        c2.metric("Pings processed", f"{len(df):,}")
        if model_metrics:
            c3.metric("Risk model ROC-AUC", f"{model_metrics['roc_auc']:.3f}")
        else:
            c3.metric("Risk model", "Not trained (fleet too small)")

        st.subheader("Fleet overview")
        st.image(str(fleet_png), use_container_width=True)

        if demo_png:
            st.subheader(f"Single-bike demo: {demo_bike}")
            st.image(str(demo_png), use_container_width=True)

        st.subheader("Rider archetypes")
        st.dataframe(
            archetypes[["bike_id", "archetype_label", "idle_ratio", "daily_range_mean",
                        "current_dormant_streak_days"]].sort_values("archetype_label"),
            use_container_width=True,
        )

        if len(risk_scores) > 0:
            st.subheader("Highest dormancy/breakdown risk (top 15)")
            st.dataframe(risk_scores.head(15), use_container_width=True)

        if model_metrics:
            with st.expander("Model validation details"):
                st.json(model_metrics)

        # ================= DOWNLOADS =================
        st.subheader("Downloads")
        d1, d2, d3, d4 = st.columns(4)
        d1.download_button("Anonymized pings (CSV)",
                            df_out.to_csv(index=False).encode(), "anonymized_pings.csv")
        d2.download_button("Daily bike features (CSV)",
                            daily.to_csv(index=False).encode(), "daily_bike_features.csv")
        d3.download_button("Bike archetypes (CSV)",
                            archetypes.to_csv(index=False).encode(), "bike_archetypes.csv")
        if len(risk_scores) > 0:
            d4.download_button("Dormancy risk scores (CSV)",
                                risk_scores.to_csv(index=False).encode(), "dormancy_risk_scores.csv")

        if model_metrics:
            st.download_button("Model metrics (JSON)",
                                json.dumps(model_metrics, indent=2).encode(), "model_metrics.json")
else:
    st.info("Upload a file above to get started.")
