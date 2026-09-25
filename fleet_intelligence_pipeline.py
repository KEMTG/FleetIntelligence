"""
=================================================================================
 FLEET INTELLIGENCE & ANALYTICS PLATFORM -- END-TO-END PIPELINE
=================================================================================
Single-file pipeline for e-mobility / battery-swap motorcycle telematics data.

USAGE
-----
    # Full fleet run (all bikes) -- trains fleet-wide models
    python fleet_intelligence_pipeline.py --input data.csv

    # Full fleet run + a detailed single-bike demo report/plot
    python fleet_intelligence_pipeline.py --input data.csv --bike LGMMWSAB2P0A00185

    # Excel input works the same way
    python fleet_intelligence_pipeline.py --input fleet_data.xlsx

    # Custom output folder
    python fleet_intelligence_pipeline.py --input data.csv --outdir results/

INPUT FORMAT
------------
CSV or XLSX with (in any column order, case-insensitive, flexible naming):
    date, time, latitude, longitude, <bike id column: "name"/"serial"/"bike_id"/"id">

WHAT IT DOES (maps to the platform's 5 modules)
-------------------------------------------------
  MODULE 1 - Anonymization:      HMAC-SHA256 pseudonymization of bike IDs.
  MODULE 2 - Feature Engineering: Haversine speed/distance, anomaly flags
                                   (GPS drift, offline gaps, speed spikes),
                                   per-bike DBSCAN stop-point clustering.
  MODULE 3 - BI & Behavior:      Daily/rush-hour metrics, rider archetype
                                   clustering (Gaussian Mixture Model,
                                   model-selected via BIC, validated via
                                   silhouette score).
  MODULE 4 - Commercial ML:      >>> Flagship trained & validated model <<<
                                   Predictive Dormancy / Breakdown-Risk
                                   Early-Warning model (RandomForest),
                                   trained on self-supervised labels derived
                                   from the trajectory data itself, validated
                                   with a bike-grouped train/test split
                                   (prevents leakage) and standard classification
                                   metrics (accuracy, precision, recall, F1, ROC-AUC).
  MODULE 5 - Architecture:       Not code -- see the accompanying write-up.
                                   (Model registry / scoring pattern is noted
                                   in the report this script generates.)

OUTPUTS (written to --outdir, default ./fleet_intelligence_output/)
---------------------------------------------------------------------
  anonymized_pings.csv          Cleaned, anonymized ping-level data
  daily_bike_features.csv       Per-bike-day behavioral feature table
  bike_archetypes.csv           Per-bike archetype cluster assignment
  stop_clusters.csv             Per-bike spatial stop-point clusters
  dormancy_risk_scores.csv      Latest-day risk score per bike (from the model)
  model_metrics.json            Validation metrics for the trained model
  fleet_summary_report.md       Human-readable findings report
  [bike]_demo.png               4-panel chart, only if --bike is given
  fleet_overview.png            Fleet-wide charts (archetypes, risk distribution)

SECURITY NOTE ON ANONYMIZATION
-------------------------------
The HMAC secret is generated fresh per run unless FLEET_HMAC_SECRET is set
in the environment. For reproducible anonymization across runs (e.g. so a
client's re-uploaded data maps to the same pseudonyms), export the same
secret each time:
    export FLEET_HMAC_SECRET="<a long random string, kept in your vault>"
Never commit this secret or ship it alongside the anonymized data.
=================================================================================
"""

import argparse
import hashlib
import hmac
import json
import os
import secrets
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Matplotlib is optional at import time so the pipeline still runs headless-only
# environments without a display backend.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)


# =================================================================================
# MODULE 1: ANONYMIZATION
# =================================================================================

def get_hmac_secret() -> str:
    """
    Load the HMAC secret from the environment, or generate a fresh one for
    this run. In production, set FLEET_HMAC_SECRET from your secret vault
    (see module docstring) rather than relying on the auto-generated value.
    """
    secret = os.environ.get("FLEET_HMAC_SECRET")
    if secret:
        print("[Module 1] Using HMAC secret from FLEET_HMAC_SECRET environment variable.")
        return secret
    secret = secrets.token_hex(32)
    print("[Module 1] No FLEET_HMAC_SECRET set -- generated a one-off secret for this run.")
    print("[Module 1] WARNING: anonymized IDs will NOT be reproducible across runs.")
    return secret


def anonymize_series(id_series: pd.Series, secret: str, length: int = 16) -> pd.Series:
    """Deterministic, non-reversible HMAC-SHA256 pseudonymization of an ID column."""
    def _h(x):
        digest = hmac.new(secret.encode("utf-8"), str(x).encode("utf-8"), hashlib.sha256).hexdigest()
        return f"ANON-{digest[:length].upper()}"
    return id_series.apply(_h)


# =================================================================================
# LOADING & COLUMN NORMALIZATION
# =================================================================================

def load_raw(path: str) -> pd.DataFrame:
    ext = Path(path).suffix.lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type '{ext}'. Use .csv, .xlsx, or .xls.")
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map flexible/messy column names onto the canonical schema."""
    cols = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=cols)

    def find(*candidates):
        for c in candidates:
            if c in df.columns:
                return c
        for col in df.columns:
            for c in candidates:
                if c in col:
                    return col
        return None

    date_col = find("date")
    time_col = find("time")
    lat_col = find("latitude", "lat")
    lon_col = find("longitude", "lon", "lng")
    id_col = find("bike_id", "serial", "name", "id", "vehicle")

    missing = [n for n, v in [("date", date_col), ("time", time_col), ("latitude", lat_col),
                               ("longitude", lon_col), ("bike id", id_col)] if v is None]
    if missing:
        raise ValueError(f"Could not find required column(s): {missing}. "
                          f"Available columns: {list(df.columns)}")

    df = df.rename(columns={
        date_col: "date", time_col: "time", lat_col: "latitude",
        lon_col: "longitude", id_col: "bike_id",
    })
    return df[["date", "time", "latitude", "longitude", "bike_id"]]


# =================================================================================
# MODULE 2: CLEANING, FEATURE ENGINEERING, ANOMALY FLAGS, STOP CLUSTERING
# =================================================================================

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    print("[Module 2] Cleaning and engineering features...")
    df = df.copy()
    df = df[df["bike_id"].astype(str) != "-1"]
    df["dt"] = pd.to_datetime(
        df["date"].astype(str).str.strip() + " " + df["time"].astype(str).str.strip(),
        errors="coerce", dayfirst=False,
    )
    # Fallback parse for date formats like DD/MM/YYYY if the first pass mostly failed
    if df["dt"].isna().mean() > 0.5:
        df["dt"] = pd.to_datetime(
            df["date"].astype(str).str.strip() + " " + df["time"].astype(str).str.strip(),
            errors="coerce", dayfirst=True,
        )
    before = len(df)
    df = df.dropna(subset=["dt", "latitude", "longitude"])
    df = df.sort_values(["bike_id", "dt"]).drop_duplicates(subset=["bike_id", "dt"]).reset_index(drop=True)
    print(f"    rows: {before} -> {len(df)} after dropping invalid/duplicate rows "
          f"({df['bike_id'].nunique()} unique bikes)")

    g = df.groupby("bike_id")
    df["prev_lat"] = g["latitude"].shift()
    df["prev_lon"] = g["longitude"].shift()
    df["prev_dt"] = g["dt"].shift()
    df["dist_m"] = haversine(df["prev_lat"], df["prev_lon"], df["latitude"], df["longitude"])
    df["dt_sec"] = (df["dt"] - df["prev_dt"]).dt.total_seconds()
    df["speed_kmh"] = np.where(df["dt_sec"] > 0, (df["dist_m"] / df["dt_sec"]) * 3.6, np.nan)

    # --- Anomaly / noise flags ---
    df["gap_flag"] = df["dt_sec"] > 1800          # offline / transmission gap (>30 min)
    df["spike_flag"] = df["speed_kmh"] > 120       # physically implausible for this vehicle class
    df["static_flag"] = df["dist_m"] < 15          # GPS jitter / idle (sub-15m "movement")

    # --- Cleaned speed: interpolate over spikes, zero out static noise ---
    df["speed_clean"] = df["speed_kmh"].where(~df["spike_flag"])
    df["speed_clean"] = g["speed_clean"].transform(lambda x: x.interpolate(limit=3)) \
        if False else df.groupby("bike_id")["speed_clean"].transform(lambda x: x.interpolate(limit=3))
    df.loc[df["static_flag"], "speed_clean"] = 0

    n_gaps, n_spikes, n_static = df["gap_flag"].sum(), df["spike_flag"].sum(), df["static_flag"].sum()
    print(f"    anomalies flagged -> offline gaps: {n_gaps}, speed spikes: {n_spikes}, "
          f"static/idle pings: {n_static} ({n_static/len(df):.1%})")
    return df


def cluster_stops(df: pd.DataFrame, eps_m: float = 30, min_samples: int = 8) -> pd.DataFrame:
    """Per-bike DBSCAN over all pings to find stop points (home base, hubs, swap stations)."""
    print("[Module 2] Clustering stop points per bike (DBSCAN, haversine metric)...")

    def _cluster(g):
        if len(g) < min_samples:
            return pd.Series(-1, index=g.index, name="stop_cluster")
        coords = np.radians(g[["latitude", "longitude"]].values)
        db = DBSCAN(eps=eps_m / 6371000, min_samples=min_samples, metric="haversine").fit(coords)
        return pd.Series(db.labels_, index=g.index, name="stop_cluster")

    df = df.copy()
    # Build the label Series per-group and concat explicitly -- more robust across
    # pandas versions than relying on groupby.apply's automatic Series/DataFrame
    # inference, which behaves inconsistently when there is only a single group.
    parts = [_cluster(g) for _, g in df.groupby("bike_id")]
    labels = pd.concat(parts).sort_index()
    df["stop_cluster"] = labels
    return df


def summarize_stop_clusters(df: pd.DataFrame) -> pd.DataFrame:
    clustered = df[df["stop_cluster"] != -1]
    summary = clustered.groupby(["bike_id", "stop_cluster"]).agg(
        n_points=("stop_cluster", "count"),
        lat=("latitude", "mean"),
        lon=("longitude", "mean"),
        first_seen=("dt", "min"),
        last_seen=("dt", "max"),
    ).reset_index()
    # Heuristic label: dominant stay per bike (most visited) -> likely home base
    summary["rank_within_bike"] = summary.groupby("bike_id")["n_points"].rank(ascending=False, method="first")
    summary["likely_role"] = np.where(
        summary["rank_within_bike"] == 1, "home_base_or_depot", "secondary_stop"
    )
    return summary


# =================================================================================
# MODULE 3: BI & RIDER ARCHETYPE CLUSTERING
# =================================================================================

def _shannon_entropy(series: pd.Series) -> float:
    p = series.value_counts(normalize=True)
    return float(-(p * np.log2(p)).sum()) if len(p) > 0 else 0.0


def build_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    print("[Module 3] Building daily per-bike behavioral features...")
    df = df.copy()
    df["date_only"] = df["dt"].dt.date
    daily = df.groupby(["bike_id", "date_only"]).agg(
        first_ping=("dt", "min"), last_ping=("dt", "max"),
        total_dist_km=("dist_m", lambda x: np.nansum(x) / 1000),
        moving_pings=("static_flag", lambda x: (~x).sum()),
        total_pings=("dt", "count"),
        max_speed=("speed_clean", "max"),
    ).reset_index()
    daily["idle_ratio"] = 1 - daily["moving_pings"] / daily["total_pings"]
    daily["operating_hours"] = (daily["last_ping"] - daily["first_ping"]).dt.total_seconds() / 3600
    return daily


def build_rider_archetypes(df: pd.DataFrame, daily: pd.DataFrame, min_pings: int = 30):
    """
    Module 3 model: Gaussian Mixture Model clustering of riders into behavioral
    archetypes (e.g. high-utilization ride-hail vs. fixed-route commuter vs.
    low-utilization/dormant). Model order (k) is selected via BIC; cluster
    quality is validated via silhouette score.
    """
    print("[Module 3] Training rider archetype model (Gaussian Mixture, BIC-selected k)...")
    df = df.copy()
    df["hour"] = df["dt"].dt.hour
    df["dow"] = df["dt"].dt.dayofweek
    df["is_weekend"] = df["dow"] >= 5
    df["cell"] = df["latitude"].round(3).astype(str) + "_" + df["longitude"].round(3).astype(str)

    per_bike_daily = daily.groupby("bike_id")["total_dist_km"].agg(["mean", "std"]) \
        .rename(columns={"mean": "daily_range_mean", "std": "daily_range_std"})

    rush = df[df["hour"].isin([7, 8, 17, 18])]
    offpeak = df[~df["hour"].isin([7, 8, 17, 18])]
    rush_speed = rush.groupby("bike_id")["speed_clean"].mean().rename("rush_speed")
    offpeak_speed = offpeak.groupby("bike_id")["speed_clean"].mean().rename("offpeak_speed")

    feat = df.groupby("bike_id").agg(
        idle_ratio=("static_flag", "mean"),
        weekend_ratio=("is_weekend", "mean"),
        route_entropy=("cell", _shannon_entropy),
        n_pings=("dt", "count"),
    ).join(per_bike_daily).join(rush_speed).join(offpeak_speed).fillna(0)

    eligible = feat[feat["n_pings"] >= min_pings].copy()
    excluded = len(feat) - len(eligible)
    if excluded:
        print(f"    excluded {excluded} bikes with <{min_pings} pings (insufficient data for archetyping)")

    feature_cols = ["idle_ratio", "weekend_ratio", "route_entropy",
                     "daily_range_mean", "daily_range_std", "rush_speed", "offpeak_speed"]

    if len(eligible) < 4:
        print(f"    WARNING: only {len(eligible)} bike(s) with enough data -- archetype clustering needs "
              f"a fleet (several bikes) to be meaningful. Skipping Module 3 model for this run; "
              f"single-bike behavioral metrics are still available in daily_bike_features.csv.")
        eligible["archetype"] = -1
        eligible["archetype_label"] = "N/A (fleet too small to cluster)"
        return eligible.reset_index(), {}

    X = eligible[feature_cols]
    Xs = StandardScaler().fit_transform(X)

    best_k, best_bic, best_model = None, np.inf, None
    bic_scores = {}
    max_k = max(2, min(7, len(eligible) // 2))  # never propose more components than the data can support
    for k in range(2, max_k + 1):
        gmm = GaussianMixture(n_components=k, random_state=42, n_init=3).fit(Xs)
        bic = gmm.bic(Xs)
        bic_scores[k] = bic
        if bic < best_bic:
            best_bic, best_k, best_model = bic, k, gmm

    labels = best_model.predict(Xs)
    sil = silhouette_score(Xs, labels) if len(set(labels)) > 1 else float("nan")
    eligible["archetype"] = labels
    print(f"    model selection: k={best_k} chosen by BIC (candidates: "
          f"{ {k: round(v,1) for k, v in bic_scores.items()} })")
    print(f"    validation: silhouette score = {sil:.3f} "
          f"(higher = better separated clusters; unsupervised, no ground truth available)")

    # Heuristic, human-readable naming of each cluster from its centroid profile.
    # Uses RELATIVE ranking across the discovered clusters (rather than fixed absolute
    # thresholds) so labels stay differentiated regardless of how many clusters k finds.
    profile = eligible.groupby("archetype")[feature_cols].mean()
    ranked_by_range = profile["daily_range_mean"].sort_values()
    names = {}
    # Lowest daily range + highest idle ratio among the low-range end -> stage-based riders:
    # bikes that park at a known spot and wait for walk-up/offline customers between short trips,
    # rather than continuously roaming. High idle time here reflects waiting-for-a-fare, not
    # necessarily a broken-down or abandoned bike -- see the sustained-dormancy flag below for that.
    low_util_cluster = ranked_by_range.index[0]
    names[low_util_cluster] = "Stage-Based / Walk-up Riders"
    remaining = [c for c in profile.index if c != low_util_cluster]
    if remaining:
        # Among the rest, highest route entropy + weekend usage -> on-demand / ride-hailing style
        entropy_rank = profile.loc[remaining, "route_entropy"].sort_values(ascending=False)
        on_demand_cluster = entropy_rank.index[0]
        names[on_demand_cluster] = "Ride-Hailing / On-Demand"
        for c in remaining:
            if c != on_demand_cluster:
                names[c] = "Fixed-Route Commuter"
    eligible["archetype_label"] = eligible["archetype"].map(names)

    return eligible.reset_index(), {"k": best_k, "bic_scores": bic_scores, "silhouette": sil, "cluster_profile": profile}


# =================================================================================
# MODULE 4: FLAGSHIP MODEL -- PREDICTIVE DORMANCY / BREAKDOWN-RISK
# =================================================================================

def build_dormancy_training_table(daily: pd.DataFrame):
    """
    Builds a daily bike-level feature/label table for the dormancy-risk model.
    Label: will this bike be effectively dormant (near-zero movement, near-100%
    idle) for the NEXT 3 consecutive days? This is a self-supervised label
    derived from the trajectory data -- no external ground truth is needed,
    which is what makes this trainable directly on any client's raw stream.
    """
    print("[Module 4] Building self-supervised training labels for the dormancy-risk model...")
    daily = daily.copy()
    daily["date_only"] = pd.to_datetime(daily["date_only"])

    full_range = pd.date_range(daily["date_only"].min(), daily["date_only"].max(), freq="D")
    bikes = daily["bike_id"].unique()
    idx = pd.MultiIndex.from_product([bikes, full_range], names=["bike_id", "date_only"])
    d = daily.set_index(["bike_id", "date_only"]).reindex(idx).reset_index()
    # A day with zero pings is treated as fully idle/offline (conservative for risk purposes)
    d["total_dist_km"] = d["total_dist_km"].fillna(0)
    d["idle_ratio"] = d["idle_ratio"].fillna(1.0)
    d["total_pings"] = d["total_pings"].fillna(0)
    d = d.sort_values(["bike_id", "date_only"]).reset_index(drop=True)

    d["is_dormant_day"] = ((d["total_dist_km"] < 0.5) & (d["idle_ratio"] > 0.95)).astype(int)

    g = d.groupby("bike_id")
    d["roll3_dist"] = g["total_dist_km"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    d["roll3_idle"] = g["idle_ratio"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    d["dist_trend"] = g["total_dist_km"].transform(lambda x: x.diff())
    d["pings_trend"] = g["total_pings"].transform(lambda x: x.diff())
    d["days_active_so_far"] = g["is_dormant_day"].transform(lambda x: (1 - x).expanding().sum())
    d["day_num"] = g.cumcount()

    fwd1 = g["is_dormant_day"].shift(-1)
    fwd2 = g["is_dormant_day"].shift(-2)
    fwd3 = g["is_dormant_day"].shift(-3)
    d["label_dormant_next3"] = ((fwd1 == 1) & (fwd2 == 1) & (fwd3 == 1)).astype(int)

    max_date = d["date_only"].max()
    cutoff = max_date - pd.Timedelta(days=3)
    # Need >=2 days lookback for rolling/trend features and >=3 days lookahead for the label
    valid = d[(d["day_num"] >= 2) & (d["date_only"] <= cutoff)].copy()
    return d, valid


FEATURE_COLS = ["total_dist_km", "idle_ratio", "total_pings", "roll3_dist", "roll3_idle",
                 "dist_trend", "pings_trend", "days_active_so_far", "day_num"]


def train_and_validate_risk_model(valid: pd.DataFrame, min_rows: int = 200):
    """
    Trains a RandomForest classifier to predict near-term dormancy/breakdown
    risk, and validates it with a GROUPED train/test split (grouped by bike_id,
    so no bike appears in both train and test -- this prevents leakage that
    would otherwise inflate the reported accuracy).
    """
    print("[Module 4] Training the dormancy/breakdown-risk model (RandomForestClassifier)...")
    if len(valid) < min_rows:
        print(f"    WARNING: only {len(valid)} training rows available -- results will be unstable. "
              f"This model needs a reasonably sized fleet and multi-week history to validate well.")

    X = valid[FEATURE_COLS].fillna(0)
    y = valid["label_dormant_next3"]
    groups = valid["bike_id"]

    if y.nunique() < 2:
        print("    WARNING: only one class present in labels -- cannot train/validate a classifier. "
              "This typically means the dataset is too short or too uniform (e.g. one bike, few days).")
        return None, None

    if groups.nunique() < 4:
        print(f"    WARNING: only {groups.nunique()} bike(s) in this input -- a grouped train/test split "
              f"needs multiple bikes to hold out fairly. Skipping Module 4 model training/validation for "
              f"this run (this model is designed to be trained fleet-wide, then applied to score individual "
              f"bikes -- it isn't meant to be trained on a single vehicle's history).")
        return None, None

    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=10,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "train_bikes": int(groups.iloc[train_idx].nunique()),
        "test_bikes": int(groups.iloc[test_idx].nunique()),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)) if y_test.nunique() > 1 else None,
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "feature_importances": dict(sorted(
            zip(FEATURE_COLS, clf.feature_importances_.tolist()),
            key=lambda kv: -kv[1]
        )),
        "positive_rate_overall": float(y.mean()),
    }
    print(f"    validation (held-out bikes, n={metrics['test_bikes']}): "
          f"accuracy={metrics['accuracy']:.3f}  precision={metrics['precision']:.3f}  "
          f"recall={metrics['recall']:.3f}  f1={metrics['f1']:.3f}  "
          f"roc_auc={metrics['roc_auc'] if metrics['roc_auc'] is None else round(metrics['roc_auc'],3)}")
    return clf, metrics


def score_latest_day(clf, full_daily_reindexed: pd.DataFrame) -> pd.DataFrame:
    """Apply the trained model to each bike's most recent day to get a live risk score."""
    if clf is None:
        return pd.DataFrame()
    latest = full_daily_reindexed.sort_values("date_only").groupby("bike_id").tail(1).copy()
    X = latest[FEATURE_COLS].fillna(0)
    latest["dormancy_risk_score"] = clf.predict_proba(X)[:, 1]
    return latest[["bike_id", "date_only", "total_dist_km", "idle_ratio", "dormancy_risk_score"]] \
        .sort_values("dormancy_risk_score", ascending=False)


def flag_sustained_dropout(full_daily_reindexed: pd.DataFrame, min_consecutive_dormant: int = 5) -> pd.DataFrame:
    """
    Distinguishes normal stage-riding idle time (waiting between fares, but still
    getting trips most days) from a bike that has gone genuinely dark: a long,
    unbroken run of fully-dormant days with NO trips at all, running right up to
    the end of the tracked period. This is the sharper, second-layer signal that
    separates "just a stage rider" from "possibly broken down / abandoned / lease-
    defaulted" within the low-daily-range archetype.
    """
    d = full_daily_reindexed.sort_values(["bike_id", "date_only"]).copy()
    last_date = d["date_only"].max()

    def _current_streak(g):
        # counts consecutive dormant days ending at the bike's most recent tracked day
        vals = g["is_dormant_day"].values
        streak = 0
        for v in vals[::-1]:
            if v == 1:
                streak += 1
            else:
                break
        return streak

    streaks = d.groupby("bike_id").apply(_current_streak).rename("current_dormant_streak_days")
    out = streaks.reset_index()
    out["sustained_dropout_flag"] = out["current_dormant_streak_days"] >= min_consecutive_dormant
    return out


# =================================================================================
# PLOTTING
# =================================================================================

def plot_single_bike(df: pd.DataFrame, daily: pd.DataFrame, stop_summary: pd.DataFrame,
                      bike_id: str, outpath: Path):
    bdf = df[df["bike_id"] == bike_id]
    bdaily = daily[daily["bike_id"] == bike_id].copy()
    bdaily["date_only"] = pd.to_datetime(bdaily["date_only"])
    bstops = stop_summary[stop_summary["bike_id"] == bike_id]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    ax = axes[0, 0]
    ax.bar(bdaily["date_only"], bdaily["total_dist_km"], color="#2563eb")
    ax.set_title(f"{bike_id} -- Daily Distance (km)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    ax = axes[0, 1]
    ax.plot(bdaily["date_only"], bdaily["idle_ratio"] * 100, marker="o", color="#dc2626")
    ax.set_title("Daily Idle Ratio (%)")
    ax.set_ylim(0, 105)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    ax = axes[1, 0]
    moving = bdf[~bdf["static_flag"]]["speed_clean"].dropna()
    if len(moving) > 0:
        ax.hist(moving, bins=25, color="#16a34a", edgecolor="white")
    ax.set_title("Speed Distribution (moving pings)")
    ax.set_xlabel("km/h")

    ax = axes[1, 1]
    noise = bdf[bdf["stop_cluster"] == -1]
    ax.scatter(noise["longitude"], noise["latitude"], s=4, color="lightgray", label="transit/noise")
    cmap = plt.cm.tab10
    for i, row in bstops.reset_index().iterrows():
        pts = bdf[bdf["stop_cluster"] == row["stop_cluster"]]
        ax.scatter(pts["longitude"], pts["latitude"], s=8, color=cmap(i % 10),
                   label=f"{row['likely_role']} (n={row['n_points']})")
    ax.set_title("Spatial Stop-Point Clusters")
    ax.legend(fontsize=7, loc="best")

    plt.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


def plot_fleet_overview(archetypes: pd.DataFrame, risk_scores: pd.DataFrame, outpath: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    if (archetypes["archetype"] != -1).any():
        counts = archetypes["archetype_label"].value_counts()
        total = counts.sum()
        bars = ax.bar(counts.index, counts.values, color="#2563eb")
        for bar, cnt in zip(bars, counts.values):
            pct = cnt / total * 100
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.01,
                    f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, counts.max() * 1.12)  # headroom so the labels don't get clipped
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    else:
        ax.text(0.5, 0.5, "Fleet too small to cluster\n(need multiple bikes)",
                ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Fleet Rider Archetypes")
    ax.set_ylabel("number of bikes")

    ax = axes[1]
    if len(risk_scores) > 0:
        ax.hist(risk_scores["dormancy_risk_score"], bins=25, color="#dc2626", edgecolor="white")
    ax.set_title("Fleet-Wide Dormancy Risk Score Distribution\n(latest day per bike)")
    ax.set_xlabel("predicted probability of dormancy in next 3 days")

    plt.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


# =================================================================================
# REPORT
# =================================================================================

def write_report(outdir: Path, n_bikes: int, n_rows: int, archetype_info: dict,
                  archetypes: pd.DataFrame, model_metrics: dict, risk_scores: pd.DataFrame,
                  demo_bike: str = None):
    lines = []
    lines.append("# Fleet Intelligence & Analytics -- Run Report\n")
    lines.append(f"**Fleet size:** {n_bikes} bikes | **Pings processed:** {n_rows:,}\n")

    lines.append("## Module 3: Rider Archetypes\n")
    if archetype_info and "k" in archetype_info:
        lines.append(f"- Model: Gaussian Mixture Model, k={archetype_info['k']} "
                      f"components chosen by BIC, plus a 4th rule-based archetype "
                      f"('Dormant / Dropped-Out') split out post-hoc from sustained-inactivity bikes.\n"
                      f"- Validation: silhouette score = {archetype_info['silhouette']:.3f} "
                      f"(unsupervised cluster separation quality, on the original 3 GMM clusters).\n")
        for label, cnt in archetypes["archetype_label"].value_counts().items():
            lines.append(f"  - **{label}**: {cnt} bikes\n")

        if "sustained_dropout_flag" in archetypes.columns and (archetypes["archetype_label"] == "Dormant / Dropped-Out").any():
            lines.append(
                f"\n  **Note on Dormant / Dropped-Out:** these bikes were originally clustered as "
                f"'Stage-Based / Walk-up Riders' (short trips, high idle time waiting for walk-up fares -- "
                f"a normal, high-volume riding style) but were pulled into their own 4th archetype because "
                f"they show a sustained run of 5+ fully-dormant days (zero trips at all) running up to the "
                f"end of the tracked period. That's a materially different operational state -- worth "
                f"investigating as a possible breakdown, abandonment, or lease drop-off -- rather than just "
                f"a stage rider having a slow stretch. See `current_dormant_streak_days` in "
                f"bike_archetypes.csv for how long each has been inactive.\n"
            )

    lines.append("\n## Module 4: Predictive Dormancy / Breakdown-Risk Model (flagship)\n")
    if model_metrics:
        lines.append(
            f"- Model: RandomForestClassifier, trained on self-supervised labels "
            f"(will this bike go dormant for 3+ consecutive days?).\n"
            f"- Validation: grouped train/test split by bike_id "
            f"({model_metrics['train_bikes']} train bikes / {model_metrics['test_bikes']} test bikes, "
            f"no bike in both sets).\n"
            f"- **Accuracy:** {model_metrics['accuracy']:.3f}\n"
            f"- **Precision:** {model_metrics['precision']:.3f}\n"
            f"- **Recall:** {model_metrics['recall']:.3f}\n"
            f"- **F1:** {model_metrics['f1']:.3f}\n"
            f"- **ROC-AUC:** {model_metrics['roc_auc']}\n"
            f"- Baseline positive rate (naive 'always predict dormant'): "
            f"{model_metrics['positive_rate_overall']:.1%}\n"
            f"- Top predictive features: "
            + ", ".join(list(model_metrics['feature_importances'].keys())[:3]) + "\n"
        )
        if len(risk_scores) > 0:
            top = risk_scores.head(10)
            lines.append("\n**Top 10 highest-risk bikes (most recent day scored):**\n\n")
            lines.append("| bike_id | last date | risk score |\n|---|---|---|\n")
            for _, r in top.iterrows():
                lines.append(f"| {r['bike_id']} | {r['date_only'].date()} | {r['dormancy_risk_score']:.2f} |\n")
    else:
        lines.append("- Not enough data / label diversity to train and validate this model on this input "
                      "(needs a multi-bike fleet with several weeks of history).\n")

    lines.append("\n## Notes & Caveats\n")
    lines.append(
        "- The dormancy label is *self-supervised* (derived from movement data itself, not confirmed "
        "breakdown/lease-default records). It is a strong **proxy** for breakdown/churn risk, not a "
        "ground-truth diagnosis -- validate against real maintenance/lease logs before using for automated "
        "actions.\n"
        "- Module 4's other commercial models (swap-demand forecasting, fraud/misuse scoring, route/energy "
        "optimization) need additional data this ping-only stream doesn't contain (confirmed swap events, "
        "road network graph) -- see the accompanying strategic write-up for how to extend this pipeline "
        "once that data is available.\n"
        "- Module 5 (B2B architecture / pitch) is a design document, not code -- see the separate write-up.\n"
    )

    if demo_bike:
        lines.append(f"\n## Single-Bike Demo: `{demo_bike}`\nSee `{demo_bike}_demo.png` for the detailed chart.\n")

    (outdir / "fleet_summary_report.md").write_text("".join(lines))
    print(f"[Report] Written to {outdir / 'fleet_summary_report.md'}")


# =================================================================================
# MAIN
# =================================================================================

def main():
    parser = argparse.ArgumentParser(description="Fleet Intelligence & Analytics Pipeline")
    parser.add_argument("--input", default=r"C:\Users\T470s\Downloads\data.csv",
                         help="Path to CSV or XLSX telematics file")
    parser.add_argument("--bike", default=None,
                         help="Specific bike_id for a detailed single-bike demo report/plot "
                              "(the fleet-wide models still train on ALL bikes). Omit to skip.")
    parser.add_argument("--outdir", default="fleet_intelligence_output", help="Output folder")
    parser.add_argument("--no-anonymize", action="store_true",
                         help="Skip Module 1 anonymization (keep raw bike IDs in outputs). "
                              "Not recommended for anything leaving your internal environment.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading '{args.input}'...")
    raw = load_raw(args.input)
    raw = normalize_columns(raw)
    print(f"Loaded {len(raw):,} raw rows.")

    # ---- Module 2 ----
    df = clean_and_engineer(raw)
    df = cluster_stops(df)
    stop_summary = summarize_stop_clusters(df)
    stop_summary.to_csv(outdir / "stop_clusters.csv", index=False)

    # ---- Module 3 ----
    daily = build_daily_features(df)
    archetypes, archetype_info = build_rider_archetypes(df, daily)

    # ---- Module 4 (flagship trained + validated model) ----
    full_daily_reindexed, valid = build_dormancy_training_table(daily)
    clf, model_metrics = train_and_validate_risk_model(valid)
    risk_scores = score_latest_day(clf, full_daily_reindexed) if clf is not None else pd.DataFrame()
    if len(risk_scores) > 0:
        risk_scores.to_csv(outdir / "dormancy_risk_scores.csv", index=False)
    if model_metrics:
        with open(outdir / "model_metrics.json", "w") as f:
            json.dump(model_metrics, f, indent=2)

    # Sharper second-layer signal: separates normal stage-riding idle time (waiting for
    # a fare, but still active most days) from bikes that have gone genuinely dark for
    # a sustained stretch. Bikes flagged here are pulled OUT of "Stage-Based / Walk-up
    # Riders" into their own 4th archetype, since sustained zero-activity is a materially
    # different operational state (possible breakdown/abandonment/lease drop-off), not
    # just a riding style.
    dropout = flag_sustained_dropout(full_daily_reindexed)
    archetypes = archetypes.merge(dropout, on="bike_id", how="left")
    dropout_mask = archetypes["sustained_dropout_flag"] == True
    n_reassigned = int(dropout_mask.sum())
    archetypes.loc[dropout_mask, "archetype_label"] = "Dormant / Dropped-Out"
    if n_reassigned:
        print(f"[Module 3] Reassigned {n_reassigned} bikes into a 4th archetype "
              f"'Dormant / Dropped-Out' (5+ consecutive fully-inactive days).")
    archetypes.to_csv(outdir / "bike_archetypes.csv", index=False)

    daily.to_csv(outdir / "daily_bike_features.csv", index=False)

    # ---- Module 1 (applied last, to outputs going out the door) ----
    if not args.no_anonymize:
        secret = get_hmac_secret()
        df_out = df.copy()
        df_out["bike_id"] = anonymize_series(df_out["bike_id"], secret)
        df_out[["dt", "latitude", "longitude", "bike_id", "speed_clean",
                "gap_flag", "spike_flag", "static_flag", "stop_cluster"]].to_csv(
            outdir / "anonymized_pings.csv", index=False)
    else:
        df[["dt", "latitude", "longitude", "bike_id", "speed_clean",
            "gap_flag", "spike_flag", "static_flag", "stop_cluster"]].to_csv(
            outdir / "anonymized_pings.csv", index=False)

    # ---- Plots ----
    plot_fleet_overview(archetypes, risk_scores, outdir / "fleet_overview.png")
    if args.bike:
        if args.bike in df["bike_id"].unique():
            plot_single_bike(df, daily, stop_summary, args.bike, outdir / f"{args.bike}_demo.png")
        else:
            print(f"WARNING: bike '{args.bike}' not found in dataset -- skipping demo plot.")
            args.bike = None

    # ---- Report ----
    write_report(outdir, df["bike_id"].nunique(), len(df), archetype_info,
                 archetypes, model_metrics, risk_scores, demo_bike=args.bike)

    print(f"\nDone. All outputs written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
