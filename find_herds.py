#!/usr/bin/env python3
"""
find_herds.py — Detect whaling vessel herds using HDBSCAN.

Finds groups of 2+ vessels that were within visual range (~25 km) of each other
for any duration, classified into duration bins from 1-2 days up to >45 days.

Usage:
    python find_herds.py                              # full run, defaults
    python find_herds.py --diagnostic                 # parameter sweep
    python find_herds.py --year-range 1840 1870       # era-specific
    python find_herds.py --spatial-threshold 10       # tight convoy only
    python find_herds.py --geojson                    # also write GeoJSON

Install: pip install hdbscan scikit-learn pandas numpy
"""

import argparse
import datetime
import json
import sqlite3
import sys

import hdbscan
import numpy as np
import pandas as pd


# ── Duration bin boundaries (days) ──────────────────────────────────────────

DURATION_BINS = [
    (1,  2,  "1-2d"),
    (3,  5,  "3-5d"),
    (6,  10, "6-10d"),
    (11, 15, "11-15d"),
    (16, 30, "16-30d"),
    (31, 45, "31-45d"),
    (46, 99999, ">45d"),
]


def assign_duration_bin(days: int) -> str:
    for lo, hi, label in DURATION_BINS:
        if lo <= days <= hi:
            return label
    return ">45d"


def assign_ocean_region(lat: float, lon: float) -> str:
    if lat > 60:
        return "Arctic"
    if lat < -55:
        return "Antarctic"
    lon = ((lon + 180) % 360) - 180
    if -80 <= lon <= 20:
        return "S Atlantic" if lat < 0 else "N Atlantic"
    if 20 < lon <= 120:
        return "Indian Ocean"
    return "S Pacific" if lat < 0 else "N Pacific"


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Detect whaling vessel herds via single-pass HDBSCAN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--db", default="whaling_voyages.db")
    p.add_argument(
        "--spatial-threshold", type=float, default=25.0, metavar="KM",
        help="Visual range in km. Sets both spatial proximity and time-axis scale "
             "(1 day = 1 × spatial-threshold km). Try 10 (tight convoy), "
             "25 (visual range), 50 (loose fishing ground).",
    )
    p.add_argument(
        "--min-cluster-size", type=int, default=2,
        help="Min HDBSCAN cluster size (observations). 2 captures 1-day meetings.",
    )
    p.add_argument(
        "--min-samples", type=int, default=1,
        help="HDBSCAN min_samples. 1 = no noise filtering.",
    )
    p.add_argument(
        "--min-vessels", type=int, default=2,
        help="Min distinct vessels per cluster to call it a herd.",
    )
    p.add_argument(
        "--min-voyage-days", type=int, default=1,
        help="Min days a voyage must appear in a cluster to count as a member.",
    )
    p.add_argument(
        "--year-range", type=int, nargs=2, default=None, metavar=("START", "END"),
        help="Restrict to observations within this year range.",
    )
    p.add_argument("--output", default="herds.csv")
    p.add_argument("--geojson", action="store_true", help="Also write herds.geojson")
    p.add_argument(
        "--diagnostic", action="store_true",
        help="Sweep spatial-threshold ∈ {10, 25, 50} km and print sensitivity table.",
    )
    return p.parse_args()


# ── Data loading ─────────────────────────────────────────────────────────────

def load_data(db_path: str, year_range=None):
    """Load non-Townsend observations joined with primary voyage metadata."""
    conn = sqlite3.connect(db_path)

    where_clauses = ["o.source != 'Townsend'", "o.day >= 1"]
    params = []
    if year_range:
        where_clauses.append("o.year BETWEEN ? AND ?")
        params.extend(year_range)
    where = " AND ".join(where_clauses)

    obs_df = pd.read_sql_query(
        f"""
        SELECT o.sequence, o.voyageID,
               o.lat, o.lon,
               o.day, o.month, o.year,
               o.encounter, o.n_struck, o.source,
               v.vessel, v.voyageName, v.ground
        FROM aowl_observations o
        LEFT JOIN voyages v ON o.voyageID = v.voyageID AND v.voyageRank = 1
        WHERE {where}
        """,
        conn, params=params,
    )

    voyage_meta = pd.read_sql_query(
        """
        SELECT voyageID, vessel, voyageName, ground,
               COALESCE(sperm, 0) AS sperm,
               COALESCE(oil,   0) AS oil,
               COALESCE(bone,  0) AS bone,
               yearOut, yearIn
        FROM voyages WHERE voyageRank = 1
        """,
        conn,
    ).drop_duplicates("voyageID").set_index("voyageID")

    conn.close()

    n_voyages = obs_df["voyageID"].nunique()
    print(f"Loaded {len(obs_df):,} observations across {n_voyages:,} voyages "
          f"(Townsend excluded)")
    return obs_df, voyage_meta


# ── Preparation ──────────────────────────────────────────────────────────────

def to_ordinal(day, month, year) -> int | None:
    try:
        return datetime.date(int(year), int(month), int(day)).toordinal()
    except (ValueError, TypeError):
        return None


def aggregate_voyage_days(obs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ordinal dates; average duplicate (voyageID, date) positions.
    Must run before HDBSCAN — duplicate positions on the same day would
    make one ship appear twice and skew cluster density.
    """
    obs_df = obs_df.copy()
    obs_df["ordinal"] = obs_df.apply(
        lambda r: to_ordinal(r["day"], r["month"], r["year"]), axis=1
    )
    obs_df = obs_df.dropna(subset=["ordinal"])
    obs_df["ordinal"] = obs_df["ordinal"].astype(int)

    # Aggregate: mean lat/lon; concatenate encounter types; sum struck counts
    agg = (
        obs_df.groupby(["voyageID", "ordinal"], as_index=False)
        .agg(
            lat=("lat", "mean"),
            lon=("lon", "mean"),
            encounter=("encounter", lambda x: "|".join(x.dropna().unique())),
            n_struck=("n_struck", "sum"),
            vessel=("vessel", "first"),
            voyageName=("voyageName", "first"),
            ground=("ground", "first"),
            source=("source", "first"),
        )
    )

    removed = len(obs_df) - len(agg)
    if removed:
        print(f"  Aggregated {removed:,} duplicate voyage-day rows into single positions")
    return agg


# ── Feature engineering ──────────────────────────────────────────────────────

def build_features(daily_df: pd.DataFrame, spatial_threshold_km: float) -> np.ndarray:
    """
    Convert observations to 3D metric space (lat_km, lon_km, time_km).

    lat_km  = lat_deg × 111
    lon_km  = lon_deg × 111 × cos(lat_rad)   adjusts for polar compression
    time_km = ordinal × spatial_threshold_km   so 1 day = 1 × threshold in time axis

    With spatial_threshold=25 km: points within 25 km on the same day are as
    "close" as points 25 km apart on consecutive days.
    """
    lat_rad = np.radians(daily_df["lat"].values)
    lat_km = daily_df["lat"].values * 111.0
    lon_km = daily_df["lon"].values * 111.0 * np.cos(lat_rad)
    time_km = daily_df["ordinal"].values * spatial_threshold_km

    return np.column_stack([lat_km, lon_km, time_km])


# ── HDBSCAN ──────────────────────────────────────────────────────────────────

def run_hdbscan(features: np.ndarray, min_cluster_size: int, min_samples: int) -> np.ndarray:
    """Fit HDBSCAN and return cluster label per observation (-1 = noise)."""
    print(f"  Running HDBSCAN on {len(features):,} points "
          f"(min_cluster_size={min_cluster_size}, min_samples={min_samples}) …",
          flush=True)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        core_dist_n_jobs=-1,
    )
    labels = clusterer.fit_predict(features)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    print(f"  → {n_clusters:,} raw clusters, {n_noise:,} noise observations")
    return labels


# ── Aggregation ──────────────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized Haversine distance in km."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2))
         * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def aggregate_herds(
    daily_df: pd.DataFrame,
    labels: np.ndarray,
    voyage_meta: pd.DataFrame,
    min_vessels: int,
    min_voyage_days: int,
    obs_df_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build per-herd summary rows from HDBSCAN labels.
    Returns (herds_df, obs_labelled_df).
    """
    df = daily_df.copy()
    df["herd_cluster_id"] = labels

    # Attach ordinal and labels to raw obs for the per-obs output file
    obs_out = obs_df_raw[["sequence", "voyageID", "lat", "lon",
                           "day", "month", "year"]].copy()
    obs_out["herd_cluster_id"] = labels

    # Only process valid clusters (not noise)
    clustered = df[df["herd_cluster_id"] >= 0].copy()
    if clustered.empty:
        print("  No valid clusters found.")
        return pd.DataFrame(), obs_out

    # Count days each voyage appears in each cluster
    voyage_days = (
        clustered.groupby(["herd_cluster_id", "voyageID"])
        .size()
        .reset_index(name="days_in_cluster")
    )
    # Apply min-voyage-days filter
    voyage_days = voyage_days[voyage_days["days_in_cluster"] >= min_voyage_days]

    # Count qualifying vessels per cluster
    vessel_counts = voyage_days.groupby("herd_cluster_id")["voyageID"].nunique()
    valid_clusters = vessel_counts[vessel_counts >= min_vessels].index

    print(f"  {len(valid_clusters):,} herds after filtering "
          f"(min_vessels={min_vessels}, min_voyage_days={min_voyage_days})")

    # Build summary rows
    # Pre-index raw obs by (voyageID, ordinal) for encounter lookup
    enc_lookup = (
        obs_df_raw.set_index(["voyageID", "ordinal"])["encounter"]
        if "ordinal" in obs_df_raw.columns
        else None
    )

    rows = []
    for cid in valid_clusters:
        c = clustered[clustered["herd_cluster_id"] == cid]
        vd = voyage_days[voyage_days["herd_cluster_id"] == cid]
        voyage_ids = sorted(vd["voyageID"].unique())

        start_ord = int(c["ordinal"].min())
        end_ord = int(c["ordinal"].max())
        start_date = datetime.date.fromordinal(start_ord)
        end_date = datetime.date.fromordinal(end_ord)
        duration_days = (end_date - start_date).days + 1

        # Geographic centroid (circular mean for longitude)
        lons_rad = np.radians(c["lon"].values)
        mean_lon = float(np.degrees(
            np.arctan2(np.sin(lons_rad).mean(), np.cos(lons_rad).mean())
        ))
        mean_lat = float(c["lat"].mean())

        # Pairwise distances: sample up to 500 observation pairs to keep it fast
        if len(c) >= 2:
            sample = c.sample(min(500, len(c)), random_state=42)
            lats = sample["lat"].values
            lons = sample["lon"].values
            i, j = np.triu_indices(len(lats), k=1)
            if len(i):
                dists = haversine_km(lats[i], lons[i], lats[j], lons[j])
                min_dist = float(dists.min())
                avg_dist = float(dists.mean())
            else:
                min_dist = avg_dist = float("nan")
        else:
            min_dist = avg_dist = float("nan")

        # Encounter counts across member voyages during cluster date range
        n_sight = n_strike = n_spoke = 0
        member_obs = clustered[
            (clustered["herd_cluster_id"] == cid) &
            (clustered["voyageID"].isin(voyage_ids))
        ]
        for enc_str in member_obs["encounter"].dropna():
            for e in enc_str.split("|"):
                e = e.strip()
                if e == "Sight":
                    n_sight += 1
                elif e == "Strike":
                    n_strike += 1
                elif e == "Spoke":
                    n_spoke += 1

        # Catch totals and vessel names from voyage metadata
        total_sperm = total_oil = total_bone = 0.0
        vessel_names = []
        for vid in voyage_ids:
            if vid in voyage_meta.index:
                m = voyage_meta.loc[vid]
                total_sperm += float(m.get("sperm") or 0)
                total_oil += float(m.get("oil") or 0)
                total_bone += float(m.get("bone") or 0)
                vname = str(m.get("vessel") or vid)
                if vname not in vessel_names:
                    vessel_names.append(vname)

        rows.append({
            "herd_id": int(cid),
            "duration_bin": assign_duration_bin(duration_days),
            "n_vessels": len(voyage_ids),
            "n_obs_total": len(c),
            "voyageIDs": "|".join(voyage_ids),
            "vessel_names": "|".join(vessel_names),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "duration_days": duration_days,
            "mean_lat": round(mean_lat, 3),
            "mean_lon": round(mean_lon, 3),
            "ocean_region": assign_ocean_region(mean_lat, mean_lon),
            "min_dist_km": round(min_dist, 2) if not np.isnan(min_dist) else None,
            "avg_dist_km": round(avg_dist, 2) if not np.isnan(avg_dist) else None,
            "n_sight": n_sight,
            "n_strike": n_strike,
            "n_spoke": n_spoke,
            "total_sperm_bbls": round(total_sperm, 1),
            "total_oil_bbls": round(total_oil, 1),
            "total_bone_lbs": round(total_bone, 1),
        })

    herds_df = pd.DataFrame(rows).sort_values(
        ["n_vessels", "duration_days"], ascending=False
    ).reset_index(drop=True)

    return herds_df, obs_out


# ── Export ───────────────────────────────────────────────────────────────────

def export_results(
    herds_df: pd.DataFrame,
    obs_out: pd.DataFrame,
    output_path: str,
    write_geojson: bool,
):
    herds_df.to_csv(output_path, index=False)
    print(f"  Herd summary → {output_path}")

    obs_path = output_path.replace(".csv", "_obs.csv")
    obs_out.to_csv(obs_path, index=False)
    print(f"  Observation labels → {obs_path}")

    if write_geojson:
        features = []
        for _, row in herds_df.iterrows():
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [row["mean_lon"], row["mean_lat"]],
                },
                "properties": {
                    k: (None if (isinstance(v, float) and np.isnan(v)) else v)
                    for k, v in row.items()
                    if k not in ("mean_lat", "mean_lon")
                },
            })
        geo_path = output_path.replace(".csv", ".geojson")
        with open(geo_path, "w") as f:
            json.dump({"type": "FeatureCollection", "features": features}, f, indent=2)
        print(f"  GeoJSON → {geo_path}")


# ── Print summary table ───────────────────────────────────────────────────────

def print_summary(herds_df: pd.DataFrame):
    if herds_df.empty:
        print("No herds found.")
        return

    print(f"\n{'='*70}")
    print(f"HERD SUMMARY  ({len(herds_df):,} herds total)")
    print(f"{'='*70}")

    # By duration bin
    print("\nHerds by duration bin:")
    bin_order = ["1-2d", "3-5d", "6-10d", "11-15d", "16-30d", "31-45d", ">45d"]
    counts = herds_df.groupby("duration_bin").agg(
        n_herds=("herd_id", "count"),
        avg_vessels=("n_vessels", "mean"),
        max_vessels=("n_vessels", "max"),
        n_spoke=("n_spoke", "sum"),
    ).reindex([b for b in bin_order if b in herds_df["duration_bin"].unique()])
    print(counts.to_string())

    # Top 10 by vessel count
    print("\nTop 10 herds by vessel count:")
    top = herds_df.head(10)[
        ["herd_id", "duration_bin", "n_vessels", "start_date", "end_date",
         "ocean_region", "n_spoke", "vessel_names"]
    ]
    for _, r in top.iterrows():
        vnames = r["vessel_names"]
        if len(vnames) > 60:
            vnames = vnames[:57] + "..."
        print(f"  [{r['herd_id']:>6}] {r['duration_bin']:>7}  "
              f"{r['n_vessels']:>3} vessels  "
              f"{r['start_date']} → {r['end_date']}  "
              f"{r['ocean_region']:<15}  "
              f"spoke={r['n_spoke']}  {vnames}")

    # Spoke-validated herds
    spoke = herds_df[herds_df["n_spoke"] > 0]
    if not spoke.empty:
        print(f"\nGround-truth validated herds (n_spoke > 0): {len(spoke)}")
        for _, r in spoke.iterrows():
            print(f"  [{r['herd_id']:>6}] {r['start_date']} → {r['end_date']}  "
                  f"{r['n_vessels']} vessels  {r['ocean_region']}  "
                  f"spoke={r['n_spoke']}  {r['vessel_names'][:80]}")


# ── Diagnostic sweep ─────────────────────────────────────────────────────────

def run_diagnostic(daily_df: pd.DataFrame, args):
    thresholds = [10.0, 25.0, 50.0]
    print(f"\n{'='*70}")
    print("DIAGNOSTIC: sensitivity sweep")
    print(f"  min_cluster_size={args.min_cluster_size}, "
          f"min_vessels={args.min_vessels}, "
          f"min_voyage_days={args.min_voyage_days}")
    print(f"{'='*70}")
    header = (f"{'threshold_km':>14} | {'raw_clusters':>12} | "
              + "  ".join(f"{b:>8}" for _, _, b in DURATION_BINS))
    print(header)
    print("-" * len(header))

    for thresh in thresholds:
        features = build_features(daily_df, thresh)
        labels = run_hdbscan(features, args.min_cluster_size, args.min_samples)

        # Count clusters by duration bin (without full aggregation)
        df = daily_df.copy()
        df["herd_cluster_id"] = labels
        clustered = df[df["herd_cluster_id"] >= 0]

        raw_clusters = int(labels.max()) + 1 if labels.max() >= 0 else 0

        bin_counts = {}
        for cid in range(raw_clusters):
            c = clustered[clustered["herd_cluster_id"] == cid]
            if c.empty:
                continue
            dur = int(c["ordinal"].max()) - int(c["ordinal"].min()) + 1
            b = assign_duration_bin(dur)
            bin_counts[b] = bin_counts.get(b, 0) + 1

        row = (f"{thresh:>14.0f} | {raw_clusters:>12,} | "
               + "  ".join(
                   f"{bin_counts.get(b, 0):>8,}" for _, _, b in DURATION_BINS
               ))
        print(row)

    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print(f"\n{'='*70}")
    print("Whaling Vessel Herd Detector")
    print(f"  DB:                {args.db}")
    print(f"  Spatial threshold: {args.spatial_threshold} km")
    if args.year_range:
        print(f"  Year range:        {args.year_range[0]}–{args.year_range[1]}")
    print(f"{'='*70}\n")

    # Load
    obs_df, voyage_meta = load_data(args.db, args.year_range)

    # Prepare
    print("Preparing daily positions …")
    daily_df = aggregate_voyage_days(obs_df)
    # Carry ordinal back to obs_df for encounter lookups
    ordinal_map = daily_df.set_index(["voyageID", "ordinal"])
    obs_df = obs_df.copy()
    obs_df["ordinal"] = obs_df.apply(
        lambda r: to_ordinal(r["day"], r["month"], r["year"]), axis=1
    )

    if args.diagnostic:
        run_diagnostic(daily_df, args)
        return

    # Features
    print(f"Building 3D metric features (threshold={args.spatial_threshold} km) …")
    features = build_features(daily_df, args.spatial_threshold)

    # HDBSCAN
    labels = run_hdbscan(features, args.min_cluster_size, args.min_samples)

    # Aggregate
    print("Aggregating herds …")
    herds_df, obs_out = aggregate_herds(
        daily_df, labels, voyage_meta,
        args.min_vessels, args.min_voyage_days,
        obs_df,
    )

    # Print summary
    print_summary(herds_df)

    # Export
    print(f"\nExporting …")
    export_results(herds_df, obs_out, args.output, args.geojson)

    print("\nDone.")


if __name__ == "__main__":
    sys.exit(main())
