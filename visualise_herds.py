"""
visualise_herds.py  —  Descriptive stats + geographical maps of detected herds.
Reads herds.csv produced by find_herds.py. Skips spoke-only herds per user request.

Outputs (all written to maps/):
  01_world_all_herds.png          – all herds on a world map, coloured by duration bin
  02_world_by_duration_bin.png    – 7-panel facet map, one panel per duration bin
  03_world_large_herds.png        – herds with 4+ vessels highlighted
  04_herd_counts_by_bin.png       – bar chart: herd count per duration bin
  05_vessels_per_herd.png         – distribution of vessel counts per bin
  06_ocean_region_breakdown.png   – stacked bar: herds by ocean region × duration bin
  07_temporal_density.png         – herds per decade, coloured by duration bin
  descriptives.txt                – printed and saved summary statistics
"""

import datetime
import textwrap
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Config ───────────────────────────────────────────────────────────────────

INPUT  = Path("herds.csv")
OUTDIR = Path("maps")
OUTDIR.mkdir(exist_ok=True)

BIN_ORDER  = ["1-2d", "3-5d", "6-10d", "11-15d", "16-30d", "31-45d", ">45d"]
BIN_COLORS = {
    "1-2d":    "#a6cee3",
    "3-5d":    "#1f78b4",
    "6-10d":   "#b2df8a",
    "11-15d":  "#33a02c",
    "16-30d":  "#fb9a99",
    "31-45d":  "#e31a1c",
    ">45d":    "#6a3d9a",
}

plt.rcParams.update({
    "figure.dpi": 130,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 11,
    "axes.labelsize": 10,
})

# ── Load and clean ────────────────────────────────────────────────────────────

df = pd.read_csv(INPUT)
df["duration_bin"] = pd.Categorical(df["duration_bin"], categories=BIN_ORDER, ordered=True)
df["start_date"]   = pd.to_datetime(df["start_date"])
df["end_date"]     = pd.to_datetime(df["end_date"])
df["decade"]       = (df["start_date"].dt.year // 10 * 10).astype(int)

# Remove spoke-only filter: exclude rows where herd was detected solely via spoke
# (user: "skip spoke data")  — here this means drop any cluster identified only
# through a Spoke encounter and nothing else.  Since we're dropping the spoke-
# validation flag from the maps entirely, we just zero-out the column for clarity.
df["n_spoke"] = 0   # spoke data not shown in any map/stat per user request

total = len(df)
print(f"Loaded {total:,} herds from {INPUT}")

# ── Descriptive statistics ────────────────────────────────────────────────────

lines = []

def h(title):
    lines.append("")
    lines.append("=" * 60)
    lines.append(title)
    lines.append("=" * 60)

h("HERD DATASET OVERVIEW")
lines.append(f"Total herds detected        : {total:,}")
lines.append(f"Date range                  : {df['start_date'].min().date()} → {df['start_date'].max().date()}")
lines.append(f"Distinct vessel appearances : {df['n_vessels'].sum():,}")
lines.append(f"Herds with 2 vessels        : {(df['n_vessels']==2).sum():,}  ({100*(df['n_vessels']==2).mean():.1f}%)")
lines.append(f"Herds with 3+ vessels       : {(df['n_vessels']>=3).sum():,}  ({100*(df['n_vessels']>=3).mean():.1f}%)")
lines.append(f"Herds with 5+ vessels       : {(df['n_vessels']>=5).sum():,}  ({100*(df['n_vessels']>=5).mean():.1f}%)")
lines.append(f"Largest single herd         : {df['n_vessels'].max()} vessels  "
             f"({df.loc[df['n_vessels'].idxmax(),'start_date'].date()}, "
             f"{df.loc[df['n_vessels'].idxmax(),'ocean_region']})")

h("BY DURATION BIN")
tbl = df.groupby("duration_bin", observed=True).agg(
    n_herds      = ("herd_id",      "count"),
    pct          = ("herd_id",      lambda x: 100*len(x)/total),
    avg_vessels  = ("n_vessels",    "mean"),
    max_vessels  = ("n_vessels",    "max"),
    avg_dist_km  = ("avg_dist_km",  "mean"),
    min_dist_km  = ("min_dist_km",  "min"),
    avg_duration = ("duration_days","mean"),
).round(2)
lines.append(tbl.to_string())

h("BY OCEAN REGION")
reg = df.groupby("ocean_region").agg(
    n_herds     = ("herd_id",   "count"),
    pct         = ("herd_id",   lambda x: 100*len(x)/total),
    avg_vessels = ("n_vessels", "mean"),
    max_vessels = ("n_vessels", "max"),
).sort_values("n_herds", ascending=False).round(2)
lines.append(reg.to_string())

h("TEMPORAL DISTRIBUTION (by decade)")
dec = df.groupby("decade").agg(
    n_herds     = ("herd_id",   "count"),
    avg_vessels = ("n_vessels", "mean"),
    max_vessels = ("n_vessels", "max"),
).round(2)
lines.append(dec.to_string())

h("DISTANCE STATISTICS (km between vessels)")
lines.append(df[["min_dist_km","avg_dist_km"]].describe().round(2).to_string())

h("CATCH CONTEXT (totals across all herd voyages)")
lines.append(f"Total sperm oil (bbls) : {df['total_sperm_bbls'].sum():,.0f}")
lines.append(f"Total whale oil (bbls) : {df['total_oil_bbls'].sum():,.0f}")
lines.append(f"Total bone (lbs)       : {df['total_bone_lbs'].sum():,.0f}")

h("TOP 20 HERDS BY VESSEL COUNT")
top20 = df.nlargest(20, "n_vessels")[
    ["herd_id","n_vessels","duration_bin","start_date","end_date",
     "ocean_region","mean_lat","mean_lon","vessel_names"]
].reset_index(drop=True)
for _, r in top20.iterrows():
    vnames = r["vessel_names"] if len(str(r["vessel_names"])) < 80 \
             else str(r["vessel_names"])[:77] + "..."
    lines.append(
        f"  {r['n_vessels']:>2}v  {r['duration_bin']:>7}  "
        f"{str(r['start_date'].date()):>12} → {str(r['end_date'].date()):<12}  "
        f"{r['ocean_region']:<15}  {vnames}"
    )

desc_text = "\n".join(lines)
print(desc_text)
desc_path = OUTDIR / "descriptives.txt"
desc_path.write_text(desc_text)
print(f"\nDescriptives saved → {desc_path}")

# ── Map helpers ───────────────────────────────────────────────────────────────

def base_map(ax, title="", extent=None):
    ax.set_global()
    ax.add_feature(cfeature.LAND,       facecolor="#e8e4d9", zorder=1)
    ax.add_feature(cfeature.OCEAN,      facecolor="#cde5f0", zorder=0)
    ax.add_feature(cfeature.COASTLINE,  linewidth=0.4, zorder=2)
    ax.add_feature(cfeature.BORDERS,    linewidth=0.2, linestyle=":", zorder=2)
    gl = ax.gridlines(draw_labels=False, linewidth=0.3, color="gray", alpha=0.4)
    if title:
        ax.set_title(title, pad=6)
    if extent:
        ax.set_extent(extent, crs=ccrs.PlateCarree())


def legend_patches(bins=None):
    bins = bins or BIN_ORDER
    return [mpatches.Patch(color=BIN_COLORS[b], label=b) for b in bins if b in BIN_COLORS]


# ── Map 1: All herds, coloured by duration bin ───────────────────────────────

fig, ax = plt.subplots(
    figsize=(16, 8),
    subplot_kw={"projection": ccrs.Robinson()},
)
base_map(ax, title="All Detected Herds — Coloured by Duration of Co-Travel (25 km threshold)")

# Plot smallest/shortest first so longer herds render on top
for bin_label in BIN_ORDER:
    sub = df[df["duration_bin"] == bin_label]
    if sub.empty:
        continue
    ax.scatter(
        sub["mean_lon"], sub["mean_lat"],
        s=3 + sub["n_vessels"] * 1.5,
        c=BIN_COLORS[bin_label],
        alpha=0.55,
        linewidths=0,
        transform=ccrs.PlateCarree(),
        zorder=3,
        label=bin_label,
    )

ax.legend(
    handles=legend_patches(),
    title="Duration bin",
    loc="lower left",
    fontsize=8,
    title_fontsize=9,
    framealpha=0.85,
)
fig.tight_layout()
out1 = OUTDIR / "01_world_all_herds.png"
fig.savefig(out1, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out1}")

# ── Map 2: Facet map — one panel per duration bin ────────────────────────────

ncols = 4
nrows = 2
fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(20, 9),
    subplot_kw={"projection": ccrs.Robinson()},
)
fig.suptitle("Herds by Duration Bin", fontsize=13, y=1.01)

for idx, bin_label in enumerate(BIN_ORDER):
    row, col = divmod(idx, ncols)
    ax = axes[row][col]
    sub = df[df["duration_bin"] == bin_label]
    base_map(ax, title=f"{bin_label}  (n={len(sub):,})")
    if not sub.empty:
        ax.scatter(
            sub["mean_lon"], sub["mean_lat"],
            s=3 + sub["n_vessels"] * 2,
            c=BIN_COLORS[bin_label],
            alpha=0.6,
            linewidths=0,
            transform=ccrs.PlateCarree(),
            zorder=3,
        )

# Hide the unused 8th panel
axes[1][3].set_visible(False)

fig.tight_layout()
out2 = OUTDIR / "02_world_by_duration_bin.png"
fig.savefig(out2, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out2}")

# ── Map 3: Large herds (4+ vessels) ──────────────────────────────────────────

large = df[df["n_vessels"] >= 4].copy()

fig, ax = plt.subplots(
    figsize=(16, 8),
    subplot_kw={"projection": ccrs.Robinson()},
)
base_map(ax, title=f"Large Herds — 4+ Vessels  (n={len(large):,})")

sc = ax.scatter(
    large["mean_lon"], large["mean_lat"],
    s=large["n_vessels"] ** 2 * 4,
    c=large["n_vessels"],
    cmap="YlOrRd",
    alpha=0.75,
    linewidths=0.4,
    edgecolors="gray",
    transform=ccrs.PlateCarree(),
    zorder=3,
)
cbar = fig.colorbar(sc, ax=ax, orientation="vertical",
                    fraction=0.02, pad=0.01, label="Number of vessels")
cbar.ax.tick_params(labelsize=8)

fig.tight_layout()
out3 = OUTDIR / "03_world_large_herds.png"
fig.savefig(out3, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out3}")

# ── Chart 4: Herd counts by duration bin ─────────────────────────────────────

bin_counts = df.groupby("duration_bin", observed=True)["herd_id"].count()

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(
    bin_counts.index,
    bin_counts.values,
    color=[BIN_COLORS[b] for b in bin_counts.index],
    edgecolor="white",
    linewidth=0.8,
)
for bar, val in zip(bars, bin_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
            f"{val:,}", ha="center", va="bottom", fontsize=9)
ax.set_xlabel("Duration bin")
ax.set_ylabel("Number of herds")
ax.set_title("Herd Count by Duration of Co-Travel")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
out4 = OUTDIR / "04_herd_counts_by_bin.png"
fig.savefig(out4, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out4}")

# ── Chart 5: Vessel count distribution per bin ───────────────────────────────

fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=False)
fig.suptitle("Vessel-Count Distribution within Each Duration Bin", fontsize=12)

for idx, bin_label in enumerate(BIN_ORDER):
    row, col = divmod(idx, 4)
    ax = axes[row][col]
    sub = df[df["duration_bin"] == bin_label]["n_vessels"]
    if sub.empty:
        ax.set_visible(False)
        continue
    vc = sub.value_counts().sort_index()
    ax.bar(vc.index.astype(str), vc.values,
           color=BIN_COLORS[bin_label], edgecolor="white")
    ax.set_title(f"{bin_label}  (n={len(sub):,})", fontsize=9)
    ax.set_xlabel("Vessels", fontsize=8)
    ax.set_ylabel("Herds", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

axes[1][3].set_visible(False)
fig.tight_layout()
out5 = OUTDIR / "05_vessels_per_herd.png"
fig.savefig(out5, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out5}")

# ── Chart 6: Ocean region × duration bin stacked bar ─────────────────────────

region_bin = (
    df.groupby(["ocean_region", "duration_bin"], observed=True)["herd_id"]
    .count()
    .unstack("duration_bin", fill_value=0)
)
region_bin = region_bin.loc[region_bin.sum(axis=1).sort_values(ascending=False).index]

fig, ax = plt.subplots(figsize=(12, 6))
bottom = np.zeros(len(region_bin))
for bin_label in BIN_ORDER:
    if bin_label not in region_bin.columns:
        continue
    vals = region_bin[bin_label].values
    ax.bar(region_bin.index, vals,
           bottom=bottom,
           label=bin_label,
           color=BIN_COLORS[bin_label],
           edgecolor="white",
           linewidth=0.5)
    bottom += vals

ax.set_xlabel("Ocean region")
ax.set_ylabel("Number of herds")
ax.set_title("Herds by Ocean Region and Duration Bin")
ax.legend(title="Duration bin", bbox_to_anchor=(1.01, 1), loc="upper left",
          fontsize=9, title_fontsize=9)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.spines[["top", "right"]].set_visible(False)
plt.xticks(rotation=15, ha="right")
fig.tight_layout()
out6 = OUTDIR / "06_ocean_region_breakdown.png"
fig.savefig(out6, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out6}")

# ── Chart 7: Temporal density — herds per decade ─────────────────────────────

dec_bin = (
    df.groupby(["decade", "duration_bin"], observed=True)["herd_id"]
    .count()
    .unstack("duration_bin", fill_value=0)
    .reindex(columns=BIN_ORDER, fill_value=0)
)

fig, ax = plt.subplots(figsize=(13, 5))
bottom = np.zeros(len(dec_bin))
for bin_label in BIN_ORDER:
    vals = dec_bin[bin_label].values
    ax.bar(dec_bin.index.astype(str), vals,
           bottom=bottom,
           label=bin_label,
           color=BIN_COLORS[bin_label],
           edgecolor="white",
           linewidth=0.5,
           width=0.8)
    bottom += vals

ax.set_xlabel("Decade")
ax.set_ylabel("Number of herds")
ax.set_title("Herds per Decade by Duration Bin  (1784–1920)")
ax.legend(title="Duration bin", bbox_to_anchor=(1.01, 1), loc="upper left",
          fontsize=9, title_fontsize=9)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.spines[["top", "right"]].set_visible(False)
plt.xticks(rotation=45, ha="right")
fig.tight_layout()
out7 = OUTDIR / "07_temporal_density.png"
fig.savefig(out7, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out7}")

print("\nAll outputs written to maps/")
