"""
Build whaling_voyages.db from three source files:
  - aowl_20250916_part1.txt  (at-sea observations, rows 1-200000)
  - aowl_20250916_part2.txt  (at-sea observations, rows 200001+)
  - voyages_20240911.txt     (voyage metadata)

VoyageID links both tables.
"""

import csv
import sqlite3
import sys
from pathlib import Path

AOWL_PART1 = Path("/root/.claude/uploads/af5474ea-9fcf-5f2f-9ebb-be93426a65c1/7aec134d-aowl_20250916_part1.txt")
AOWL_PART2 = Path("/root/.claude/uploads/af5474ea-9fcf-5f2f-9ebb-be93426a65c1/0a8c9c41-aowl_20250916_part2.txt")
VOYAGES    = Path("/root/.claude/uploads/af5474ea-9fcf-5f2f-9ebb-be93426a65c1/3c546209-voyages_20240911.txt")
DB_PATH    = Path("whaling_voyages.db")


DDL = """
CREATE TABLE IF NOT EXISTS voyages (
    voyageID          TEXT NOT NULL,
    voyageRank        INTEGER,
    voyageName        TEXT,
    port              TEXT,
    sailingFrom       TEXT,
    ground            TEXT,
    yearOut           INTEGER,
    dayOut            TEXT,
    yearIn            INTEGER,
    dayIn             TEXT,
    returnCode        TEXT,
    agentID           TEXT,
    agent             TEXT,
    bone              REAL,
    sperm             REAL,
    oil               REAL,
    customsDistrict   TEXT,
    logbookExists     TEXT,
    aowlLink          TEXT,
    crewListLink      TEXT,
    source            TEXT,
    masterID          TEXT,
    master            TEXT,
    fate              TEXT,
    birth             TEXT,
    birthLocation     TEXT,
    death             TEXT,
    deathLocation     TEXT,
    wife              TEXT,
    wifeToSea         TEXT,
    vitalRecordsSource TEXT,
    laterOccupation   TEXT,
    vesselID          TEXT,
    vessel            TEXT,
    rig               TEXT,
    tonnage           TEXT,
    builtPlace        TEXT,
    builtDate         TEXT,
    endStatus         TEXT,
    dennisWood        TEXT,
    logbookScan       TEXT,
    PRIMARY KEY (voyageID, voyageRank)
);

CREATE TABLE IF NOT EXISTS aowl_observations (
    sequence   INTEGER PRIMARY KEY,
    voyageID   TEXT NOT NULL,
    lat        REAL,
    lon        REAL,
    day        INTEGER,
    month      INTEGER,
    year       INTEGER,
    encounter  TEXT,
    species    TEXT,
    n_struck   INTEGER,
    n_tried    INTEGER,
    place      TEXT,
    source     TEXT,
    remarks    TEXT,
    FOREIGN KEY (voyageID) REFERENCES voyages(voyageID)
);

CREATE INDEX IF NOT EXISTS idx_aowl_voyageID ON aowl_observations(voyageID);
CREATE INDEX IF NOT EXISTS idx_aowl_date     ON aowl_observations(year, month, day);
CREATE INDEX IF NOT EXISTS idx_aowl_encounter ON aowl_observations(encounter);
CREATE INDEX IF NOT EXISTS idx_voyages_vessel ON voyages(vesselID);
"""


def coerce_null(val):
    """Return None for 'NULL' or empty strings."""
    if val is None:
        return None
    s = val.strip()
    return None if s in ("NULL", "") else s


def coerce_int(val):
    s = coerce_null(val)
    if s is None:
        return None
    try:
        return int(s)
    except ValueError:
        return None


def coerce_float(val):
    s = coerce_null(val)
    if s is None:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_voyages(conn, path: Path):
    print(f"Loading voyages from {path.name} …", flush=True)
    inserted = 0
    with path.open(encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        rows = []
        for row in reader:
            rows.append((
                coerce_null(row.get("voyageID")),
                coerce_int(row.get("voyageRank")),
                coerce_null(row.get("voyageName")),
                coerce_null(row.get("port")),
                coerce_null(row.get("sailingFrom")),
                coerce_null(row.get("ground")),
                coerce_int(row.get("yearOut")),
                coerce_null(row.get("dayOut")),
                coerce_int(row.get("yearIn")),
                coerce_null(row.get("dayIn")),
                coerce_null(row.get("returnCode")),
                coerce_null(row.get("agentID")),
                coerce_null(row.get("agent")),
                coerce_float(row.get("bone")),
                coerce_float(row.get("sperm")),
                coerce_float(row.get("oil")),
                coerce_null(row.get("customsDistrict")),
                coerce_null(row.get("logbookExists")),
                coerce_null(row.get("aowlLink")),
                coerce_null(row.get("crewListLink")),
                coerce_null(row.get("source")),
                coerce_null(row.get("masterID")),
                coerce_null(row.get("master")),
                coerce_null(row.get("fate")),
                coerce_null(row.get("birth")),
                coerce_null(row.get("birthLocation")),
                coerce_null(row.get("death")),
                coerce_null(row.get("deathLocation")),
                coerce_null(row.get("wife")),
                coerce_null(row.get("wifeToSea")),
                coerce_null(row.get("vitalRecordsSource")),
                coerce_null(row.get("laterOccupation")),
                coerce_null(row.get("vesselID")),
                coerce_null(row.get("vessel")),
                coerce_null(row.get("rig")),
                coerce_null(row.get("tonnage")),
                coerce_null(row.get("builtPlace")),
                coerce_null(row.get("builtDate")),
                coerce_null(row.get("end")),
                coerce_null(row.get("dennisWood")),
                coerce_null(row.get("logbookScan")),
            ))
            if len(rows) >= 5000:
                conn.executemany(
                    "INSERT OR IGNORE INTO voyages VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    rows,
                )
                inserted += len(rows)
                rows.clear()

        if rows:
            conn.executemany(
                "INSERT OR IGNORE INTO voyages VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                rows,
            )
            inserted += len(rows)

    conn.commit()
    print(f"  → {inserted} voyage rows inserted.")


def load_aowl(conn, path: Path):
    print(f"Loading observations from {path.name} …", flush=True)
    inserted = 0
    with path.open(encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        rows = []
        for row in reader:
            rows.append((
                coerce_int(row.get("sequence")),
                coerce_null(row.get("VoyageID")),
                coerce_float(row.get("Lat")),
                coerce_float(row.get("Lon")),
                coerce_int(row.get("Day")),
                coerce_int(row.get("Month")),
                coerce_int(row.get("Year")),
                coerce_null(row.get("Encounter")),
                coerce_null(row.get("Species")),
                coerce_int(row.get("NStruck")),
                coerce_int(row.get("NTried")),
                coerce_null(row.get("Place")),
                coerce_null(row.get("Source")),
                coerce_null(row.get("Remarks")),
            ))
            if len(rows) >= 10000:
                conn.executemany(
                    "INSERT OR IGNORE INTO aowl_observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    rows,
                )
                inserted += len(rows)
                rows.clear()

        if rows:
            conn.executemany(
                "INSERT OR IGNORE INTO aowl_observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                rows,
            )
            inserted += len(rows)

    conn.commit()
    print(f"  → {inserted} observation rows inserted.")


def main():
    if DB_PATH.exists():
        DB_PATH.unlink()
        print(f"Removed existing {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(DDL)
    conn.commit()

    load_voyages(conn, VOYAGES)
    load_aowl(conn, AOWL_PART1)
    load_aowl(conn, AOWL_PART2)

    # Summary stats
    print("\nDatabase summary:")
    for table in ("voyages", "aowl_observations"):
        (n,) = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        print(f"  {table}: {n:,} rows")

    # Distinct voyageIDs in each table
    (nv,) = conn.execute("SELECT COUNT(DISTINCT voyageID) FROM voyages").fetchone()
    (na,) = conn.execute("SELECT COUNT(DISTINCT voyageID) FROM aowl_observations").fetchone()
    print(f"  Distinct voyageIDs in voyages:           {nv:,}")
    print(f"  Distinct voyageIDs in aowl_observations: {na:,}")

    # Observation date range
    row = conn.execute("SELECT MIN(year), MAX(year) FROM aowl_observations").fetchone()
    print(f"  Observation years: {row[0]} – {row[1]}")

    # Encounter type breakdown
    print("\nEncounter breakdown:")
    for enc, cnt in conn.execute(
        "SELECT encounter, COUNT(*) FROM aowl_observations GROUP BY encounter ORDER BY COUNT(*) DESC"
    ):
        print(f"  {enc or 'NULL':12s}: {cnt:,}")

    conn.close()
    print(f"\nDatabase written to {DB_PATH.resolve()}")


if __name__ == "__main__":
    sys.exit(main())
