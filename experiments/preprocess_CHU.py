"""Preprocess per-patient CSV files into a single training-ready CSV.

Directory structure expected:
    root/
        Hincourt/
            01. LY TOT HINCOURT.csv
            02. LT_NK HINCOURT.csv
            ...
        Peikrishvili/
            01. LY TOT PEIKRISHVILI.csv
            ...

Label is extracted from the filename by stripping the leading index and the
patient name suffix (e.g. "02. LT_NK PEIKRISHVILI.csv" → "LT_NK").
"""

import dawgz
import re
import pandas as pd

from pathlib import Path

from marvin.preprocessing import autologicle_transform, zscore_standardize

# ── Edit these ─────────────────────────────────────────────────────────────────
ROOT = Path("/home/fbodart/CHU/KIT BCP_ALL MRD (CYTOGNOS)/MO NORMALES")
OUTPUT = Path("/home/fbodart/CHU/marvin_preprocessed.csv")

PATIENTS = ["Hincourt", "Peikrishvili", "Spaubek"]

SKIP_LABELS = {"LY TOT", "DEBRIS", "DOUBLETS"}

DROP_COLS = {"FSC-A", "FSC-H", "FSC-W", "SSC-A", "SSC-H", "SSC-W", "TIME"}
# ──────────────────────────────────────────────────────────────────────────────


def extract_label(filename: str, patient: str) -> str:
    """'02. LT_NK PEIKRISHVILI.csv' + 'Peikrishvili' → 'LT_NK'"""
    name = Path(filename).stem
    name = re.sub(r"^\d+\.\s*", "", name)
    name = re.sub(re.escape(patient.upper()), "", name, flags=re.IGNORECASE)
    return name.strip()


def load_patient(patient_dir: Path, patient: str) -> pd.DataFrame:
    frames = []
    for csv_file in sorted(patient_dir.glob("*.csv")):
        label = extract_label(csv_file.name, patient)
        if label in SKIP_LABELS:
            continue

        df = pd.read_csv(csv_file, sep=";")

        # Comma-decimal → dot-decimal (European locale CSVs)
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = pd.to_numeric(
                    df[col].str.replace(",", ".", regex=False), errors="coerce"
                )

        df["label"] = label
        df["patient_id"] = patient
        frames.append(df)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def preprocess(patients: list[str], root: Path, output: Path) -> pd.DataFrame:
    all_frames = []
    for patient in patients:
        patient_dir = root / patient
        if not patient_dir.exists():
            print(f"Warning: {patient_dir} not found, skipping.")
            continue
        df = load_patient(patient_dir, patient)
        if df.empty:
            continue
        print(f"  {patient}: {len(df)} cells, labels: {sorted(df['label'].unique())}")
        all_frames.append(df)

    data = pd.concat(all_frames, ignore_index=True)

    drop = [c for c in DROP_COLS if c in data.columns]
    marker_cols = [c for c in data.columns if c not in drop and c not in ("label", "patient_id")]
    meta = data[["patient_id", "label"]]
    markers = data[marker_cols].astype(float)

    print(f"\nMarkers ({len(marker_cols)}): {marker_cols}")
    print(f"Total cells: {len(markers)}")

    markers = autologicle_transform(markers)
    markers = zscore_standardize(markers)

    result = pd.concat([markers, meta], axis=1)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"\nSaved → {output}")

    counts = result["label"].value_counts(normalize=True) * 100
    print("\nLabel distribution (%):")
    print(counts.round(1).to_string())

    return result


def run() -> None:
    print(f"Processing {len(PATIENTS)} patients from {ROOT}\n")
    preprocess(PATIENTS, ROOT, OUTPUT)


if __name__ == "__main__":
    job = dawgz.job(run, cpus=4, mem="8GB", time="1:00:00")()
    dawgz.schedule(job, backend="slurm")
