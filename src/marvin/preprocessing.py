import fcsparser
import numpy as np
import pandas as pd
import re

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

_NON_MARKER_PATTERNS = re.compile(
    r"^(time|event_?no|event_?length|file_?offset|beadde|center|offset|residual|"
    r"width|dna|live|dead|viability|fsc|ssc|af|cisplatin|cell_?length)",
    re.IGNORECASE,
)
# If isotope prefix before the marker name, strip it
_ISOTOPE_PREFIX = re.compile(r"^(?:\d{2,3}[A-Z][a-z]{0,2}|[A-Z][a-z]?\d{2,3})(?:Di|Dd)?[_\-]")


def _clean_name(name: str) -> str:
    name = name.strip()
    name = _ISOTOPE_PREFIX.sub("", name)
    name = re.sub(r"\(.*?\)", "", name)
    name = re.sub(r"[\s_]+", "", name)
    name = re.sub(r"[^\w\-\/]", "", name)
    return name.strip()


def standardize_channel_names(
    df: pd.DataFrame, rename: dict[str, str] | None = None
) -> pd.DataFrame:
    df = df.rename(columns={c: _clean_name(c) for c in df.columns})
    if rename:
        df = df.rename(columns=rename)
    return df


def select_markers(
    df: pd.DataFrame,
    markers: list[str] | None = None,
    drop: list[str] | None = None,
) -> pd.DataFrame:
    if markers is not None:
        missing = [m for m in markers if m not in df.columns]
        if missing:
            raise ValueError(f"Markers not found: {missing}. Available: {list(df.columns)}")
        return df[markers]
    mask = df.columns.map(lambda c: bool(_NON_MARKER_PATTERNS.match(c)))
    df = df.loc[:, ~mask]
    if drop:
        df = df.drop(columns=[c for c in drop if c in df.columns])
    return df


def arcsinh_transform(df: pd.DataFrame, cofactor: float = 5.0) -> pd.DataFrame:
    """arcsinh(x / cofactor). Cofactor 5 is the CyTOF standard."""
    return pd.DataFrame(np.arcsinh(df.values / cofactor), index=df.index, columns=df.columns)


def autologicle_transform(df: pd.DataFrame, m: float = 4.5, a: float = 0.0) -> pd.DataFrame:
    """Per-channel logicle with W estimated from each channel's negative tail.

    W=0 for channels without negatives (CyTOF); W>0 for compensated flow data.
    Ref: Blampey et al., Scyan (2023); Parks & Moore, Cytometry A (2012).
    """
    try:
        from flowutils import transforms as ft
    except ImportError as err:
        raise ImportError("autologicle requires flowutils: run `uv add flowutils`") from err

    values = df.values.copy().astype(float)
    t = max(float(np.percentile(values[np.isfinite(values)], 99.99)), 1.0)

    result = np.empty_like(values)
    for i in range(values.shape[1]):
        col = np.clip(values[:, i], -t, t)
        r = float(np.percentile(col, 5))
        w = float(np.clip((m - np.log10(t / abs(r))) / 2.0, 0.0, m / 2.0)) if r < 0 else 0.0
        result[:, i] = ft.logicle(col, None, t=t, m=m, w=w, a=a)

    return pd.DataFrame(result, index=df.index, columns=df.columns)


def zscore_standardize(df: pd.DataFrame) -> pd.DataFrame:
    mean = df.values.mean(axis=0)
    std = df.values.std(axis=0)
    return pd.DataFrame((df.values - mean) / std, index=df.index, columns=df.columns)


def read_fcs(path: str | Path) -> pd.DataFrame:
    _, df = fcsparser.parse(str(path), reformat_meta=True)
    return df


@dataclass
class FCSFileConfig:
    path: str | Path
    patient_id: str | None = None
    label: str | None = None  # None → NaN in CSV → -1 in dataloader via pd.factorize

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if self.patient_id is None:
            self.patient_id = self.path.stem


@dataclass
class PreprocessingConfig:
    transform: Literal["arcsinh", "autologicle"] = "arcsinh"
    cofactor: float = 5.0  # arcsinh only; 5 for CyTOF, 150 for flow
    markers: list[str] | None = None
    channel_rename: dict[str, str] = field(default_factory=dict)
    drop_channels: list[str] = field(default_factory=list)
    add_patient_id: bool = True


def preprocess(
    files: list[FCSFileConfig],
    config: PreprocessingConfig | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Read FCS files => clean names => transform => z-score => save CSV."""
    if config is None:
        config = PreprocessingConfig()

    frames: list[pd.DataFrame] = []
    for fc in files:
        df = read_fcs(fc.path)
        df = standardize_channel_names(df, rename=config.channel_rename or None)
        df = select_markers(df, markers=config.markers, drop=config.drop_channels)
        if config.transform == "autologicle":
            df = autologicle_transform(df)
        else:
            df = arcsinh_transform(df, cofactor=config.cofactor)
        if config.add_patient_id:
            df["patient_id"] = fc.patient_id
        df["label"] = fc.label if fc.label is not None else np.nan
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    meta_cols = [c for c in ("patient_id", "label") if c in combined.columns]
    meta = combined[meta_cols]
    markers_df = zscore_standardize(combined.drop(columns=meta_cols))
    result = pd.concat([markers_df, meta], axis=1)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)

    return result
