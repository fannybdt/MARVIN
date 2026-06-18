"""Per-patient cell population analysis: unstimulated vs. peanut-stimulated conditions."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F

from heracls.core import from_dict
from omegaconf import OmegaConf
from pathlib import Path

from marvin.data import CytometryDataset
from marvin.model import MARVIN
from marvin.train import TrainConfig

DATA_PATH = Path("scyan_datasets/poised/poised_with_patients.csv")
PLOTS_DIR = Path("PLOTS")
OUT_DIR = Path("outputs/poised_discovery")

CONDITION_COL = "category"
STIM_VALUE = "Peanut stimulated"
UNSTIM_VALUE = "Unstimulated"
PATIENT_COL = "patient"
LABEL_COL = "label"
UNKNOWN_LABEL = "Unknown"

CLUSTER_OF_INTEREST = 22  # cluster index for the marker heatmap
BATCH_SIZE = 1024


def load_model(out_dir: Path, device: torch.device) -> tuple[MARVIN, int, list[str]]:
    """Load model from out_dir/config.yaml and return (model, n_labeled, cluster_names)."""
    cfg = from_dict(TrainConfig, OmegaConf.to_object(OmegaConf.load(out_dir / "config.yaml")))
    dataset = CytometryDataset(cfg.data, cfg.model.M, masked_classes=cfg.discovery.masked_classes)
    n_labeled = int(dataset.c[dataset.c >= 0].max().item()) + 1
    K = n_labeled + cfg.discovery.n_supp_clusters

    cluster_names = list(dataset.class_names)
    for i in range(cfg.discovery.n_supp_clusters):
        cluster_names.append(f"Subpopulation {i + 1}")

    model = cfg.model.build(K=K)
    model.load_state_dict(torch.load(out_dir / f"MARVIN_{cfg.run_id}_best.pt", weights_only=True))
    model.to(device).eval()
    return model, n_labeled, cluster_names


def predict(model: MARVIN, x: torch.Tensor, device: torch.device) -> np.ndarray:
    preds = []
    with torch.no_grad():
        for start in range(0, len(x), BATCH_SIZE):
            batch = x[start : start + BATCH_SIZE].to(device)
            preds.append(F.softmax(model.q_c_x(batch), dim=-1).argmax(dim=-1).cpu().numpy())
    return np.concatenate(preds)


def proportions(indices: np.ndarray, n: int) -> np.ndarray:
    counts = np.bincount(indices, minlength=n).astype(float)
    return counts / counts.sum()


def compute_patient_proportions(
    df: pd.DataFrame,
    model: MARVIN,
    n_markers: int,
    cluster_names: list[str],
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns four DataFrames indexed by patient:
        pred_stim, pred_unstim  — model-predicted proportions (n_patients × n_clusters)
        gt_stim,   gt_unstim    — ground-truth proportions    (n_patients × n_gt_classes)
    """
    K = len(cluster_names)
    gt_classes = sorted(c for c in df[LABEL_COL].dropna().unique() if c != UNKNOWN_LABEL)
    gt_idx = {name: i for i, name in enumerate(gt_classes)}
    patients = sorted(df[PATIENT_COL].unique())

    pred_stim_rows, pred_unstim_rows = [], []
    gt_stim_rows, gt_unstim_rows = [], []

    for patient in patients:
        pdata = df[df[PATIENT_COL] == patient]
        stim = pdata[(pdata[CONDITION_COL] == STIM_VALUE) & (pdata[LABEL_COL] != UNKNOWN_LABEL)]
        unstim = pdata[
            (pdata[CONDITION_COL] == UNSTIM_VALUE) & (pdata[LABEL_COL] != UNKNOWN_LABEL)
        ]

        for subset, pred_rows, gt_rows in [
            (stim, pred_stim_rows, gt_stim_rows),
            (unstim, pred_unstim_rows, gt_unstim_rows),
        ]:
            x = torch.tensor(subset.iloc[:, :n_markers].values, dtype=torch.float32)
            pred_rows.append(proportions(predict(model, x, device), K))

            gt_indices = np.array([gt_idx.get(l, -1) for l in subset[LABEL_COL]])
            valid = gt_indices >= 0
            gt_rows.append(proportions(gt_indices[valid], len(gt_classes)))

    index = pd.Index(patients, name="patient")
    return (
        pd.DataFrame(pred_stim_rows, index=index, columns=cluster_names),
        pd.DataFrame(pred_unstim_rows, index=index, columns=cluster_names),
        pd.DataFrame(gt_stim_rows, index=index, columns=gt_classes),
        pd.DataFrame(gt_unstim_rows, index=index, columns=gt_classes),
    )


def pct_variation(stim: pd.DataFrame, unstim: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    var = (stim - unstim) / (unstim + 1e-6) * 100
    return var.mean(), var.std()


def plot_variation_chart(
    pred_stim: pd.DataFrame,
    pred_unstim: pd.DataFrame,
    gt_stim: pd.DataFrame,
    gt_unstim: pd.DataFrame,
    output_path: Path,
) -> None:
    """Mean ± std % variation per cell type, model vs ground truth."""
    pred_mean, pred_std = pct_variation(pred_stim, pred_unstim)
    gt_mean, gt_std = pct_variation(gt_stim, gt_unstim)

    # Align GT to pred column order (NaN for supplementary clusters)
    gt_mean_aligned = pred_mean.index.map(lambda n: gt_mean.get(n, np.nan))
    gt_std_aligned = pred_mean.index.map(lambda n: gt_std.get(n, np.nan))
    has_gt = ~np.isnan(gt_mean_aligned.astype(float))

    x = np.arange(len(pred_mean))
    bar_w = 0.4

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(
        x, pred_mean.values, bar_w, yerr=pred_std.values, label="Model", color="#F65353", capsize=4
    )
    ax.bar(
        x[has_gt] + bar_w,
        gt_mean_aligned[has_gt].astype(float),
        bar_w,
        yerr=gt_std_aligned[has_gt].astype(float),
        label="Ground truth",
        color="#F59A23",
        capsize=4,
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x + bar_w / 2)
    ax.set_xticklabels(pred_mean.index, rotation=50, ha="right")
    ax.set_ylabel("Mean % variation (stim vs. unstim)")
    ax.set_title("Cell population shift after peanut stimulation")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_patient_trajectories(
    stim: pd.DataFrame,
    unstim: pd.DataFrame,
    output_path: Path,
    n_cols: int = 4,
) -> None:
    """Line per patient showing proportion shift unstim → stim, one subplot per cell type."""
    cell_types = stim.columns.tolist()
    n_rows = int(np.ceil(len(cell_types) / n_cols))
    palette = sns.color_palette("tab20", len(stim))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for j, ct in enumerate(cell_types):
        ax = axes[j]
        for patient, color in zip(stim.index, palette):
            ax.plot(
                ["Unstim", "Stim"],
                [unstim.loc[patient, ct] * 100, stim.loc[patient, ct] * 100],
                marker="o",
                color=color,
                linewidth=1.2,
            )
        ax.set_title(ct, fontsize=9)
        ax.set_ylabel("Proportion (%)", fontsize=8)
        ax.grid(alpha=0.3)

    for j in range(len(cell_types), len(axes)):
        fig.delaxes(axes[j])

    handles = [
        plt.Line2D([0], [0], color=c, marker="o", linestyle="", label=f"Patient {p}")
        for p, c in zip(stim.index, palette)
    ]
    fig.legend(handles=handles, loc="lower right", fontsize=9)
    plt.suptitle("Per-patient proportions: unstimulated vs. peanut stimulated", fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_marker_heatmap(
    df: pd.DataFrame,
    model: MARVIN,
    n_markers: int,
    cluster_idx: int,
    cluster_name: str,
    device: torch.device,
    output_path: Path,
) -> None:
    """Mean marker expression for cells predicted in a given cluster, stim vs unstim."""
    marker_names = df.columns[:n_markers].tolist()
    rows = {}
    for condition, label in [(UNSTIM_VALUE, "Unstimulated"), (STIM_VALUE, "Stimulated")]:
        subset = df[df[CONDITION_COL] == condition].reset_index(drop=True)
        x = torch.tensor(subset.iloc[:, :n_markers].values, dtype=torch.float32)
        mask = predict(model, x, device) == cluster_idx
        selected = subset.iloc[mask, :n_markers]
        rows[label] = (
            selected.mean(axis=0) if mask.any() else pd.Series(np.nan, index=marker_names)
        )

    mean_df = pd.DataFrame(rows, index=marker_names).T
    vmin, vmax = np.nanmin(mean_df.values), np.nanmax(mean_df.values)

    fig, axes = plt.subplots(2, 1, figsize=(16, 4), sharex=True)
    for ax, (label, row) in zip(axes, mean_df.iterrows()):
        sns.heatmap(
            row.values.reshape(1, -1),
            ax=ax,
            cmap="coolwarm",
            xticklabels=marker_names,
            yticklabels=[label],
            vmin=vmin,
            vmax=vmax,
            linewidths=0.2,
            linecolor="white",
            cbar_kws={"shrink": 0.5},
        )
        ax.tick_params(axis="x", labelsize=7)
    axes[0].set_title(f"Mean marker expression — {cluster_name}")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    PLOTS_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, n_labeled, cluster_names = load_model(OUT_DIR, device)
    df = pd.read_csv(DATA_PATH)
    n_markers = model.M

    print(f"Patients: {df[PATIENT_COL].nunique()}, cells: {len(df)}")

    pred_stim, pred_unstim, gt_stim, gt_unstim = compute_patient_proportions(
        df, model, n_markers, cluster_names, device
    )

    plot_variation_chart(
        pred_stim,
        pred_unstim,
        gt_stim,
        gt_unstim,
        PLOTS_DIR / "variation_stim_vs_unstim.svg",
    )
    plot_patient_trajectories(
        pred_stim,
        pred_unstim,
        PLOTS_DIR / "patient_trajectories.svg",
    )

    cluster_name = (
        cluster_names[CLUSTER_OF_INTEREST]
        if CLUSTER_OF_INTEREST < len(cluster_names)
        else f"Cluster {CLUSTER_OF_INTEREST}"
    )
    plot_marker_heatmap(
        df,
        model,
        n_markers,
        CLUSTER_OF_INTEREST,
        cluster_name,
        device,
        PLOTS_DIR / f"marker_heatmap_{cluster_name.replace(' ', '_')}.svg",
    )
