"""Analyze which labeled populations are assigned to supplementary clusters after training."""

import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter, defaultdict
from pathlib import Path

from marvin.data import CytometryDataset
from marvin.model import MARVIN


def build_assignment_matrix(
    model: MARVIN,
    dataset: CytometryDataset,
    n_labeled: int,
    batch_size: int = 2048,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """Return a DataFrame (rows=supplementary clusters, cols=cell types) with cell counts."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_supp = model.K - n_labeled
    counts: dict[int, Counter] = defaultdict(Counter)

    model.eval().to(device)
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            idx = range(start, min(start + batch_size, len(dataset)))
            x = torch.stack([dataset[i][0] for i in idx]).to(device)
            c = torch.stack([dataset[i][1] for i in idx])

            c_pred = F.softmax(model.q_c_x(x), dim=-1).argmax(dim=-1).cpu()

            for pred, true in zip(c_pred.tolist(), c.tolist()):
                if pred >= n_labeled and true >= 0:
                    counts[pred][dataset.class_names[true]] += 1

    rows = {}
    for k in range(n_labeled, model.K):
        label = f"Supp [{k - n_labeled}]"
        rows[label] = {name: counts[k].get(name, 0) for name in dataset.class_names}

    return pd.DataFrame(rows, index=dataset.class_names).T  # shape: (n_supp, n_classes)


def plot_discovery(
    matrix: pd.DataFrame,
    output_path: str | Path | None = None,
) -> None:
    """One bar chart per supplementary cluster showing cell type distribution."""
    if matrix.empty or matrix.values.sum() == 0:
        print("No labeled cells were assigned to any supplementary cluster.")
        return

    n_supp = len(matrix)
    fig, axes = plt.subplots(1, n_supp, figsize=(5 * n_supp, max(4, len(matrix.columns) * 0.4 + 1)))
    if n_supp == 1:
        axes = [axes]

    for ax, (cluster_label, row) in zip(axes, matrix.iterrows()):
        total = row.sum()
        if total == 0:
            ax.set_title(f"{cluster_label}\n(empty)", fontsize=10)
            ax.axis("off")
            continue

        top = row[row > 0].sort_values(ascending=True)
        colors = sns.color_palette("muted", len(top))
        top.plot.barh(ax=ax, color=colors)
        ax.set_title(f"{cluster_label}\n(n={int(total)})", fontsize=10)
        ax.set_xlabel("cells")
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {output_path}")
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    import sys
    from heracls.core import from_dict
    from omegaconf import OmegaConf
    from marvin.train import TrainConfig

    if len(sys.argv) < 2:
        print("Usage: python experiments/discovery_analysis.py <out_dir> [output_figure.png]")
        sys.exit(1)

    out_dir = Path(sys.argv[1])
    fig_path = Path(sys.argv[2]) if len(sys.argv) > 2 else out_dir / "discovery_analysis.png"

    cfg = from_dict(TrainConfig, OmegaConf.to_object(OmegaConf.load(out_dir / "config.yaml")))

    dataset = CytometryDataset(cfg.data, cfg.model.M, masked_classes=cfg.discovery.masked_classes)
    n_labeled = int(dataset.c[dataset.c >= 0].max().item()) + 1
    K = n_labeled + cfg.discovery.n_supp_clusters

    model = cfg.model.build(K=K)
    model.load_state_dict(torch.load(out_dir / f"MARVIN_{cfg.run_id}_best.pt", weights_only=True))

    matrix = build_assignment_matrix(model, dataset, n_labeled)
    print(matrix.to_string())
    plot_discovery(matrix, output_path=fig_path)
