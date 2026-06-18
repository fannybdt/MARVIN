import itertools
import pandas as pd
import torch

from collections.abc import Iterator
from dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset, Subset


@dataclass(kw_only=True)
class DataLoaderConfig:
    batch_size: int
    shuffle: bool
    num_workers: int = 4
    drop_last: bool = False

    def build(self, dataset: Dataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
        )


class CytometryDataset(Dataset):
    def __init__(self, root: str, M: int, masked_classes: list[str] | None = None) -> None:
        data = pd.read_csv(root)
        data = data.sample(frac=1, random_state=42).reset_index(drop=True)
        if "label" not in data.columns:
            c = pd.array([-1] * len(data), dtype="int64")
        else:
            if masked_classes:
                available = sorted(data["label"].dropna().unique().tolist())
                unknown = [c for c in masked_classes if c not in available]
                if unknown:
                    raise ValueError(
                        f"Unknown masked_classes: {unknown}\n"
                        f"Available classes: {available}"
                    )
                data.loc[data["label"].isin(masked_classes), "label"] = None
            c = pd.factorize(data["label"], sort=True)[0]
        feature_cols = [col for col in data.columns if col not in ("label", "patient_id")][:M]
        x = data[feature_cols]
        self.x = torch.tensor(x.values, dtype=torch.float32)
        self.c = torch.tensor(c, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.c[idx]

    def split(self) -> tuple[Subset, Subset, Subset]:
        n = int(0.8 * len(self))
        n2 = int(0.9 * len(self))
        return (
            Subset(self, range(n)),
            Subset(self, range(n, n2)),
            Subset(self, range(n2, len(self))),
        )


def infinite_stream(loader: DataLoader) -> Iterator[tuple[int, tuple]]:
    for epoch in itertools.count():
        yield from ((epoch, batch) for batch in loader)


def compute_prior(c: torch.Tensor, K: int, n_supp_clusters: int, unknown_mass: float) -> torch.Tensor:
    labeled = c[c >= 0]
    C = int(labeled.max().item()) + 1
    assert C + n_supp_clusters == K, (
        f"n_labeled_classes ({C}) + n_supp_clusters ({n_supp_clusters}) must equal K ({K})"
    )
    counts = torch.bincount(labeled, minlength=C).float()
    probas = counts / counts.sum()
    probas_scaled = probas * (1.0 - unknown_mass)
    unknown = torch.full((n_supp_clusters,), unknown_mass / n_supp_clusters)
    prior = torch.cat([probas_scaled, unknown])
    return prior.log()
