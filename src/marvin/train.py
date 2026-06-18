"""Train MARVIN on flow cytometry data."""

import dawgz
import torch
import wandb

from dataclasses import asdict, dataclass, field
from heracls.core import from_dict
from omegaconf import OmegaConf
from pathlib import Path
from typing import Literal

from marvin.data import CytometryDataset, DataLoaderConfig, compute_prior, infinite_stream
from marvin.metrics import compute_metrics, discovery_bar_chart, evaluate_testset
from marvin.model import MARVINConfig
from marvin.optim import AdamWConfig


@dataclass(kw_only=True)
class WandBConfig:
    project: str
    dir: str
    entity: str | None = None
    name: str | None = None
    id: str | None = None
    disable: bool = False

    def build(self, config: dict):
        return wandb.init(
            project=self.project,
            entity=self.entity,
            name=self.name,
            id=self.id,
            dir=self.dir,
            config=config,
            mode="disabled" if self.disable else "online",
        )


@dataclass(kw_only=True)
class DiscoveryConfig:
    enabled: bool = False
    n_supp_clusters: int = 0
    unknown_mass: float = 0.002
    masked_classes: list[str] = field(default_factory=list)


@dataclass(kw_only=True)
class TrainConfig:
    run_id: str
    out_dir: str
    data: str
    num_steps: int | None = None
    num_epochs: int | None = None
    eval_every: int = 500
    save_every: int = 5000
    masked: bool
    mask_ratio: float
    train_loader: DataLoaderConfig
    val_loader: DataLoaderConfig
    test_loader: DataLoaderConfig
    model: MARVINConfig
    optimizer: AdamWConfig
    wandb: WandBConfig
    discovery: DiscoveryConfig
    backend: Literal["slurm", "async"]
    slurm: dict


def train(cfg: TrainConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if (cfg.num_steps is None) == (cfg.num_epochs is None):
        raise ValueError("Specify exactly one of num_steps or num_epochs in the config.")

    dataset = CytometryDataset(cfg.data, cfg.model.M, masked_classes=cfg.discovery.masked_classes)
    train_ds, val_ds, test_ds = dataset.split()

    if cfg.num_epochs is not None:
        steps_per_epoch = len(train_ds) // cfg.train_loader.batch_size
        num_steps = cfg.num_epochs * steps_per_epoch
    else:
        num_steps = cfg.num_steps

    trainloader = cfg.train_loader.build(train_ds)
    valloader = cfg.val_loader.build(val_ds)
    testloader = cfg.test_loader.build(test_ds)

    train_stream = infinite_stream(trainloader)
    val_stream = infinite_stream(valloader)

    n_labeled = int(dataset.c[dataset.c >= 0].max().item()) + 1
    K = n_labeled + cfg.discovery.n_supp_clusters
    model = cfg.model.build(K=K).to(device)

    if cfg.discovery.enabled:
        prior_log = compute_prior(
            dataset.c, K, cfg.discovery.n_supp_clusters, cfg.discovery.unknown_mass
        )
        model.freeze_prior(prior_log.to(device))
    optimizer, warmup_scheduler, step_scheduler = cfg.optimizer.build(model)

    run = cfg.wandb.build(config={
        "model": f"MARVIN_{cfg.run_id}",
        "batch_size": cfg.train_loader.batch_size,
        "num_steps": num_steps,
        "learning_rate": cfg.optimizer.lr,
        "eval_every": cfg.eval_every,
        "factor": cfg.model.factor,
        "D": cfg.model.D,
        "lr": cfg.optimizer.lr,
    })

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\n{'='*50}")
    print(f"  MARVIN — {cfg.run_id}")
    print(f"{'='*50}")
    print(f"  device: {device}")
    print(f"  parameters: {n_params:.2f}M")
    print(f"  steps: {num_steps}")
    print(f"  K={K} ({n_labeled} labeled + {cfg.discovery.n_supp_clusters} supplementary)")
    print("  labeled classes:")
    for i, name in enumerate(dataset.class_names):
        print(f"    [{i}] {name}")
    print(f"{'='*50}\n")

    prev_epoch = 0
    best_val_loss = float("inf")
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(asdict(cfg)), f"{cfg.out_dir}/config.yaml")
    model.train()
    for step in range(1, num_steps + 1):
        epoch, (x, c) = next(train_stream)
        x, c = x.to(device), c.to(device)

        optimizer.zero_grad()

        loss = model.loss_unsupervised(x)
        unique_classes = torch.unique(c)

        if cfg.masked:
            mask = torch.rand(len(c)) < cfg.mask_ratio
            for cls in unique_classes:
                class_indices = (c == cls).nonzero(as_tuple=True)[0]
                if len(class_indices) > 0:
                    mask[class_indices[0]] = False
            c[mask] = -1

        supervised = c >= 0
        x_sup, c_sup = x[supervised], c[supervised]
        if len(x_sup) > 0:
            loss += model.loss_supervised(x_sup, c_sup)

        loss.backward()

        if step <= cfg.optimizer.warmup_steps:
            warmup_scheduler.step()

        optimizer.step()

        if epoch > prev_epoch:
            step_scheduler.step()
            prev_epoch = epoch

        if step % cfg.eval_every == 0 or step == num_steps:
            model.eval()
            with torch.no_grad():
                _, (x_val, c_val) = next(val_stream)
                x_val, c_val = x_val.to(device), c_val.to(device)

                val_loss = model.loss_unsupervised(x_val)
                supervised_val = c_val >= 0
                x_sup_val, c_sup_val = x_val[supervised_val], c_val[supervised_val]
                if len(x_sup_val) > 0:
                    val_loss += model.loss_supervised(x_sup_val, c_sup_val)

                accuracy, f1score, balanced_accuracy, discovered = compute_metrics(model, x_val, c_val, n_labeled)

                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    torch.save(model.state_dict(), f"{cfg.out_dir}/MARVIN_{cfg.run_id}_best.pt")
            model.train()

            log = {
                "step": step,
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": loss.item(),
                "validation_loss": val_loss.item(),
                "accuracy": accuracy,
                "F1-score": f1score,
                "balanced accuracy": balanced_accuracy,
            }
            if cfg.discovery.enabled and discovered:
                log["discovery/supplementary_assignments"] = discovery_bar_chart(discovered, dataset.class_names)
            run.log(log)
            print(
                f"Step {step}/{num_steps},"
                f" val loss = {val_loss:.4f}, train loss = {loss.item():.4f}",
                flush=True,
            )

        if step % cfg.save_every == 0 or step == num_steps:
            checkpoint = f"{cfg.out_dir}/MARVIN_{cfg.run_id}_{step}.pt"
            torch.save(model.state_dict(), checkpoint)
            print(f"Model saved to {checkpoint}")

    model.load_state_dict(torch.load(f"{cfg.out_dir}/MARVIN_{cfg.run_id}_best.pt", weights_only=True))
    test_loss, accuracy, balanced_accuracy, f1score = evaluate_testset(model, testloader, device, n_labeled)
    run.log({
        "test_loss": test_loss.item() if hasattr(test_loss, "item") else test_loss,
        "accuracy : test": accuracy,
        "F1-score : test": f1score,
        "balanced accuracy : test": balanced_accuracy,
    }, step=num_steps)
    run.finish()


if __name__ == "__main__":
    cfg_dict = OmegaConf.to_object(
        OmegaConf.load(Path(__file__).parent.parent.parent / "config" / "train.yaml")
    )
    cfg = from_dict(TrainConfig, cfg_dict)
    train_job = dawgz.job(train, **cfg.slurm)(cfg)
    dawgz.schedule(train_job, backend=cfg.backend)
