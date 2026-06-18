"""Train MARVIN on flow cytometry data."""

import dawgz
import torch
import wandb

from dataclasses import asdict, dataclass
from heracls.core import from_dict
from omegaconf import OmegaConf
from pathlib import Path
from typing import Literal

from marvin.data import CytometryDataset, DataLoaderConfig, compute_prior, infinite_stream
from marvin.metrics import compute_metrics, evaluate_testset
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


@dataclass(kw_only=True)
class TrainConfig:
    run_id: str
    out_dir: str
    data: str
    num_steps: int
    eval_every: int
    save_every: int
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

    dataset = CytometryDataset(cfg.data, cfg.model.M)
    train_ds, val_ds, test_ds = dataset.split()

    trainloader = cfg.train_loader.build(train_ds)
    valloader = cfg.val_loader.build(val_ds)
    testloader = cfg.test_loader.build(test_ds)

    train_stream = infinite_stream(trainloader)
    val_stream = infinite_stream(valloader)

    model = cfg.model.build().to(device)

    if cfg.discovery.enabled:
        prior_log = compute_prior(
            dataset.c, cfg.model.K, cfg.discovery.n_supp_clusters, cfg.discovery.unknown_mass
        )
        model.freeze_prior(prior_log.to(device))
    optimizer, warmup_scheduler, step_scheduler = cfg.optimizer.build(model)

    run = cfg.wandb.build(config={
        "model": f"MARVIN_{cfg.run_id}",
        "batch_size": cfg.train_loader.batch_size,
        "num_steps": cfg.num_steps,
        "learning_rate": cfg.optimizer.lr,
        "eval_every": cfg.eval_every,
        "factor": cfg.model.factor,
        "D": cfg.model.D,
        "lr": cfg.optimizer.lr,
    })

    print("\n\n= \t = \t = \t =")
    print(
        f" MARVIN {cfg.run_id} will train on {torch.cuda.device_count()} {device}"
        f" and has {sum(p.numel() for p in model.parameters()) / 1e6} M parameters",
        flush=True,
    )
    print("= \t = \t = \t =")

    prev_epoch = 0
    best_val_loss = float("inf")
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(asdict(cfg)), f"{cfg.out_dir}/config.yaml")
    model.train()
    for step in range(1, cfg.num_steps + 1):
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

        if step % cfg.eval_every == 0 or step == cfg.num_steps:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    run.log({f"grad_{name}": wandb.Histogram(param.grad.cpu().numpy())})

            model.eval()
            with torch.no_grad():
                _, (x_val, c_val) = next(val_stream)
                x_val, c_val = x_val.to(device), c_val.to(device)

                val_loss = model.loss_unsupervised(x_val)
                supervised_val = c_val >= 0
                x_sup_val, c_sup_val = x_val[supervised_val], c_val[supervised_val]
                if len(x_sup_val) > 0:
                    val_loss += model.loss_supervised(x_sup_val, c_sup_val)

                accuracy, f1score, balanced_accuracy = compute_metrics(model, x_val, c_val)

                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    torch.save(model.state_dict(), f"{cfg.out_dir}/MARVIN_{cfg.run_id}_best.pt")
            model.train()

            run.log({
                "step": step,
                "train_loss": loss.item(),
                "validation_loss": val_loss.item(),
                "accuracy": accuracy,
                "F1-score": f1score,
                "balanced accuracy": balanced_accuracy,
            })
            print(
                f"Step {step}/{cfg.num_steps},"
                f" val loss = {val_loss:.4f}, train loss = {loss.item():.4f}",
                flush=True,
            )

        if step % cfg.save_every == 0 or step == cfg.num_steps:
            checkpoint = f"{cfg.out_dir}/MARVIN_{cfg.run_id}_{step}.pt"
            torch.save(model.state_dict(), checkpoint)
            print(f"Model saved to {checkpoint}")

    model.load_state_dict(torch.load(f"{cfg.out_dir}/MARVIN_{cfg.run_id}_best.pt", weights_only=True))
    test_loss, accuracy, balanced_accuracy, f1score = evaluate_testset(model, testloader, device)
    run.log({
        "test_loss": test_loss.item() if hasattr(test_loss, "item") else test_loss,
        "accuracy : test": accuracy,
        "F1-score : test": f1score,
        "balanced accuracy : test": balanced_accuracy,
    }, step=cfg.num_steps)
    run.finish()


if __name__ == "__main__":
    cfg_dict = OmegaConf.to_object(
        OmegaConf.load(Path(__file__).parent.parent.parent / "config" / "train.yaml")
    )
    cfg = from_dict(TrainConfig, cfg_dict)
    train_job = dawgz.job(train, **cfg.slurm)(cfg)
    dawgz.schedule(train_job, backend=cfg.backend)
