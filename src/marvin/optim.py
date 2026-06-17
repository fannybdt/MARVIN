import torch

from dataclasses import dataclass
from torch.optim.lr_scheduler import StepLR

from marvin.model import MARVIN


class WarmupScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        base_lr: float,
        max_lr: float,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.current_step = 0

    def step(self) -> None:
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            lr = self.base_lr + (self.max_lr - self.base_lr) * (self.current_step / self.warmup_steps)
        else:
            lr = self.max_lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr


@dataclass(kw_only=True)
class AdamWConfig:
    lr: float
    base_lr: float
    warmup_steps: int
    step_size: int
    gamma: float

    def build(self, model: MARVIN) -> tuple[torch.optim.Optimizer, WarmupScheduler, StepLR]:
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)
        warmup_scheduler = WarmupScheduler(
            optimizer,
            warmup_steps=self.warmup_steps,
            base_lr=self.base_lr,
            max_lr=self.lr,
        )
        step_scheduler = StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        return optimizer, warmup_scheduler, step_scheduler
