import torch

from dataclasses import dataclass
from torch import nn
from torch.nn import functional as F


class Concat(nn.Module):
    def __init__(self, f_in: list[int], f_out: int):
        super().__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.projs = nn.ModuleList([nn.Linear(f, f_out) for f in f_in])

    def forward(self, xs):
        return torch.cat([proj(x) for proj, x in zip(self.projs, xs)], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, f_in: int, f_out: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(f_in, f_out),
            nn.SiLU(),
            nn.Linear(f_out, f_out),
            nn.Dropout(dropout),
        )
        self.shortcut = nn.Identity() if f_in == f_out else nn.Linear(f_in, f_out)

    def forward(self, x):
        return F.relu(self.net(x) + self.shortcut(x))


class ResidualNetwork(nn.Module):
    def __init__(self, f_in: int, f_out: int, factor: int, dropout: float, num_blocks: int):
        super().__init__()
        layers = [ResidualBlock(f_in, factor * f_in, dropout)]
        for _ in range(num_blocks - 1):
            layers.append(ResidualBlock(factor * f_in, factor * f_in, dropout))
        layers.append(nn.Linear(factor * f_in, f_out))
        self.blocks = nn.ModuleList(layers)

    def forward(self, x):
        y = x
        for block in self.blocks:
            y = block(y)
        return y


class MARVIN(nn.Module):
    def __init__(self, M: int, K: int, D: int, factor: int, dropout: float, num_blocks: int):
        super().__init__()
        self.M = M
        self.K = K
        self.D = D
        self.factor = factor
        self.dropout = dropout

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # q(c, z|x) = q(c|x) q(z|c, x)
        self.q_c_x = ResidualNetwork(M, K, factor, dropout, num_blocks)

        self.q_z_cx = nn.Sequential(
            Concat([K, M], (K + M) // 2),
            nn.ReLU(),
            ResidualNetwork((K + M) // 2 * 2, 2 * D, factor, dropout, num_blocks),
        )

        self.p_c = nn.Parameter(torch.zeros(self.K))
        self.p_z_c = ResidualNetwork(K, 2 * D, factor, dropout, num_blocks)
        self.p_x_z = ResidualNetwork(D, 2 * M, factor, dropout, num_blocks)

    def loss_unsupervised(self, x):
        elbo = 0.0

        p_c = F.softmax(self.p_c, dim=0)
        p_c = torch.clamp(p_c, min=1e-6)
        q_c_x = F.softmax(self.q_c_x(x), dim=-1)
        q_c_x = torch.clamp(q_c_x, min=1e-6)

        for k in range(self.K):
            c = torch.tensor([k] * len(x)).to(self.device)
            codes = self.one_hot(c, self.K).to(self.device)
            q_z_cx = self.q_z_cx([codes, x])
            mu_z = q_z_cx[:, : self.D]
            logvar_z = q_z_cx[:, self.D :]
            z = mu_z + logvar_z.exp() ** 0.5 * torch.randn_like(mu_z)

            x_recon = self.p_x_z(z)
            mu_x = x_recon[:, : self.M]
            logvar_x = x_recon[:, self.M :]
            logvar_x = torch.clamp(logvar_x, min=-6, max=3)
            elbo += q_c_x[:, k] * (0.5 * ((x - mu_x) ** 2 / logvar_x.exp() + logvar_x)).sum(dim=-1)

            p_z_c = self.p_z_c(codes)
            elbo += q_c_x[:, k] * (q_c_x[:, k].log() - p_c[k].log())
            elbo += q_c_x[:, k] * self.kl_gaussian(
                mu_z, logvar_z, p_z_c[:, : self.D], p_z_c[:, self.D :]
            )

        return elbo.mean()

    def loss_supervised(self, x, c):
        q_c_x = self.q_c_x(x)
        return F.cross_entropy(q_c_x, c)

    def one_hot(self, c, K):
        return torch.eye(K, device=c.device)[c]

    def freeze_prior(self, log_prior: torch.Tensor) -> None:
        with torch.no_grad():
            self.p_c.copy_(log_prior)
        self.p_c.requires_grad_(False)

    def kl_gaussian(self, mu_q, logvar_q, mu_p, logvar_p):
        return (
            0.5 * (logvar_p - logvar_q)
            + 0.5 * (logvar_q.exp() + (mu_q - mu_p) ** 2) / logvar_p.exp()
            - 0.5
        ).sum(dim=-1)


@dataclass(kw_only=True)
class MARVINConfig:
    M: int
    K: int | None = None  # if None, derived from data: (C - n_masked) + n_supp_clusters
    D: int = 32
    factor: int = 4
    dropout: float = 0.2
    num_blocks: int = 2

    def build(self, K: int | None = None) -> MARVIN:
        k = K if K is not None else self.K
        if k is None:
            raise ValueError("K must be set in config or derived from data")
        return MARVIN(self.M, k, self.D, self.factor, self.dropout, self.num_blocks)
