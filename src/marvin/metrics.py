import torch
import torch.nn.functional as F

from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader

from .model import MARVIN


def compute_metrics(
    model: MARVIN,
    x: torch.Tensor,
    c: torch.Tensor,
) -> tuple[float, float, float]:
    q_c_x = F.softmax(model.q_c_x(x), dim=-1)
    print(f"q(c|x) = {q_c_x}", flush=True)
    c_pred = torch.multinomial(q_c_x, 1, replacement=True).squeeze()

    accuracy = (c_pred == c).sum().item() / c.size(0)
    f1 = f1_score(c.cpu().numpy(), c_pred.cpu().numpy(), average="macro")
    bal_accuracy = balanced_accuracy_score(c.cpu().numpy(), c_pred.cpu().numpy())
    return accuracy, f1, bal_accuracy


def evaluate_testset(
    model: MARVIN,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, float, float, float]:
    model.eval()
    test_loss: torch.Tensor | float = 0.0
    accuracy = 0.0
    f1 = 0.0
    bal_accuracy = 0.0

    with torch.no_grad():
        for x, c_true in loader:
            x = x.to(device)
            c_true = c_true.to(device)

            loss = model.loss_unsupervised(x)
            supervised = c_true >= 0
            x_sup, c_sup = x[supervised], c_true[supervised]
            if len(x_sup) > 0:
                loss += model.loss_supervised(x_sup, c_sup)
            test_loss += loss

            q_c_x = F.softmax(model.q_c_x(x), dim=-1)
            c = torch.multinomial(q_c_x, 1, replacement=True).squeeze()

            accuracy += (c == c_true).sum().item() / c_true.size(0)
            f1 += f1_score(c_true.cpu().numpy(), c.cpu().numpy(), average="macro")
            bal_accuracy += balanced_accuracy_score(c_true.cpu().numpy(), c.cpu().numpy())

    n = len(loader)
    return test_loss / n, accuracy / n, bal_accuracy / n, f1 / n
