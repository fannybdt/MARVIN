import torch
import torch.nn.functional as F
import warnings

from collections import Counter
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader

from marvin.model import MARVIN


def compute_metrics(
    model: MARVIN,
    x: torch.Tensor,
    c: torch.Tensor,
    n_labeled: int,
) -> tuple[float, float, float, list[int]]:
    labeled = c >= 0
    x_lab, c_lab = x[labeled], c[labeled]
    if len(c_lab) == 0:
        return 0.0, 0.0, 0.0, []

    q_c_x = F.softmax(model.q_c_x(x_lab), dim=-1)
    c_pred = torch.multinomial(q_c_x, 1, replacement=True).squeeze()

    # True labels of cells pushed to supplementary clusters
    in_supp = c_pred >= n_labeled
    discovered = c_lab[in_supp].cpu().tolist()

    c_pred_k, c_lab_k = c_pred[~in_supp], c_lab[~in_supp]
    if len(c_pred_k) == 0:
        return 0.0, 0.0, 0.0, discovered

    accuracy = (c_pred_k == c_lab_k).sum().item() / c_lab_k.size(0)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        f1 = f1_score(c_lab_k.cpu().numpy(), c_pred_k.cpu().numpy(), average="macro")
        bal_accuracy = balanced_accuracy_score(c_lab_k.cpu().numpy(), c_pred_k.cpu().numpy())
    return accuracy, f1, bal_accuracy, discovered


def discovery_bar_chart(discovered: list[int], class_names: list[str]) -> object:
    """WandB bar chart of true labels pushed to supplementary clusters."""
    import wandb
    counts = Counter(discovered)
    table = wandb.Table(columns=["class", "count"])
    for idx, name in enumerate(class_names):
        table.add_data(name, counts.get(idx, 0))
    return wandb.plot.bar(table, "class", "count", title="Cells in supplementary clusters")


def evaluate_testset(
    model: MARVIN,
    loader: DataLoader,
    device: torch.device,
    n_labeled: int,
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

            labeled = c_true >= 0
            x_lab, c_lab = x[labeled], c_true[labeled]
            if len(c_lab) == 0:
                continue

            q_c_x = F.softmax(model.q_c_x(x_lab), dim=-1)
            c_pred = torch.multinomial(q_c_x, 1, replacement=True).squeeze()

            known = c_pred < n_labeled
            c_pred_k, c_lab_k = c_pred[known], c_lab[known]
            if len(c_pred_k) == 0:
                continue

            accuracy += (c_pred_k == c_lab_k).sum().item() / c_lab_k.size(0)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
                f1 += f1_score(c_lab_k.cpu().numpy(), c_pred_k.cpu().numpy(), average="macro")
                bal_accuracy += balanced_accuracy_score(c_lab_k.cpu().numpy(), c_pred_k.cpu().numpy())

    n = len(loader)
    return test_loss / n, accuracy / n, bal_accuracy / n, f1 / n
