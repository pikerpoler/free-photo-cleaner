"""Training-process checks on the current 53-feature schema.

The plan dropped Live Photo, portrait, panorama, HDR, burst, and GPS.
WhatsApp stays, even though it is constant on this export.

Metrics are on the held-out test fold. MLP numbers use the epoch with the
best validation log loss, and the last epoch is reported beside it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit

from evaluate_models import (
    MLP,
    build_matrix,
    fit_gbdt_clean,
    load_cache,
    make_splits,
    mlp_predict,
    pack,
    positive_proba,
    score_metrics,
)
from photo_features import FEATURE_NAMES

ROOT = Path(__file__).resolve().parent
DROPPED = ("live_photo", "portrait", "panorama", "hdr", "burst", "has_gps")
SEEDS = 10


def column_index(names: list[str], wanted: list[str]) -> list[int]:
    return [names.index(name) for name in wanted]


def take_columns(x: np.ndarray, names: list[str], drop: tuple[str, ...]) -> tuple[np.ndarray, list[str]]:
    keep_names = [name for name in names if name not in drop]
    return x[:, column_index(names, keep_names)], keep_names


class ColumnScaler:
    def __init__(self, mode: str):
        self.mode = mode
        self.mu = None
        self.sd = None
        self.lo = None
        self.hi = None

    def fit(self, x: np.ndarray) -> "ColumnScaler":
        self.mu = x.mean(axis=0)
        self.sd = x.std(axis=0)
        self.lo = x.min(axis=0)
        self.hi = x.max(axis=0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mode == "none":
            return np.array(x, copy=True)
        if self.mode == "guard":
            scale = np.where(self.sd < 1e-8, 1.0, self.sd)
            return (x - self.mu) / scale
        if self.mode == "flat_zero":
            flat = self.sd < 1e-8
            scale = np.where(flat, 1.0, self.sd)
            z = (x - self.mu) / scale
            z = np.array(z, copy=True)
            z[:, flat] = 0.0
            return z
        if self.mode == "minmax":
            span = self.hi - self.lo
            flat = span < 1e-8
            span = np.where(flat, 1.0, span)
            z = (x - self.lo) / span
            z = np.array(z, copy=True)
            z[:, flat] = 0.0
            return z
        raise ValueError(self.mode)


def scale_train_test(x_train, x_test, mode: str):
    scaler = ColumnScaler(mode).fit(x_train)
    return scaler.transform(x_train), scaler.transform(x_test)


def as_test(metrics: dict) -> dict:
    out = {}
    for key, value in metrics.items():
        if key.startswith("test_") or key in ("best_epoch", "epochs_ran", "best_val_log_loss"):
            out[key] = value
        else:
            out[f"test_{key}"] = value
    return out


def logistic_fold(x_train, y_train, x_test, y_test, seed: int, balanced: bool, mode: str):
    del seed
    xtr, xte = scale_train_test(x_train, x_test, mode)
    clf = LogisticRegression(
        C=1.0,
        class_weight="balanced" if balanced else None,
        solver="lbfgs",
        max_iter=500,
    )
    clf.fit(xtr, y_train)
    return as_test(score_metrics(y_test, positive_proba(clf, xte)))


def gbdt_fold(x_train, y_train, x_test, y_test, seed: int, balanced: bool, learning_rate: float):
    row, _ = fit_gbdt_clean(
        x_train,
        y_train,
        x_test,
        y_test,
        seed,
        n_estimators=40,
        learning_rate=learning_rate,
        balanced=balanced,
    )
    return {key: value for key, value in row.items() if key.startswith("test_") or key == "chosen_trees"}


def train_mlp_checkpoints(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    hidden: tuple[int, ...],
    seed: int,
    lr: float,
    balanced: bool,
    max_epochs: int = 80,
    patience: int = 8,
):
    """Train up to 80 epochs. Return the best validation-loss weights and the last weights."""
    torch.manual_seed(seed)
    model = MLP(x_train.shape[1], hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    x_t = torch.from_numpy(np.ascontiguousarray(x_train, dtype=np.float32))
    y_t = torch.from_numpy(np.ascontiguousarray(y_train, dtype=np.float32))
    if balanced:
        n_pos = max(float(y_train.sum()), 1.0)
        n_neg = max(float(len(y_train) - y_train.sum()), 1.0)
        w_pos = len(y_train) / (2.0 * n_pos)
        w_neg = len(y_train) / (2.0 * n_neg)
        weights = torch.where(y_t > 0.5, torch.full_like(y_t, w_pos), torch.full_like(y_t, w_neg))
    else:
        weights = torch.ones_like(y_t)
    rng = np.random.default_rng(seed)
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    best_ll = np.inf
    best_epoch = 1
    wait = 0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    epochs_ran = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = rng.permutation(len(x_train))
        for start in range(0, len(x_train), 32):
            idx = order[start : start + 32]
            opt.zero_grad()
            loss = (bce(model(x_t[idx]), y_t[idx]) * weights[idx]).mean()
            loss.backward()
            opt.step()
        epochs_ran = epoch
        p = np.clip(mlp_predict(model, x_val), 1e-7, 1 - 1e-7)
        ll = float(np.mean(-(y_val * np.log(p) + (1 - y_val) * np.log(1 - p))))
        if ll < best_ll:
            best_ll = ll
            best_epoch = epoch
            wait = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    last_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return best_state, last_state, best_epoch, epochs_ran, best_ll


def mlp_fold(
    x_train,
    y_train,
    x_test,
    y_test,
    seed: int,
    hidden: tuple[int, ...],
    lr: float,
    balanced: bool,
    mode: str,
):
    inner = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    inner_train, inner_val = next(inner.split(x_train, y_train))
    scaler = ColumnScaler(mode).fit(x_train[inner_train])
    xtr = scaler.transform(x_train[inner_train])
    xva = scaler.transform(x_train[inner_val])
    xte = scaler.transform(x_test)
    best_state, last_state, best_epoch, epochs_ran, best_val_ll = train_mlp_checkpoints(
        xtr,
        y_train[inner_train],
        xva,
        y_train[inner_val],
        hidden,
        seed,
        lr,
        balanced,
    )
    model = MLP(xtr.shape[1], hidden)
    model.load_state_dict(best_state)
    best = as_test(score_metrics(y_test, mlp_predict(model, xte)))
    model.load_state_dict(last_state)
    last = as_test(score_metrics(y_test, mlp_predict(model, xte)))
    best["best_epoch"] = float(best_epoch)
    best["epochs_ran"] = float(epochs_ran)
    best["best_val_log_loss"] = float(best_val_ll)
    last["best_epoch"] = float(best_epoch)
    last["epochs_ran"] = float(epochs_ran)
    return best, last


def run_folds(splits, x, y, trainer) -> dict:
    folds = []
    for seed, (train_idx, test_idx) in enumerate(splits):
        folds.append(trainer(x[train_idx], y[train_idx], x[test_idx], y[test_idx], seed))
    return pack("study", folds, "53")


def brief(result: dict) -> str:
    mean = result["mean"]
    std = result["std"]
    collapsed = result.get("n_collapsed_keep", 0)
    return (
        f"auc {mean['test_roc_auc']:.3f}±{std['test_roc_auc']:.3f}  "
        f"f1 {mean['test_f1_delete']:.3f}±{std['test_f1_delete']:.3f}  "
        f"P {mean['test_precision_delete']:.3f}  R {mean['test_recall_delete']:.3f}  "
        f"pred+ {mean['test_pred_delete_rate']:.3f}  "
        f"acc {mean['test_accuracy']:.3f}  "
        f"collapsed {collapsed}/{result['n_folds']}"
    )


def with_collapse(packed: dict) -> dict:
    recalls = [fold["test_recall_delete"] for fold in packed["folds"]]
    packed["n_collapsed_keep"] = int(sum(value < 0.05 for value in recalls))
    return packed


def main() -> None:
    data, _meta = load_cache(ROOT / "output")
    y = data["y"].astype(int)
    x59 = build_matrix(data, 640, 0.25)
    x53, names53 = take_columns(x59, FEATURE_NAMES, DROPPED)
    splits = make_splits(y, SEEDS)
    constant = [name for i, name in enumerate(names53) if np.std(x53[:, i]) < 1e-8]
    lines = [
        f"n={len(y)} delete_rate={y.mean():.3f} features={len(names53)} constant={constant}",
        "Test metrics. F1/precision/recall are for the delete class at threshold 0.5.",
        "AUC does not use a threshold. Always-keep has delete F1 = 0 and accuracy = 0.61.",
        "",
    ]

    def say(text: str = "") -> None:
        print(text)
        lines.append(text)

    naive = (x53 - x53.mean(0)) / x53.std(0)
    say(f"Unguarded z-score has non-finite values: {bool(~np.isfinite(naive).all())}")
    say("")

    # Baselines on the 53-d vector.
    base_folds = []
    keep_folds = []
    for _seed, (train_idx, test_idx) in enumerate(splits):
        ytr, yte = y[train_idx], y[test_idx]
        base_folds.append(as_test(score_metrics(yte, np.full(len(yte), float(ytr.mean())))))
        keep_folds.append(as_test(score_metrics(yte, np.zeros(len(yte)))))
    baselines = {
        "base_rate_probs": with_collapse(pack("base_rate", base_folds, "53")),
        "always_keep": with_collapse(pack("always_keep", keep_folds, "53")),
    }
    say("## Baselines")
    for name, result in baselines.items():
        say(f"{name:<22} {brief(result)}")
    say("")

    say("## Balance, plan settings, best MLP checkpoint")
    balance_rows = {}
    for balanced in (False, True):
        label = "balanced" if balanced else "unbalanced"
        logistic = with_collapse(
            run_folds(
                splits,
                x53,
                y,
                lambda xtr, ytr, xte, yte, seed, balanced=balanced: logistic_fold(
                    xtr, ytr, xte, yte, seed, balanced, "guard"
                ),
            )
        )
        gbdt = with_collapse(
            run_folds(
                splits,
                x53,
                y,
                lambda xtr, ytr, xte, yte, seed, balanced=balanced: gbdt_fold(
                    xtr, ytr, xte, yte, seed, balanced, 0.1
                ),
            )
        )
        balance_rows[f"logistic_{label}"] = logistic
        balance_rows[f"gbdt_{label}"] = gbdt
        say(f"logistic {label:<12} {brief(logistic)}")
        say(f"gbdt {label:<16} {brief(gbdt)}")
        for hidden, hidden_name in (((16,), "mlp_small"), ((64, 32), "mlp_big")):
            best_folds = []
            last_folds = []
            for seed, (train_idx, test_idx) in enumerate(splits):
                best, last = mlp_fold(
                    x53[train_idx],
                    y[train_idx],
                    x53[test_idx],
                    y[test_idx],
                    seed,
                    hidden,
                    1e-3,
                    balanced,
                    "guard",
                )
                best_folds.append(best)
                last_folds.append(last)
            best_packed = with_collapse(pack(hidden_name, best_folds, "53"))
            last_packed = with_collapse(pack(hidden_name, last_folds, "53"))
            balance_rows[f"{hidden_name}_{label}_best"] = best_packed
            balance_rows[f"{hidden_name}_{label}_last"] = last_packed
            say(f"{hidden_name} {label:<12} best {brief(best_packed)}")
            say(f"{hidden_name} {label:<12} last {brief(last_packed)}")
    say("")

    say("## MLP learning rate, best checkpoint vs last epoch")
    lr_rows = {}
    for lr in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2):
        for balanced in (False, True):
            label = "balanced" if balanced else "unbalanced"
            for hidden, hidden_name in (((16,), "mlp_small"), ((64, 32), "mlp_big")):
                best_folds = []
                last_folds = []
                for seed, (train_idx, test_idx) in enumerate(splits):
                    best, last = mlp_fold(
                        x53[train_idx],
                        y[train_idx],
                        x53[test_idx],
                        y[test_idx],
                        seed,
                        hidden,
                        lr,
                        balanced,
                        "guard",
                    )
                    best_folds.append(best)
                    last_folds.append(last)
                key = f"{hidden_name}_lr{lr:g}_{label}"
                lr_rows[key] = {
                    "best": with_collapse(pack(key, best_folds, "53")),
                    "last": with_collapse(pack(key, last_folds, "53")),
                }
                say(f"{key:<32} best {brief(lr_rows[key]['best'])}")
                say(f"{'':<32} last {brief(lr_rows[key]['last'])}")
    say("")

    say("## GBDT learning rate")
    gbdt_lr = {}
    for lr in (0.05, 0.1, 0.3):
        for balanced in (False, True):
            label = "balanced" if balanced else "unbalanced"
            result = with_collapse(
                run_folds(
                    splits,
                    x53,
                    y,
                    lambda xtr, ytr, xte, yte, seed, lr=lr, balanced=balanced: gbdt_fold(
                        xtr, ytr, xte, yte, seed, balanced, lr
                    ),
                )
            )
            gbdt_lr[f"gbdt_lr{lr:g}_{label}"] = result
            say(f"gbdt lr={lr:<5} {label:<12} {brief(result)}")
    say("")

    say("## Constant columns and scaling")
    # Dropping a column that is always 0 cannot change a model that already zeros it.
    text_and_wa = ["log_text_count", "text_area_clipped", "whatsapp"]
    x_drop_const, _ = take_columns(x53, names53, tuple(text_and_wa))
    scale_rows = {}
    for mode in ("guard", "flat_zero", "minmax", "none"):
        for balanced in (False, True):
            label = "balanced" if balanced else "unbalanced"
            logistic = with_collapse(
                run_folds(
                    splits,
                    x53,
                    y,
                    lambda xtr, ytr, xte, yte, seed, mode=mode, balanced=balanced: logistic_fold(
                        xtr, ytr, xte, yte, seed, balanced, mode
                    ),
                )
            )
            scale_rows[f"logistic_{mode}_{label}"] = logistic
            say(f"logistic {mode:<10} {label:<12} {brief(logistic)}")
            for matrix, tag in ((x53, "with_constants"), (x_drop_const, "drop_constants")):
                if mode != "guard":
                    continue
                result = with_collapse(
                    run_folds(
                        splits,
                        matrix,
                        y,
                        lambda xtr, ytr, xte, yte, seed, balanced=balanced: logistic_fold(
                            xtr, ytr, xte, yte, seed, balanced, "guard"
                        ),
                    )
                )
                scale_rows[f"logistic_{tag}_{label}"] = result
                say(f"logistic {tag:<16} {label:<12} {brief(result)}")
        for hidden, hidden_name in (((16,), "mlp_small"), ((64, 32), "mlp_big")):
            best_folds = []
            for seed, (train_idx, test_idx) in enumerate(splits):
                best, _last = mlp_fold(
                    x53[train_idx],
                    y[train_idx],
                    x53[test_idx],
                    y[test_idx],
                    seed,
                    hidden,
                    1e-3,
                    True,
                    mode,
                )
                best_folds.append(best)
            packed = with_collapse(pack(hidden_name, best_folds, "53"))
            scale_rows[f"{hidden_name}_{mode}_balanced_best"] = packed
            say(f"{hidden_name} {mode:<10} balanced     best {brief(packed)}")
    say("")

    say("## GPS column, same splits, guarded logistic and clean GBDT")
    gps_rows = {}
    for matrix, tag in ((x53, "no_gps"), (x59, "with_gps")):
        logistic = with_collapse(
            run_folds(
                splits,
                matrix,
                y,
                lambda xtr, ytr, xte, yte, seed: logistic_fold(xtr, ytr, xte, yte, seed, True, "guard"),
            )
        )
        gbdt = with_collapse(
            run_folds(
                splits,
                matrix,
                y,
                lambda xtr, ytr, xte, yte, seed: gbdt_fold(xtr, ytr, xte, yte, seed, False, 0.1),
            )
        )
        gps_rows[f"logistic_{tag}"] = logistic
        gps_rows[f"gbdt_{tag}"] = gbdt
        say(f"logistic {tag:<10} {brief(logistic)}")
        say(f"gbdt     {tag:<10} {brief(gbdt)}")

    payload = {
        "n_features": len(names53),
        "feature_names": names53,
        "constant_features": constant,
        "unguarded_zscore_nonfinite": bool(~np.isfinite(naive).all()),
        "baselines": baselines,
        "balance": balance_rows,
        "mlp_lr": {key: value for key, value in lr_rows.items()},
        "gbdt_lr": gbdt_lr,
        "scaling": scale_rows,
        "gps": gps_rows,
    }
    out = ROOT / "output" / "training_study.json"
    out.write_text(json.dumps(payload, indent=2))
    report = ROOT / "output" / "training_study.md"
    report.write_text("\n".join(lines) + "\n")
    say("")
    say(f"wrote {out}")


if __name__ == "__main__":
    main()
