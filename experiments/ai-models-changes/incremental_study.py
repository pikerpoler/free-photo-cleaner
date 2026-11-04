"""Two swipe sessions, with the first session's photos gone before the second.

Session 1 is 200 labeled photos. The model trains, and those rows are then
discarded. Session 2 is 200 new labeled photos. The update sees the saved
model and the new rows only. The same 200 held-out photos are scored after
session 1 and after the update.

A single decision tree cannot do this: ``DecisionTreeClassifier`` has no
update, and a second ``fit`` replaces the tree. Gradient boosting can.
``warm_start`` keeps the session-1 trees and appends trees fit on session 2.
Logistic regression also has no ``partial_fit``. The session-1 fit stores the
weights and the Hessian of its objective (54 by 54, intercept included).
Session 2 minimizes its own loss plus that quadratic, which is the local
shape of the session-1 objective, L2 included. The MLP keeps its weights and
keeps training. The session-1 scaler stays frozen, because a new scale would
change what those weights mean.

Replay fits the same model from scratch on both sessions. That is the score
you get if the deleted photos' feature rows were kept. Session-2-only fits
from scratch on the second session, which is the score if the first model
was thrown away.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedShuffleSplit

from evaluate_models import (
    MLP,
    build_matrix,
    fit_gbdt_clean,
    load_cache,
    mlp_predict,
    pack,
    positive_proba,
    score_metrics,
)
from photo_features import FEATURE_NAMES
from training_study import ColumnScaler, take_columns

ROOT = Path(__file__).resolve().parent
DROPPED = ("live_photo", "portrait", "panorama", "hdr", "burst", "has_gps")
SEEDS = 10
TEST_N = 200
SESSION_N = 200
MAX_TREES = 40
C_L2 = 1.0


def balanced_weights(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y).astype(int)
    n = len(y)
    n_pos = max(float(y.sum()), 1.0)
    n_neg = max(float(n - y.sum()), 1.0)
    weights = np.empty(n, dtype=np.float64)
    weights[y == 1] = n / (2.0 * n_pos)
    weights[y == 0] = n / (2.0 * n_neg)
    return weights


def split_sessions(y: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified test set, then two stratified sessions, from what remains."""
    pools = [rng.permutation(np.flatnonzero(y == cls)) for cls in (0, 1)]

    def draw(n: int) -> np.ndarray:
        total = sum(len(pool) for pool in pools)
        n1 = int(round(n * len(pools[1]) / total))
        n1 = min(max(n1, 2), len(pools[1]), n - 2)
        n0 = n - n1
        if n0 > len(pools[0]):
            raise RuntimeError(f"need {n0} keeps, have {len(pools[0])}")
        take = np.concatenate([pools[0][:n0], pools[1][:n1]])
        pools[0] = pools[0][n0:]
        pools[1] = pools[1][n1:]
        return rng.permutation(take)

    test = draw(TEST_N)
    day1 = draw(SESSION_N)
    day2 = draw(SESSION_N)
    return day1, day2, test


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def pack_theta(coef: np.ndarray, intercept: float) -> np.ndarray:
    return np.concatenate([np.asarray(coef, dtype=np.float64).ravel(), [float(intercept)]])


def logistic_proba(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return sigmoid(x @ theta[:-1] + theta[-1])


def logistic_hessian(x: np.ndarray, y: np.ndarray, theta: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Hessian of ``C * sum(weight * log loss) + 0.5 ||weights||^2``.

    The identity from the L2 term is added on the coefficients only.
    """
    p = logistic_proba(x, theta)
    curv = weights * p * (1.0 - p)
    aug = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    hess = C_L2 * (aug.T @ (aug * curv[:, None]))
    hess[:-1, :-1] += np.eye(x.shape[1])
    hess = 0.5 * (hess + hess.T)
    eig = float(np.linalg.eigvalsh(hess).min())
    if eig < 1e-8:
        hess = hess + (1e-8 - eig) * np.eye(hess.shape[0])
    return hess


def session1_gradient(x: np.ndarray, y: np.ndarray, theta: np.ndarray, weights: np.ndarray) -> np.ndarray:
    resid = weights * (logistic_proba(x, theta) - y)
    aug = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    grad = C_L2 * (aug.T @ resid)
    grad[:-1] += theta[:-1]
    return grad


def fit_logistic_batch(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, ColumnScaler]:
    scaler = ColumnScaler("guard").fit(x)
    z = scaler.transform(x)
    weights = balanced_weights(y)
    clf = LogisticRegression(C=C_L2, solver="lbfgs", max_iter=500)
    clf.fit(z, y, sample_weight=weights)
    return pack_theta(clf.coef_, float(clf.intercept_[0])), scaler


def update_logistic(x: np.ndarray, y: np.ndarray, theta1: np.ndarray, hess: np.ndarray) -> np.ndarray:
    """Minimize session-2 loss plus the quadratic memory of session 1."""
    weights = balanced_weights(y)
    aug = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    y = np.asarray(y, dtype=np.float64)

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        z = aug @ theta
        loss = np.where(z >= 0.0, z + np.log1p(np.exp(-np.clip(z, 0.0, None))), np.log1p(np.exp(np.clip(z, None, 0.0))))
        loss = loss - y * z
        diff = theta - theta1
        value = float(C_L2 * np.sum(weights * loss) + 0.5 * diff @ hess @ diff)
        resid = C_L2 * weights * (sigmoid(z) - y)
        grad = aug.T @ resid + hess @ diff
        return value, grad

    start, _ = objective(theta1)
    result = minimize(objective, theta1, method="L-BFGS-B", jac=True, options={"maxiter": 500, "ftol": 1e-12})
    if not result.success or not np.isfinite(result.fun) or result.fun > start + 1e-4:
        return theta1.copy()
    return np.asarray(result.x, dtype=np.float64)


def gbdt_params(seed: int, n_estimators: int) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=3,
        min_samples_leaf=8,
        learning_rate=0.1,
        loss="log_loss",
        random_state=seed,
    )


def fit_gbdt_session(x_train, y_train, x_test, y_test, seed: int) -> tuple[dict, GradientBoostingClassifier]:
    return fit_gbdt_clean(
        x_train,
        y_train,
        x_test,
        y_test,
        seed,
        n_estimators=MAX_TREES,
        learning_rate=0.1,
        balanced=True,
    )


def append_trees(model: GradientBoostingClassifier, x: np.ndarray, y: np.ndarray, n_new: int, weights: np.ndarray):
    if n_new <= 0:
        return model
    grown = copy.deepcopy(model)
    grown.set_params(warm_start=True, n_estimators=int(model.n_estimators_) + int(n_new))
    grown.fit(x, y, sample_weight=weights)
    return grown


def choose_extra_trees(
    model: GradientBoostingClassifier,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    weights: np.ndarray,
) -> int:
    """How many new trees help session-2 validation loss. Zero is allowed."""
    base = int(model.n_estimators_)
    probe = append_trees(model, x_train, y_train, MAX_TREES, weights)
    best_extra = 0
    best_ll = log_loss(y_val, np.clip(positive_proba(model, x_val), 1e-7, 1 - 1e-7), labels=[0, 1])
    for extra, proba in enumerate(probe.staged_predict_proba(x_val), start=1):
        if extra <= base:
            continue
        ll = log_loss(y_val, np.clip(proba[:, 1], 1e-7, 1 - 1e-7), labels=[0, 1])
        if ll < best_ll - 1e-6:
            best_ll = ll
            best_extra = extra - base
    return best_extra


def train_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray | None,
    y_val: np.ndarray | None,
    hidden: tuple[int, ...],
    seed: int,
    state: dict | None,
    lr: float = 1e-3,
    max_epochs: int = 80,
) -> tuple[dict, int, int]:
    """Adam, balanced loss, patience 8. ``state`` continues a saved network.

    With a saved network, epoch 0 is the incoming weights, and it wins if
    another pass does not improve the session-2 validation loss. With no
    validation slice, the run goes for ``max_epochs`` and keeps the last weights.
    """
    torch.manual_seed(seed)
    model = MLP(x_train.shape[1], hidden)
    if state is not None:
        model.load_state_dict(state)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    x_t = torch.from_numpy(np.ascontiguousarray(x_train, dtype=np.float32))
    y_t = torch.from_numpy(np.ascontiguousarray(y_train, dtype=np.float32))
    if y_val is not None:
        y_val = np.asarray(y_val, dtype=np.float64)
    weights_np = balanced_weights(y_train)
    weights = torch.from_numpy(np.ascontiguousarray(weights_np, dtype=np.float32))
    rng = np.random.default_rng(seed)
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")

    def current_ll() -> float:
        p = np.clip(mlp_predict(model, x_val), 1e-7, 1 - 1e-7)
        return float(np.mean(-(y_val * np.log(p) + (1.0 - y_val) * np.log(1.0 - p))))

    if x_val is None:
        best_ll = np.inf
        best_epoch = max_epochs
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    elif state is None:
        best_ll = np.inf
        best_epoch = 1
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    else:
        best_ll = current_ll()
        best_epoch = 0
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    wait = 0
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
        if x_val is None:
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            continue
        ll = current_ll()
        if ll < best_ll:
            best_ll = ll
            best_epoch = epoch
            wait = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= 8:
                break
    return best_state, best_epoch, epochs_ran


def inner_indices(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, val_idx = next(splitter.split(np.zeros(len(y)), y))
    return train_idx, val_idx


def run_mlp_condition(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    hidden: tuple[int, ...],
    seed: int,
    state: dict | None = None,
    scaler: ColumnScaler | None = None,
) -> tuple[dict, dict, ColumnScaler, np.ndarray]:
    tr, va = inner_indices(y_train, seed)
    if scaler is None:
        scaler = ColumnScaler("guard").fit(x_train[tr])
    xtr = scaler.transform(x_train[tr])
    xva = scaler.transform(x_train[va])
    xte = scaler.transform(x_test)
    best_state, best_epoch, epochs_ran = train_mlp(xtr, y_train[tr], xva, y_train[va], hidden, seed, state)
    model = MLP(xtr.shape[1], hidden)
    model.load_state_dict(best_state)
    proba = mlp_predict(model, xte)
    row = score_metrics(y_test, proba)
    row["best_epoch"] = float(best_epoch)
    row["epochs_ran"] = float(epochs_ran)
    return row, best_state, scaler, proba


def row_from_proba(y: np.ndarray, p: np.ndarray, extra: dict | None = None) -> dict:
    row = score_metrics(y, p)
    if extra:
        row.update(extra)
    return row


def brief(result: dict) -> str:
    mean = result["mean"]
    std = result["std"]
    recalls = [fold["recall_delete"] for fold in result["folds"]]
    collapsed = sum(value < 0.05 for value in recalls)
    return (
        f"auc {mean['roc_auc']:.3f}±{std['roc_auc']:.3f}  "
        f"f1 {mean['f1_delete']:.3f}±{std['f1_delete']:.3f}  "
        f"R {mean['recall_delete']:.3f}  "
        f"pred+ {mean['pred_delete_rate']:.3f}  "
        f"p@20 {mean['precision_at_20']:.3f}  "
        f"ll {mean['log_loss']:.3f}  "
        f"collapsed {collapsed}/{result['n_folds']}"
    )


def main() -> None:
    torch.set_num_threads(4)
    data, _meta = load_cache(ROOT / "output")
    y = data["y"].astype(int)
    x59 = build_matrix(data, 640, 0.25)
    x, names = take_columns(x59, FEATURE_NAMES, DROPPED)
    hidden_sets = {"mlp_small": (16,), "mlp_big": (64, 32)}
    conditions = ("day1", "incremental", "day2_only", "replay")
    model_names = ("logistic", "gbdt", "mlp_small", "mlp_big")
    folds: dict[tuple[str, str], list[dict]] = {(model, cond): [] for model in model_names for cond in conditions}
    slice_rates = []

    for seed in range(SEEDS):
        rng = np.random.default_rng(seed)
        i1, i2, i_test = split_sessions(y, rng)
        x1, y1 = x[i1], y[i1]
        x2, y2 = x[i2], y[i2]
        x_test, y_test = x[i_test], y[i_test]
        x_both = np.concatenate([x1, x2])
        y_both = np.concatenate([y1, y2])
        slice_rates.append(
            {
                "day1": float(y1.mean()),
                "day2": float(y2.mean()),
                "test": float(y_test.mean()),
            }
        )

        theta1, scaler1 = fit_logistic_batch(x1, y1)
        weights1 = balanced_weights(y1)
        grad = session1_gradient(scaler1.transform(x1), y1, theta1, weights1)
        hess = logistic_hessian(scaler1.transform(x1), y1, theta1, weights1)
        if seed == 0 and float(np.max(np.abs(grad))) > 0.5:
            raise RuntimeError(f"session-1 logistic gradient is {np.max(np.abs(grad)):.3f}, objective does not match")
        z_test = scaler1.transform(x_test)
        p_day1 = logistic_proba(z_test, theta1)
        theta2 = update_logistic(scaler1.transform(x2), y2, theta1, hess)
        p_inc = logistic_proba(z_test, theta2)
        theta_only, scaler_only = fit_logistic_batch(x2, y2)
        theta_replay, scaler_replay = fit_logistic_batch(x_both, y_both)
        step = float(np.linalg.norm(theta2 - theta1))
        folds[("logistic", "day1")].append(row_from_proba(y_test, p_day1, {"grad_max": float(np.max(np.abs(grad)))}))
        folds[("logistic", "incremental")].append(
            row_from_proba(y_test, p_inc, {"weight_step": step, "mean_abs_dp": float(np.mean(np.abs(p_inc - p_day1)))})
        )
        folds[("logistic", "day2_only")].append(
            row_from_proba(y_test, logistic_proba(scaler_only.transform(x_test), theta_only))
        )
        folds[("logistic", "replay")].append(
            row_from_proba(y_test, logistic_proba(scaler_replay.transform(x_test), theta_replay))
        )

        _row, gbdt1 = fit_gbdt_session(x1, y1, x_test, y_test, seed)
        p_gbdt1 = positive_proba(gbdt1, x_test)
        tr, va = inner_indices(y2, seed + 100)
        weights2 = balanced_weights(y2)
        extra = choose_extra_trees(gbdt1, x2[tr], y2[tr], x2[va], y2[va], weights2[tr])
        gbdt_inc = append_trees(gbdt1, x2, y2, extra, weights2)
        p_gbdt_inc = positive_proba(gbdt_inc, x_test)
        if extra > 0 and float(np.max(np.abs(p_gbdt_inc - p_gbdt1))) < 1e-8:
            raise RuntimeError("added trees did not change scores")
        _row2, gbdt_only = fit_gbdt_session(x2, y2, x_test, y_test, seed + 100)
        _row3, gbdt_replay = fit_gbdt_session(x_both, y_both, x_test, y_test, seed + 200)
        tree_extra = {"day1_trees": float(gbdt1.n_estimators_), "extra_trees": float(extra)}
        folds[("gbdt", "day1")].append(row_from_proba(y_test, p_gbdt1, {"day1_trees": float(gbdt1.n_estimators_)}))
        folds[("gbdt", "incremental")].append(
            row_from_proba(
                y_test,
                p_gbdt_inc,
                {**tree_extra, "mean_abs_dp": float(np.mean(np.abs(p_gbdt_inc - p_gbdt1)))},
            )
        )
        folds[("gbdt", "day2_only")].append(row_from_proba(y_test, positive_proba(gbdt_only, x_test)))
        folds[("gbdt", "replay")].append(row_from_proba(y_test, positive_proba(gbdt_replay, x_test)))

        for name, hidden in hidden_sets.items():
            day1_row, state1, scaler_m, p_mlp1 = run_mlp_condition(x1, y1, x_test, y_test, hidden, seed)
            inc_row, _state2, _scaler, p_mlp_inc = run_mlp_condition(
                x2, y2, x_test, y_test, hidden, seed + 100, state=state1, scaler=scaler_m
            )
            only_row, _, _, _ = run_mlp_condition(x2, y2, x_test, y_test, hidden, seed + 100)
            replay_row, _, _, _ = run_mlp_condition(x_both, y_both, x_test, y_test, hidden, seed + 200)
            inc_row["mean_abs_dp"] = float(np.mean(np.abs(p_mlp_inc - p_mlp1)))
            folds[(name, "day1")].append(day1_row)
            folds[(name, "incremental")].append(inc_row)
            folds[(name, "day2_only")].append(only_row)
            folds[(name, "replay")].append(replay_row)

        print(
            f"seed {seed}  logistic inc auc {folds[('logistic', 'incremental')][-1]['roc_auc']:.3f}  "
            f"step {step:.3f}  gbdt extra {extra:.0f}  "
            f"gbdt inc auc {folds[('gbdt', 'incremental')][-1]['roc_auc']:.3f}  "
            f"mlp extra epochs {folds[('mlp_small', 'incremental')][-1]['best_epoch']:.0f}",
            flush=True,
        )

    lines = [
        f"n={len(y)} features={len(names)} delete_rate={y.mean():.3f}",
        f"Each seed: stratified test {TEST_N}, session 1 {SESSION_N}, session 2 {SESSION_N}, unused {len(y) - TEST_N - 2 * SESSION_N}.",
        "Sessions are random draws from the labeled set, not a later calendar day.",
        "Same test photos after session 1 and after the session-2 update.",
        "Balanced class weights. Logistic and the MLP use the guarded scaler.",
        "Trees and the MLP pick the round on an inner 20% of the session they are fitting. Logistic uses every row in the session.",
        "Incremental logistic freezes the session-1 scaler and adds the session-1 Hessian as a quadratic penalty.",
        "Incremental trees keep the session-1 trees and may append up to 40 trees fit on session 2, including zero.",
        "Incremental MLP starts from the session-1 best validation checkpoint, freezes that scaler, and may stop at epoch 0.",
        "",
        f"Mean delete rate  day1 {np.mean([r['day1'] for r in slice_rates]):.3f}  "
        f"day2 {np.mean([r['day2'] for r in slice_rates]):.3f}  "
        f"test {np.mean([r['test'] for r in slice_rates]):.3f}",
        "",
        "AUC ranks P(delete). F1, recall, and pred+ are the delete class at 0.5. p@20 is precision in the top 20% of scores.",
        "",
    ]

    def say(text: str = "") -> None:
        print(text, flush=True)
        lines.append(text)

    packed = {}
    say(f"{'model':<12} {'condition':<14} result")
    for model in model_names:
        for cond in conditions:
            result = pack(f"{model}_{cond}", folds[(model, cond)], "53")
            packed[f"{model}_{cond}"] = result
            say(f"{model:<12} {cond:<14} {brief(result)}")
        say("")

    say("## Update size")
    inc_log = packed["logistic_incremental"]["mean"]
    inc_gbdt = packed["gbdt_incremental"]["mean"]
    say(
        f"logistic weight step L2 {inc_log['weight_step']:.3f}  "
        f"mean |dp| {inc_log['mean_abs_dp']:.3f}  "
        f"session-1 grad max {packed['logistic_day1']['mean']['grad_max']:.4f}"
    )
    say(
        f"gbdt session-1 trees {inc_gbdt['day1_trees']:.1f}  "
        f"extra trees {inc_gbdt['extra_trees']:.1f}  "
        f"mean |dp| {inc_gbdt['mean_abs_dp']:.3f}"
    )
    for model in ("mlp_small", "mlp_big"):
        mean = packed[f"{model}_incremental"]["mean"]
        say(
            f"{model} kept-epoch {mean['best_epoch']:.1f} of the continuation "
            f"(0 means the session-1 weights won)  epochs run {mean['epochs_ran']:.1f}  "
            f"mean |dp| {mean['mean_abs_dp']:.3f}"
        )

    payload = {
        "protocol": {
            "n": int(len(y)),
            "n_features": len(names),
            "test_n": TEST_N,
            "session_n": SESSION_N,
            "seeds": SEEDS,
            "feature_names": names,
            "balanced": True,
            "scaler": "guard, frozen after session 1 for incremental logistic and MLP",
            "logistic": "lbfgs C=1, then session-2 loss plus session-1 Hessian",
            "gbdt": "warm_start appends up to 40 trees chosen on an inner slice of session 2",
            "mlp": "Adam 1e-3, weight decay 1e-3, patience 8, best session validation loss, epoch 0 allowed",
        },
        "slice_delete_rates": slice_rates,
        "results": packed,
    }
    out_json = ROOT / "output" / "incremental_study.json"
    out_md = ROOT / "output" / "incremental_study.md"
    out_json.write_text(json.dumps(payload, indent=2))
    out_md.write_text("\n".join(lines) + "\n")
    say("")
    say(f"wrote {out_json}")


if __name__ == "__main__":
    main()
