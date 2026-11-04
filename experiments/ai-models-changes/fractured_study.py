"""Two schedules for the same 400 labeled photos, scored on the same 200 held out.

``two_session`` trains on 200, discards those rows, then updates on the other 200.
``fractured`` updates through 20, 20, 50, 30, 70, 50, 160. ``replay`` fits once
on all 400, which is what you get if the feature rows were kept.

River's online logistic regression is the same idea as ``SGDClassifier``, so
only the sklearn one is here. XGBoost is the same idea as LightGBM's continued
boosting, so only LightGBM is here. The wide net is 53→128→32→1.
"""

from __future__ import annotations

import os

# LightGBM and PyTorch both ship OpenMP. One thread avoids a hang on macOS.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import copy
import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import torch
from catboost import CatBoostClassifier
from river.forest import ARFClassifier
from river.tree import HoeffdingTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
from sklearn.naive_bayes import GaussianNB

from evaluate_models import (
    MLP,
    build_matrix,
    gbdt_model,
    load_cache,
    mlp_predict,
    pack,
    positive_proba,
    score_metrics,
    staged_best_k,
)
from incremental_study import (
    C_L2,
    MAX_TREES,
    append_trees,
    balanced_weights,
    brief,
    choose_extra_trees,
    fit_logistic_batch,
    inner_indices,
    logistic_proba,
    train_mlp,
    update_logistic,
)
from photo_features import FEATURE_NAMES
from training_study import ColumnScaler, take_columns

ROOT = Path(__file__).resolve().parent
DROPPED = ("live_photo", "portrait", "panorama", "hdr", "burst", "has_gps")
TEST_N = 200
POOL_N = 400
TWO = (200, 200)
FRACTURED = (20, 20, 50, 30, 70, 50, 160)
PROTOCOLS = ("two_session", "fractured", "replay")


def class_pools(indices: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    pools = []
    for cls in (0, 1):
        pool = indices[y[indices] == cls]
        pools.append(rng.permutation(pool))
    return pools


def draw_from_pools(pools: list[np.ndarray], n: int, rng: np.random.Generator) -> np.ndarray:
    total = sum(len(pool) for pool in pools)
    if n == total:
        take = np.concatenate(pools) if total else np.array([], dtype=int)
        pools[0] = pools[0][:0]
        pools[1] = pools[1][:0]
        return rng.permutation(take)
    n1 = int(round(n * len(pools[1]) / total))
    n1 = min(max(n1, 1), len(pools[1]), n - 1)
    n0 = n - n1
    if n0 > len(pools[0]):
        n0 = len(pools[0])
        n1 = n - n0
    if n1 > len(pools[1]) or n0 + n1 != n:
        raise RuntimeError(f"cannot draw {n} from {[len(pool) for pool in pools]}")
    take = np.concatenate([pools[0][:n0], pools[1][:n1]])
    pools[0] = pools[0][n0:]
    pools[1] = pools[1][n1:]
    return rng.permutation(take)


def make_split(y: np.ndarray, rng: np.random.Generator):
    pools = class_pools(np.arange(len(y)), y, rng)
    test = draw_from_pools(pools, TEST_N, rng)
    pool = draw_from_pools(pools, POOL_N, rng)

    def chunks(sizes: tuple[int, ...]) -> list[np.ndarray]:
        local = class_pools(pool, y, rng)
        return [draw_from_pools(local, n, rng) for n in sizes]

    return test, chunks(TWO), chunks(FRACTURED)


def try_inner(y: np.ndarray, seed: int):
    y = np.asarray(y).astype(int)
    if len(y) < 10 or int(np.min(np.bincount(y, minlength=2))) < 2:
        return None
    try:
        train_idx, val_idx = inner_indices(y, seed)
    except ValueError:
        return None
    if int(np.min(np.bincount(y[train_idx], minlength=2))) < 1:
        return None
    if int(np.min(np.bincount(y[val_idx], minlength=2))) < 1:
        return None
    return train_idx, val_idx


def data_curvature(x: np.ndarray, y: np.ndarray, theta: np.ndarray, weights: np.ndarray) -> np.ndarray:
    p = logistic_proba(x, theta)
    curv = weights * p * (1.0 - p)
    aug = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    hess = C_L2 * (aug.T @ (aug * curv[:, None]))
    return 0.5 * (hess + hess.T)


def positive_column(model, x: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(x)
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(x))
    return proba[:, classes.index(1)]


class LogisticMemory:
    """Batch logistic regression, then each new batch plus the stored Hessian."""

    def __init__(self):
        self.theta = None
        self.scaler = None
        self.hess = None

    def update(self, x: np.ndarray, y: np.ndarray, seed: int) -> dict:
        del seed
        y = np.asarray(y).astype(int)
        if self.theta is None:
            self.theta, self.scaler = fit_logistic_batch(x, y)
            z = self.scaler.transform(x)
            hess = data_curvature(z, y, self.theta, balanced_weights(y))
            hess[:-1, :-1] += np.eye(x.shape[1])
            self.hess = hess
            return {"accepted": 1.0}
        z = self.scaler.transform(x)
        updated = update_logistic(z, y, self.theta, self.hess)
        accepted = float(np.linalg.norm(updated - self.theta) > 1e-8)
        self.theta = updated
        self.hess = self.hess + data_curvature(z, y, self.theta, balanced_weights(y))
        return {"accepted": accepted}

    def predict(self, x: np.ndarray) -> np.ndarray:
        return logistic_proba(self.scaler.transform(x), self.theta)


class SklearnGBDT:
    def __init__(self):
        self.model = None

    def update(self, x: np.ndarray, y: np.ndarray, seed: int) -> dict:
        y = np.asarray(y).astype(int)
        weights = balanced_weights(y)
        split = try_inner(y, seed)
        if self.model is None:
            if split is None:
                self.model = gbdt_model(seed, 10)
                self.model.fit(x, y, sample_weight=weights)
                return {"extra_trees": 10.0}
            train_idx, val_idx = split
            selector = gbdt_model(seed, MAX_TREES)
            selector.fit(x[train_idx], y[train_idx], sample_weight=weights[train_idx])
            chosen, _ = staged_best_k(selector, x[val_idx], y[val_idx])
            self.model = gbdt_model(seed, chosen)
            self.model.fit(x, y, sample_weight=weights)
            return {"extra_trees": float(chosen)}
        if split is None:
            return {"extra_trees": 0.0}
        train_idx, val_idx = split
        extra = choose_extra_trees(self.model, x[train_idx], y[train_idx], x[val_idx], y[val_idx], weights[train_idx])
        self.model = append_trees(self.model, x, y, extra, weights)
        return {"extra_trees": float(extra)}

    def predict(self, x: np.ndarray) -> np.ndarray:
        return positive_proba(self.model, x)


class TorchMLP:
    def __init__(self, hidden: tuple[int, ...], lr: float):
        self.hidden = hidden
        self.lr = lr
        self.state = None
        self.scaler = None

    def update(self, x: np.ndarray, y: np.ndarray, seed: int) -> dict:
        y = np.asarray(y).astype(int)
        split = try_inner(y, seed)
        if self.scaler is None:
            fit_rows = x if split is None else x[split[0]]
            self.scaler = ColumnScaler("guard").fit(fit_rows)
        if split is None:
            state, best_epoch, epochs_ran = train_mlp(
                self.scaler.transform(x), y, None, None, self.hidden, seed, self.state, self.lr, 20
            )
        else:
            train_idx, val_idx = split
            state, best_epoch, epochs_ran = train_mlp(
                self.scaler.transform(x[train_idx]),
                y[train_idx],
                self.scaler.transform(x[val_idx]),
                y[val_idx],
                self.hidden,
                seed,
                self.state,
                self.lr,
            )
        self.state = state
        return {"best_epoch": float(best_epoch), "epochs_ran": float(epochs_ran)}

    def predict(self, x: np.ndarray) -> np.ndarray:
        model = MLP(x.shape[1], self.hidden)
        model.load_state_dict(self.state)
        return mlp_predict(model, self.scaler.transform(x))


class SGDLogistic:
    def __init__(self):
        self.scaler = None
        self.clf = None

    def _proba(self, z: np.ndarray) -> np.ndarray:
        if self.clf is None or not hasattr(self.clf, "coef_"):
            return np.full(len(z), 0.5)
        return positive_column(self.clf, z)

    def update(self, x: np.ndarray, y: np.ndarray, seed: int) -> dict:
        y = np.asarray(y).astype(int)
        if self.scaler is None:
            self.scaler = ColumnScaler("guard").fit(x)
        z = self.scaler.transform(x)
        weights = balanced_weights(y)
        if self.clf is None:
            self.clf = SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=1e-4,
                learning_rate="adaptive",
                eta0=0.05,
                random_state=seed,
                tol=None,
            )
        split = try_inner(y, seed)
        rng = np.random.default_rng(seed)
        if split is None:
            for _ in range(15):
                order = rng.permutation(len(y))
                self.clf.partial_fit(z[order], y[order], classes=np.array([0, 1]), sample_weight=weights[order])
            return {"best_epoch": 15.0, "epochs_ran": 15.0}
        train_idx, val_idx = split
        fresh = not hasattr(self.clf, "coef_")
        if fresh:
            best_ll = np.inf
            best_epoch = 1
            best_clf = None
        else:
            best_ll = log_loss(y[val_idx], np.clip(self._proba(z[val_idx]), 1e-7, 1 - 1e-7), labels=[0, 1])
            best_epoch = 0
            best_clf = copy.deepcopy(self.clf)
        wait = 0
        epochs_ran = 0
        for epoch in range(1, 81):
            order = rng.permutation(len(train_idx))
            idx = train_idx[order]
            self.clf.partial_fit(z[idx], y[idx], classes=np.array([0, 1]), sample_weight=weights[idx])
            epochs_ran = epoch
            ll = log_loss(y[val_idx], np.clip(self._proba(z[val_idx]), 1e-7, 1 - 1e-7), labels=[0, 1])
            if ll < best_ll:
                best_ll = ll
                best_epoch = epoch
                wait = 0
                best_clf = copy.deepcopy(self.clf)
            else:
                wait += 1
                if wait >= 8:
                    break
        if best_clf is not None:
            self.clf = best_clf
        return {"best_epoch": float(best_epoch), "epochs_ran": float(epochs_ran)}

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._proba(self.scaler.transform(x))


class IncrementalNB:
    def __init__(self):
        self.scaler = None
        self.nb = GaussianNB()

    def update(self, x: np.ndarray, y: np.ndarray, seed: int) -> dict:
        del seed
        y = np.asarray(y).astype(int)
        if self.scaler is None:
            self.scaler = ColumnScaler("guard").fit(x)
        self.nb.partial_fit(
            self.scaler.transform(x),
            y,
            classes=np.array([0, 1]),
            sample_weight=balanced_weights(y),
        )
        return {}

    def predict(self, x: np.ndarray) -> np.ndarray:
        return positive_column(self.nb, self.scaler.transform(x))


class RiverIncremental:
    def __init__(self, builder):
        self.builder = builder
        self.model = None

    def update(self, x: np.ndarray, y: np.ndarray, seed: int) -> dict:
        y = np.asarray(y).astype(int)
        if self.model is None:
            self.model = self.builder(seed)
        weights = balanced_weights(y)
        n_features = x.shape[1]
        for i in range(len(y)):
            row = {j: float(x[i, j]) for j in range(n_features)}
            self.model.learn_one(row, int(y[i]), w=float(weights[i]))
        return {"passes": 1.0}

    def predict(self, x: np.ndarray) -> np.ndarray:
        n_features = x.shape[1]
        out = np.empty(len(x), dtype=np.float64)
        for i in range(len(x)):
            row = {j: float(x[i, j]) for j in range(n_features)}
            dist = self.model.predict_proba_one(row)
            out[i] = float(dist.get(1, 0.0))
        return out


def cat_params(seed: int, iterations: int) -> dict:
    return {
        "iterations": int(iterations),
        "depth": 3,
        "learning_rate": 0.1,
        "loss_function": "Logloss",
        "min_data_in_leaf": 8,
        "verbose": False,
        "allow_writing_files": False,
        "random_seed": seed,
        "thread_count": 1,
    }


class CatBoostIncremental:
    def __init__(self):
        self.model = None

    def _fit(self, x, y, seed, init, iterations):
        model = CatBoostClassifier(**cat_params(seed, iterations))
        kwargs = {"sample_weight": balanced_weights(y)}
        if init is not None:
            kwargs["init_model"] = init
        model.fit(x, y, **kwargs)
        return model

    def update(self, x: np.ndarray, y: np.ndarray, seed: int) -> dict:
        y = np.asarray(y).astype(int)
        weights = balanced_weights(y)
        split = try_inner(y, seed)
        old = self.model
        if split is None:
            rounds = 10 if old is None else 0
            if rounds:
                self.model = self._fit(x, y, seed, old, rounds)
            return {"extra_trees": float(rounds)}
        train_idx, val_idx = split
        probe = CatBoostClassifier(**cat_params(seed, MAX_TREES), use_best_model=True, od_type="Iter", od_wait=8)
        kwargs = {"sample_weight": weights[train_idx], "eval_set": (x[val_idx], y[val_idx])}
        if old is not None:
            kwargs["init_model"] = old
        probe.fit(x[train_idx], y[train_idx], **kwargs)
        extra = int(probe.tree_count_ if old is None else probe.tree_count_ - old.tree_count_)
        if old is not None:
            old_ll = log_loss(y[val_idx], np.clip(old.predict_proba(x[val_idx])[:, 1], 1e-7, 1 - 1e-7), labels=[0, 1])
            new_ll = log_loss(y[val_idx], np.clip(probe.predict_proba(x[val_idx])[:, 1], 1e-7, 1 - 1e-7), labels=[0, 1])
            if new_ll >= old_ll - 1e-6:
                extra = 0
        if extra <= 0 and old is not None:
            return {"extra_trees": 0.0}
        extra = max(extra, 1)
        self.model = self._fit(x, y, seed, old, extra)
        return {"extra_trees": float(extra)}

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(x)[:, 1]


def lgb_params(seed: int) -> dict:
    return {
        "objective": "binary",
        "learning_rate": 0.1,
        "max_depth": 3,
        "num_leaves": 8,
        "min_data_in_leaf": 8,
        "verbosity": -1,
        "seed": seed,
        "num_threads": 1,
        "force_row_wise": True,
    }


def lgb_fit(x, y, weights, seed, rounds, init=None, valid=None):
    train = lgb.Dataset(x, label=y, weight=weights, free_raw_data=False)
    callbacks = [lgb.log_evaluation(period=0)]
    valid_sets = None
    if valid is not None:
        valid_x, valid_y, valid_w = valid
        valid_sets = [lgb.Dataset(valid_x, label=valid_y, weight=valid_w, reference=train, free_raw_data=False)]
        callbacks.append(lgb.early_stopping(8, verbose=False))
    return lgb.train(
        lgb_params(seed),
        train,
        num_boost_round=int(rounds),
        init_model=init,
        valid_sets=valid_sets,
        callbacks=callbacks,
    )


class LightGBMIncremental:
    def __init__(self):
        self.model = None

    def update(self, x: np.ndarray, y: np.ndarray, seed: int) -> dict:
        y = np.asarray(y).astype(int)
        weights = balanced_weights(y)
        split = try_inner(y, seed)
        old = self.model
        if split is None:
            rounds = 10 if old is None else 0
            if rounds:
                self.model = lgb_fit(x, y, weights, seed, rounds, old)
            return {"extra_trees": float(rounds)}
        train_idx, val_idx = split
        probe = lgb_fit(
            x[train_idx],
            y[train_idx],
            weights[train_idx],
            seed,
            MAX_TREES,
            old,
            (x[val_idx], y[val_idx], weights[val_idx]),
        )
        extra = int(probe.num_trees() if old is None else probe.num_trees() - old.num_trees())
        if old is not None:
            old_p = np.clip(old.predict(x[val_idx]), 1e-7, 1 - 1e-7)
            new_p = np.clip(probe.predict(x[val_idx]), 1e-7, 1 - 1e-7)
            old_ll = log_loss(y[val_idx], old_p, labels=[0, 1])
            new_ll = log_loss(y[val_idx], new_p, labels=[0, 1])
            if new_ll >= old_ll - 1e-6:
                extra = 0
        if extra <= 0 and old is not None:
            return {"extra_trees": 0.0}
        extra = max(extra, 1)
        self.model = lgb_fit(x, y, weights, seed, extra, old)
        return {"extra_trees": float(extra)}

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)


def factories(mlp_lr: float) -> dict:
    return {
        "logistic": LogisticMemory,
        "gbdt": SklearnGBDT,
        "mlp_small": lambda: TorchMLP((16,), mlp_lr),
        "mlp_128": lambda: TorchMLP((128, 32), mlp_lr),
        "sgd": SGDLogistic,
        "gaussian_nb": IncrementalNB,
        "hoeffding": lambda: RiverIncremental(
            lambda seed: HoeffdingTreeClassifier(grace_period=20, delta=0.01)
        ),
        "arf": lambda: RiverIncremental(lambda seed: ARFClassifier(n_models=10, seed=seed)),
        "catboost": CatBoostIncremental,
        "lightgbm": LightGBMIncremental,
    }


def run_schedule(factory, batches: list[tuple[np.ndarray, np.ndarray]], x_test, y_test, seed: int) -> list[dict]:
    model = factory()
    path = []
    for step, (x_batch, y_batch) in enumerate(batches):
        info = model.update(x_batch, y_batch, seed + 17 * (step + 1))
        row = score_metrics(y_test, model.predict(x_test))
        row.update(info)
        row["n_batch"] = float(len(y_batch))
        path.append(row)
    return path


def paired(left: list[dict], right: list[dict], key: str) -> str:
    delta = np.array([a[key] - b[key] for a, b in zip(left, right)], dtype=np.float64)
    return (
        f"{delta.mean():+.3f}±{delta.std():.3f} "
        f"higher {np.mean(delta > 1e-9):.0%} lower {np.mean(delta < -1e-9):.0%}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--mlp-lr", type=float, default=1e-3)
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)
    torch.set_num_threads(1)

    data, _meta = load_cache(ROOT / "output")
    y = data["y"].astype(int)
    x59 = build_matrix(data, 640, 0.25)
    x, names = take_columns(x59, FEATURE_NAMES, DROPPED)
    builders = factories(args.mlp_lr)
    model_names = list(builders)
    finals = {name: {protocol: [] for protocol in PROTOCOLS} for name in model_names}
    paths = {name: {protocol: [] for protocol in ("two_session", "fractured")} for name in model_names}
    chunk_rates = []

    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        test_idx, two_idx, fractured_idx = make_split(y, rng)
        x_test, y_test = x[test_idx], y[test_idx]
        two = [(x[idx], y[idx]) for idx in two_idx]
        fractured = [(x[idx], y[idx]) for idx in fractured_idx]
        replay = [(np.concatenate([batch for batch, _ in two]), np.concatenate([labels for _, labels in two]))]
        chunk_rates.append([float(labels.mean()) for _, labels in fractured])
        for name in model_names:
            paths[name]["two_session"].append(run_schedule(builders[name], two, x_test, y_test, seed))
            paths[name]["fractured"].append(run_schedule(builders[name], fractured, x_test, y_test, seed + 1000))
            finals[name]["two_session"].append(paths[name]["two_session"][-1][-1])
            finals[name]["fractured"].append(paths[name]["fractured"][-1][-1])
            finals[name]["replay"].append(run_schedule(builders[name], replay, x_test, y_test, seed + 2000)[-1])
        small_epochs = [step["best_epoch"] for step in paths["mlp_small"]["fractured"][-1]]
        wide_epochs = [step["best_epoch"] for step in paths["mlp_128"]["fractured"][-1]]
        print(
            f"seed {seed}  logistic two {finals['logistic']['two_session'][-1]['roc_auc']:.3f} "
            f"frac {finals['logistic']['fractured'][-1]['roc_auc']:.3f}  "
            f"mlp16 epochs {small_epochs}  mlp128 epochs {wide_epochs}",
            flush=True,
        )

    lines = [
        f"n={len(y)} features={len(names)} delete_rate={y.mean():.3f} mlp_lr={args.mlp_lr:g}",
        f"Each seed holds out {TEST_N} and uses the same {POOL_N} training photos for every schedule.",
        f"two_session sizes {list(TWO)}. fractured sizes {list(FRACTURED)}. replay fits those {POOL_N} at once.",
        "Boosters choose how many new trees to add, from 0 to 40, on an inner 20% of the current batch, then refit that many on the whole batch.",
        "Logistic regression stores the Hessian and freezes the scaler after the first batch. The MLP and SGD do the same with the scaler, and stop on an inner 20%.",
        "Naive Bayes updates its class statistics once per batch. Hoeffding and the forest take one weighted pass per batch.",
        "Hoeffding grace_period is 20 and delta is 0.01. The library defaults, 200 and 1e-7, are aimed at streams of millions and would not split on a few hundred labels.",
        "The adaptive forest keeps its defaults: 10 trees, grace_period 50, delta 0.01.",
        "",
        "Mean fractured delete rate by batch: "
        + " ".join(f"{n}:{rate:.3f}" for n, rate in zip(FRACTURED, np.mean(chunk_rates, axis=0))),
        "",
        "Final test metrics after the schedule has used all 400 labels.",
        "",
    ]

    def say(text: str = "") -> None:
        print(text, flush=True)
        lines.append(text)

    packed = {}
    say(f"{'model':<14} {'schedule':<12} result")
    for name in model_names:
        for protocol in PROTOCOLS:
            result = pack(f"{name}_{protocol}", finals[name][protocol], "53")
            packed[f"{name}_{protocol}"] = result
            say(f"{name:<14} {protocol:<12} {brief(result)}")
        say("")

    say("## Paired AUC and delete F1, same test photos")
    say("fractured minus two_session, then fractured minus replay, then two_session minus replay.")
    for name in model_names:
        say(
            f"{name:<14} auc frac-two {paired(finals[name]['fractured'], finals[name]['two_session'], 'roc_auc')}  "
            f"frac-replay {paired(finals[name]['fractured'], finals[name]['replay'], 'roc_auc')}"
        )
        say(
            f"{'':<14} auc two-replay {paired(finals[name]['two_session'], finals[name]['replay'], 'roc_auc')}  "
            f"f1 frac-two {paired(finals[name]['fractured'], finals[name]['two_session'], 'f1_delete')}"
        )

    say("")
    say("## Fractured test AUC after each batch")
    header = " ".join(f"{n:>7}" for n in np.cumsum(FRACTURED))
    say(f"{'model':<14} {header}")
    path_means = {}
    for name in model_names:
        step_aucs = []
        for step in range(len(FRACTURED)):
            vals = [seed_path[step]["roc_auc"] for seed_path in paths[name]["fractured"]]
            step_aucs.append(float(np.mean(vals)))
        path_means[name] = step_aucs
        say(f"{name:<14} " + " ".join(f"{value:7.3f}" for value in step_aucs))

    say("")
    say("## MLP epochs kept (0 means the incoming weights won)")
    for name in ("mlp_small", "mlp_128"):
        for protocol, sizes in (("two_session", TWO), ("fractured", FRACTURED)):
            kept = []
            ran = []
            for step in range(len(sizes)):
                kept.append(float(np.mean([seed_path[step]["best_epoch"] for seed_path in paths[name][protocol]])))
                ran.append(float(np.mean([seed_path[step]["epochs_ran"] for seed_path in paths[name][protocol]])))
            say(f"{name} {protocol} sizes {list(sizes)}")
            say("  kept " + " ".join(f"{value:.1f}" for value in kept))
            say("  ran  " + " ".join(f"{value:.1f}" for value in ran))

    payload = {
        "mlp_lr": args.mlp_lr,
        "seeds": args.seeds,
        "fractured_sizes": list(FRACTURED),
        "two_session_sizes": list(TWO),
        "feature_names": names,
        "finals": {key: value for key, value in packed.items()},
        "fractured_auc_path": path_means,
        "chunk_delete_rates": chunk_rates,
    }
    out_json = ROOT / "output" / "fractured_study.json"
    out_md = ROOT / "output" / "fractured_study.md"
    out_json.write_text(json.dumps(payload, indent=2))
    out_md.write_text("\n".join(lines) + "\n")
    say("")
    say(f"wrote {out_json}")


if __name__ == "__main__":
    main()
