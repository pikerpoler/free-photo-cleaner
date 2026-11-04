"""Fit the planned scorers on cached features and compare them honestly.

The app's GBDT and MLP pick the saved round by test log loss. That number is
optimistic, so each of those models is also scored with complexity chosen on an
inner slice of the training fold only. Logistic regression and the trees see
the same 80/20 stratified splits.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight

from photo_features import (
    AUX_NAMES,
    CAT_SLICE,
    FACE_SLICE,
    FEATURE_NAMES,
    GEOM_SLICE,
    META_SLICE,
    PHOTO_SLICE,
    TEXT_SLICE,
    aggregate_detections,
    coco_max_confidence,
    iter_detections,
    matrix_from_parts,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent
torch.set_num_threads(4)

METRIC_KEYS = [
    "log_loss",
    "accuracy",
    "balanced_accuracy",
    "roc_auc",
    "avg_precision",
    "brier",
    "precision_at_20",
    "recall_at_20",
    "mean_p",
]


def load_cache(out_dir: Path):
    data = np.load(out_dir / "features.npz", allow_pickle=False)
    meta = json.loads((out_dir / "features_meta.json").read_text())
    return data, meta


def build_matrix(data, resolution: int, conf: float, zero_faces: bool = False) -> np.ndarray:
    key = f"det{resolution}"
    dets = iter_detections(data[key], data[f"{key}_counts"])
    vision = data["vision"].astype(np.float64)
    if zero_faces:
        vision = vision.copy()
        vision[:, :4] = 0.0
    return matrix_from_parts(
        data["photo"].astype(np.float64),
        vision,
        data["meta"].astype(np.float64),
        dets,
        data["class_to_cat"],
        conf,
    )


def make_splits(y: np.ndarray, n_splits: int, test_size: float = 0.2, seed: int = 0):
    splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
    return list(splitter.split(np.zeros(len(y)), y))


def score_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    pred = (p >= 0.5).astype(int)
    out = {
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1_delete": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "precision_delete": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "recall_delete": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "pred_delete_rate": float(pred.mean()),
        "brier": float(brier_score_loss(y, p)),
        "mean_p": float(p.mean()),
    }
    if len(np.unique(y)) < 2 or np.allclose(p, p[0]):
        out["roc_auc"] = 0.5
        out["avg_precision"] = float(y.mean())
    else:
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["avg_precision"] = float(average_precision_score(y, p))
    k = max(1, int(round(0.2 * len(y))))
    jitter = np.linspace(0, 1e-9, len(p))
    top = np.argsort(-(p + jitter))[:k]
    out["precision_at_20"] = float(y[top].mean())
    out["recall_at_20"] = float(y[top].sum() / max(float(y.sum()), 1.0))
    return out


def both_sides(y_train, p_train, y_test, p_test, extra: dict | None = None) -> dict:
    row = {}
    for key, value in score_metrics(y_test, p_test).items():
        row[f"test_{key}"] = value
    for key, value in score_metrics(y_train, p_train).items():
        row[f"train_{key}"] = value
    row["overfit_log_loss"] = row["test_log_loss"] - row["train_log_loss"]
    row["overfit_auc"] = row["train_roc_auc"] - row["test_roc_auc"]
    if extra:
        row.update(extra)
    return row


def pack(name: str, folds: list[dict], features: str) -> dict:
    keys = sorted({key for fold in folds for key in fold if isinstance(fold[key], (int, float))})
    summary = {"model": name, "features": features, "n_folds": len(folds), "mean": {}, "std": {}}
    for key in keys:
        vals = np.array([fold[key] for fold in folds], dtype=np.float64)
        summary["mean"][key] = float(np.nanmean(vals))
        summary["std"][key] = float(np.nanstd(vals))
    summary["folds"] = folds
    return summary


def positive_proba(model, x: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(x)
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(x))
    return proba[:, classes.index(1)]


def scale_pair(x_train, x_test):
    scaler = StandardScaler()
    return scaler.fit_transform(x_train), scaler.transform(x_test), scaler


def fit_logistic(x_train, y_train, x_test, y_test, seed: int, C: float, balanced: bool, max_iter: int):
    del seed
    xtr, xte, _ = scale_pair(x_train, x_test)
    clf = LogisticRegression(
        C=C,
        class_weight="balanced" if balanced else None,
        solver="lbfgs",
        max_iter=max_iter,
    )
    clf.fit(xtr, y_train)
    extra = {"n_iter": float(clf.n_iter_[0]), "converged": float(clf.n_iter_[0] < max_iter)}
    row = both_sides(y_train, positive_proba(clf, xtr), y_test, positive_proba(clf, xte), extra)
    return row, clf.coef_.ravel().astype(np.float64)


def gbdt_model(seed: int, n_estimators: int, max_depth: int = 3, min_leaf: int = 8, learning_rate: float = 0.1):
    return GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_leaf,
        learning_rate=learning_rate,
        loss="log_loss",
        random_state=seed,
    )


def staged_best_k(model, x, y) -> tuple[int, np.ndarray]:
    best_k = 1
    best_ll = np.inf
    chosen = None
    for k, proba in enumerate(model.staged_predict_proba(x), start=1):
        p = np.clip(proba[:, 1], 1e-7, 1 - 1e-7)
        ll = log_loss(y, p, labels=[0, 1])
        if ll < best_ll:
            best_ll = ll
            best_k = k
            chosen = p
    return best_k, chosen


def proba_at_k(model, x, k: int) -> np.ndarray:
    chosen = None
    for i, proba in enumerate(model.staged_predict_proba(x), start=1):
        chosen = proba[:, 1]
        if i == k:
            break
    return chosen


def fit_gbdt_plan(x_train, y_train, x_test, y_test, seed: int, n_estimators: int = 40):
    model = gbdt_model(seed, n_estimators)
    model.fit(x_train, y_train)
    k, p_test = staged_best_k(model, x_test, y_test)
    p_train = proba_at_k(model, x_train, k)
    return both_sides(y_train, p_train, y_test, p_test, {"chosen_trees": float(k)}), model


def fit_gbdt_clean(
    x_train,
    y_train,
    x_test,
    y_test,
    seed: int,
    n_estimators: int = 40,
    max_depth: int = 3,
    min_leaf: int = 8,
    learning_rate: float = 0.1,
    balanced: bool = False,
):
    inner = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    inner_train, inner_val = next(inner.split(x_train, y_train))
    sample_weight = compute_sample_weight("balanced", y_train) if balanced else None
    selector = gbdt_model(seed, n_estimators, max_depth, min_leaf, learning_rate)
    sw_inner = None if sample_weight is None else sample_weight[inner_train]
    selector.fit(x_train[inner_train], y_train[inner_train], sample_weight=sw_inner)
    k, _ = staged_best_k(selector, x_train[inner_val], y_train[inner_val])
    model = gbdt_model(seed, k, max_depth, min_leaf, learning_rate)
    model.fit(x_train, y_train, sample_weight=sample_weight)
    return (
        both_sides(
            y_train,
            positive_proba(model, x_train),
            y_test,
            positive_proba(model, x_test),
            {"chosen_trees": float(k)},
        ),
        model,
    )


def fit_hist_gbdt(x_train, y_train, x_test, y_test, seed: int):
    model = HistGradientBoostingClassifier(
        max_depth=3,
        max_iter=40,
        learning_rate=0.1,
        min_samples_leaf=8,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=8,
        random_state=seed,
    )
    model.fit(x_train, y_train)
    return both_sides(
        y_train,
        positive_proba(model, x_train),
        y_test,
        positive_proba(model, x_test),
        {"chosen_trees": float(model.n_iter_)},
    )


def fit_forest(x_train, y_train, x_test, y_test, seed: int):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=8,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=4,
    )
    model.fit(x_train, y_train)
    return both_sides(y_train, positive_proba(model, x_train), y_test, positive_proba(model, x_test))


def fit_svm(x_train, y_train, x_test, y_test, seed: int):
    xtr, xte, _ = scale_pair(x_train, x_test)
    base = LinearSVC(C=1.0, class_weight="balanced", dual="auto", max_iter=5000, random_state=seed)
    model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    model.fit(xtr, y_train)
    return both_sides(y_train, positive_proba(model, xtr), y_test, positive_proba(model, xte))


def fit_nb(x_train, y_train, x_test, y_test, seed: int):
    del seed
    xtr, xte, _ = scale_pair(x_train, x_test)
    model = GaussianNB()
    model.fit(xtr, y_train)
    return both_sides(y_train, positive_proba(model, xtr), y_test, positive_proba(model, xte))


class MLP(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: tuple[int, ...]):
        super().__init__()
        layers: list[torch.nn.Module] = []
        prev = in_dim
        for width in hidden:
            layers.append(torch.nn.Linear(prev, width))
            layers.append(torch.nn.ReLU())
            prev = width
        layers.append(torch.nn.Linear(prev, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def mlp_predict(model: MLP, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)))
        return torch.sigmoid(logits).numpy()


def train_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_watch: np.ndarray | None,
    y_watch: np.ndarray | None,
    hidden: tuple[int, ...],
    seed: int,
    max_epochs: int,
    patience: int | None,
) -> tuple[MLP, int, int]:
    torch.manual_seed(seed)
    model = MLP(x_train.shape[1], hidden)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    x_t = torch.from_numpy(np.ascontiguousarray(x_train, dtype=np.float32))
    y_t = torch.from_numpy(np.ascontiguousarray(y_train, dtype=np.float32))
    n_pos = max(float(y_train.sum()), 1.0)
    neg_count = float(len(y_train) - y_train.sum())
    n_neg = max(neg_count, 1.0)
    w_pos = len(y_train) / (2.0 * n_pos)
    w_neg = len(y_train) / (2.0 * n_neg)
    weights = torch.where(y_t > 0.5, torch.full_like(y_t, w_pos), torch.full_like(y_t, w_neg))
    rng = np.random.default_rng(seed)
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    best_ll = np.inf
    best_epoch = 1
    wait = 0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    use_watch = x_watch is not None and patience is not None
    epochs_ran = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = rng.permutation(len(x_train))
        for start in range(0, len(x_train), 32):
            idx = order[start : start + 32]
            xb = x_t[idx]
            yb = y_t[idx]
            opt.zero_grad()
            loss = (bce(model(xb), yb) * weights[idx]).mean()
            loss.backward()
            opt.step()
        epochs_ran = epoch
        if not use_watch:
            best_epoch = epoch
            continue
        p = np.clip(mlp_predict(model, x_watch), 1e-7, 1 - 1e-7)
        ll = log_loss(y_watch, p, labels=[0, 1])
        if ll < best_ll:
            best_ll = ll
            best_epoch = epoch
            wait = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    if use_watch:
        model.load_state_dict(best_state)
    return model, best_epoch, epochs_ran


def fit_mlp(x_train, y_train, x_test, y_test, seed: int, hidden: tuple[int, ...], protocol: str):
    if protocol == "plan":
        xtr, xte, _ = scale_pair(x_train, x_test)
        model, best_epoch, epochs_ran = train_mlp(xtr, y_train, xte, y_test, hidden, seed, 80, 8)
        return both_sides(
            y_train,
            mlp_predict(model, xtr),
            y_test,
            mlp_predict(model, xte),
            {"best_epoch": float(best_epoch), "epochs_ran": float(epochs_ran)},
        )
    inner = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    inner_train, inner_val = next(inner.split(x_train, y_train))
    x_inner, x_val, _ = scale_pair(x_train[inner_train], x_train[inner_val])
    _, best_epoch, _ = train_mlp(
        x_inner,
        y_train[inner_train],
        x_val,
        y_train[inner_val],
        hidden,
        seed,
        80,
        8,
    )
    xtr, xte, _ = scale_pair(x_train, x_test)
    model, _, _ = train_mlp(xtr, y_train, None, None, hidden, seed, int(best_epoch), None)
    return both_sides(
        y_train,
        mlp_predict(model, xtr),
        y_test,
        mlp_predict(model, xte),
        {"best_epoch": float(best_epoch), "epochs_ran": float(best_epoch)},
    )


def constant_baseline(x_train, y_train, x_test, y_test, seed: int):
    del x_train, x_test, seed
    p_test = np.full(len(y_test), float(y_train.mean()))
    p_train = np.full(len(y_train), float(y_train.mean()))
    return both_sides(y_train, p_train, y_test, p_test)


def screenshot_baseline(x_train, y_train, x_test, y_test, seed: int):
    del seed
    idx = FEATURE_NAMES.index("screenshot")
    ss_train = x_train[:, idx] > 0.5
    ss_test = x_test[:, idx] > 0.5
    p_ss = float(y_train[ss_train].mean()) if ss_train.any() else float(y_train.mean())
    p_other = float(y_train[~ss_train].mean()) if (~ss_train).any() else float(y_train.mean())
    p_train = np.where(ss_train, p_ss, p_other)
    p_test = np.where(ss_test, p_ss, p_other)
    return both_sides(y_train, p_train, y_test, p_test)


def run_folds(name: str, features: str, splits, x, y, trainer) -> dict:
    folds = []
    for seed, (train_idx, test_idx) in enumerate(splits):
        folds.append(trainer(x[train_idx], y[train_idx], x[test_idx], y[test_idx], seed))
    return pack(name, folds, features)


def permutation_auc_drop(model, x, y, predict, groups, seed: int) -> dict[str, float]:
    base = score_metrics(y, predict(model, x))["roc_auc"]
    rng = np.random.default_rng(seed)
    drops = {}
    for name, cols in groups.items():
        scores = []
        for _ in range(4):
            xp = np.array(x, copy=True)
            xp[:, cols] = x[rng.permutation(len(x))][:, cols]
            scores.append(score_metrics(y, predict(model, xp))["roc_auc"])
        drops[name] = float(base - float(np.mean(scores)))
    drops["base_auc"] = float(base)
    return drops


def feature_groups() -> dict[str, list[int]]:
    names = FEATURE_NAMES
    return {
        "photometric": list(range(PHOTO_SLICE.start, PHOTO_SLICE.stop)),
        "yolo_geometry": list(range(GEOM_SLICE.start, GEOM_SLICE.stop)),
        "yolo_categories": list(range(CAT_SLICE.start, CAT_SLICE.stop)),
        "faces": list(range(FACE_SLICE.start, FACE_SLICE.stop)),
        "text": list(range(TEXT_SLICE.start, TEXT_SLICE.stop)),
        "metadata": list(range(META_SLICE.start, META_SLICE.stop)),
        "screenshot": [names.index("screenshot")],
        "person": [names.index("person_max_conf"), names.index("person_sum_conf")],
    }


def univariate(y: np.ndarray, values: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(values)
    nonzero = float(np.mean(values[mask] != 0)) if mask.any() else 0.0
    if mask.sum() < 20 or len(np.unique(y[mask])) < 2 or np.std(values[mask]) < 1e-12:
        return {"auc": 0.5, "strength": 0.5, "direction": "none", "nonzero": nonzero}
    auc = float(roc_auc_score(y[mask], values[mask]))
    direction = "higher_means_delete" if auc >= 0.5 else "higher_means_keep"
    return {"auc": auc, "strength": float(max(auc, 1.0 - auc)), "direction": direction, "nonzero": nonzero}


def diagnostics(x: np.ndarray, y: np.ndarray, aux: np.ndarray, meta_json: dict) -> dict:
    rows = []
    for i, name in enumerate(FEATURE_NAMES):
        stats = univariate(y, x[:, i])
        stats["name"] = name
        stats["std"] = float(np.nanstd(x[:, i]))
        stats["p99"] = float(np.nanpercentile(x[:, i], 99))
        stats["max"] = float(np.nanmax(x[:, i]))
        rows.append(stats)
    ratio = x[:, FEATURE_NAMES.index("center_border_sharpness_ratio")]
    ratio_stats = {
        "p50": float(np.percentile(ratio, 50)),
        "p90": float(np.percentile(ratio, 90)),
        "p99": float(np.percentile(ratio, 99)),
        "max": float(ratio.max()),
        "frac_above_20": float(np.mean(ratio > 20)),
        "frac_above_100": float(np.mean(ratio > 100)),
    }
    corr_pairs = []
    with np.errstate(invalid="ignore"):
        corr = np.corrcoef(x.T)
    for i in range(len(FEATURE_NAMES)):
        for j in range(i + 1, len(FEATURE_NAMES)):
            value = corr[i, j]
            if np.isfinite(value) and abs(value) >= 0.9:
                corr_pairs.append(
                    {"a": FEATURE_NAMES[i], "b": FEATURE_NAMES[j], "corr": float(value)}
                )
    corr_pairs.sort(key=lambda item: -abs(item["corr"]))

    active = np.std(x, axis=0) > 1e-8
    scaled = StandardScaler().fit_transform(x[:, active])
    singular = np.linalg.svd(scaled, compute_uv=False)
    condition = float(singular[0] / max(singular[-1], 1e-12))

    cat_multi = {}
    for cat_i, cat in enumerate(
        ["person", "vehicle", "outdoor", "animal", "accessory", "sports", "kitchen", "furniture", "electronic", "appliance", "indoor"]
    ):
        max_conf = x[:, CAT_SLICE.start + 2 * cat_i]
        sum_conf = x[:, CAT_SLICE.start + 2 * cat_i + 1]
        present = max_conf > 0
        multi = (sum_conf > max_conf + 1e-6) & present
        cat_multi[cat] = {
            "frac_present": float(present.mean()),
            "frac_of_present_with_multiple": float(multi.sum() / max(present.sum(), 1)),
            "corr_max_sum": float(np.corrcoef(max_conf, sum_conf)[0, 1]) if present.any() and max_conf.std() > 0 else 1.0,
        }

    quarter_sum_error = float(np.max(np.abs(x[:, 12:16].sum(axis=1) - 1.0)))
    aux_stats = {}
    for i, name in enumerate(AUX_NAMES):
        aux_stats[name] = univariate(y, aux[:, i])
    nonscreen = x[:, FEATURE_NAMES.index("screenshot")] < 0.5
    file_size = aux[:, AUX_NAMES.index("file_size")]
    aux_stats["file_size_among_nonscreenshots"] = univariate(y[nonscreen], -np.log1p(file_size[nonscreen]))
    aux_stats["creation_date_ms"] = univariate(y, aux[:, AUX_NAMES.index("creation_date_ms")])

    dead = [row["name"] for row in rows if row["nonzero"] < 0.01 or row["std"] < 1e-8]
    return {
        "n": int(len(y)),
        "delete_rate": float(y.mean()),
        "features": rows,
        "dead_or_near_constant": dead,
        "sharpness_ratio": ratio_stats,
        "high_correlations": corr_pairs[:40],
        "condition_number_nonzero": condition,
        "n_nonzero_variance": int(active.sum()),
        "histogram_mass_sum_max_error": quarter_sum_error,
        "category_redundancy": cat_multi,
        "aux_univariate": aux_stats,
        "detector": meta_json.get("detector"),
        "face_detector": meta_json.get("face_detector"),
        "n_screenshot": meta_json.get("n_screenshot"),
        "n_gps": meta_json.get("n_gps"),
        "n_with_face": meta_json.get("n_with_face"),
        "screenshot_flag_disagreements": meta_json.get("screenshot_flag_disagreements"),
    }


def log1p_scale_sensitive(x: np.ndarray) -> np.ndarray:
    out = np.array(x, copy=True)
    for name in ("laplacian_var", "center_border_sharpness_ratio", "hf_noise_mad"):
        idx = FEATURE_NAMES.index(name)
        out[:, idx] = np.log1p(np.clip(out[:, idx], 0, None))
    return out


def expanded_coco(data, conf: float) -> tuple[np.ndarray, np.ndarray]:
    dets = iter_detections(data["det640"], data["det640_counts"])
    n_classes = int(data["class_to_cat"].shape[0])
    per_class = np.stack([coco_max_confidence(det, n_classes, conf) for det in dets])
    geom = []
    for det in dets:
        geom.append(aggregate_detections(det, data["class_to_cat"], conf)[:6])
    geom = np.stack(geom)
    photo = data["photo"].astype(np.float64)
    vision = data["vision"].astype(np.float64)
    meta = data["meta"].astype(np.float64)
    expanded = np.concatenate([photo, geom, per_class, vision, meta], axis=1)
    return per_class, expanded


def learning_curve(x, y, sizes: list[int], repeats: int) -> list[dict]:
    rng = np.random.default_rng(1)
    rows = []
    for n in sizes:
        fold_rows = {"logistic_l2": [], "gbdt_clean": [], "mlp_small_clean": [], "mlp_big_clean": []}
        failures = 0
        for rep in range(repeats):
            try:
                idx = stratified_take(y, n, rng)
                splitter = StratifiedShuffleSplit(
                    n_splits=1, test_size=0.2, random_state=int(rng.integers(0, 10_000))
                )
                train_idx, test_idx = next(splitter.split(np.zeros(len(idx)), y[idx]))
                x_sub, y_sub = x[idx], y[idx]
                xtr, xte = x_sub[train_idx], x_sub[test_idx]
                ytr, yte = y_sub[train_idx], y_sub[test_idx]
                if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
                    raise ValueError("split missing a class")
                fold_rows["logistic_l2"].append(fit_logistic(xtr, ytr, xte, yte, rep, 1.0, True, 500)[0])
                fold_rows["gbdt_clean"].append(
                    fit_gbdt_clean(xtr, ytr, xte, yte, rep, n_estimators=40)[0]
                )
                fold_rows["mlp_small_clean"].append(fit_mlp(xtr, ytr, xte, yte, rep, (16,), "clean"))
                fold_rows["mlp_big_clean"].append(fit_mlp(xtr, ytr, xte, yte, rep, (64, 32), "clean"))
            except Exception as exc:
                failures += 1
                print(f"  learning curve n={n} rep {rep} failed: {exc}")
                continue
        point = {"n": n, "failures": failures, "repeats": repeats}
        for model, folds in fold_rows.items():
            if not folds:
                point[model] = None
                continue
            point[model] = {
                "auc_mean": float(np.mean([f["test_roc_auc"] for f in folds])),
                "auc_std": float(np.std([f["test_roc_auc"] for f in folds])),
                "log_loss_mean": float(np.mean([f["test_log_loss"] for f in folds])),
                "overfit_auc_mean": float(np.mean([f["overfit_auc"] for f in folds])),
            }
        rows.append(point)
        auc = point["logistic_l2"]["auc_mean"] if point["logistic_l2"] else float("nan")
        print(f"  learning curve n={n} logistic auc={auc:.3f}")
    return rows


def stratified_take(y: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if n >= len(y):
        return np.arange(len(y))
    idx0 = np.flatnonzero(y == 0)
    idx1 = np.flatnonzero(y == 1)
    n1 = int(round(n * float(y.mean())))
    n1 = min(max(n1, 2), len(idx1), n - 2)
    n0 = n - n1
    take0 = rng.choice(idx0, n0, replace=False)
    take1 = rng.choice(idx1, n1, replace=False)
    return np.concatenate([take0, take1])


def fmt(mean: dict, key: str) -> str:
    if key not in mean:
        return "   n/a"
    return f"{mean[key]:6.3f}"


def leaderboard_line(result: dict) -> str:
    mean = result["mean"]
    std = result["std"]
    return (
        f"{result['model']:<28} {result['features']:<22} "
        f"{fmt(mean, 'test_roc_auc')}±{std.get('test_roc_auc', 0):.3f}  "
        f"{fmt(mean, 'test_log_loss')}  {fmt(mean, 'test_accuracy')}  "
        f"{fmt(mean, 'test_balanced_accuracy')}  {fmt(mean, 'test_precision_at_20')}  "
        f"{fmt(mean, 'overfit_auc')}"
    )


def write_report(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "output")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--skip-curve", action="store_true")
    args = parser.parse_args()

    data, meta_json = load_cache(args.out)
    y = data["y"].astype(int)
    aux = data["aux"].astype(np.float64)
    x = build_matrix(data, 640, 0.25)
    splits = make_splits(y, args.seeds)
    print(f"n={len(y)} delete_rate={y.mean():.3f} features={x.shape[1]} splits={len(splits)}")

    diag = diagnostics(x, y, aux, meta_json)
    lines = []

    def say(text: str = "") -> None:
        print(text)
        lines.append(text)

    say("# Offline feature-model experiment")
    say("")
    say(f"Images {diag['n']}, delete rate {diag['delete_rate']:.3f}, detector {diag['detector']}, faces {diag['face_detector']}.")
    say(f"Screenshots {diag['n_screenshot']}, GPS {diag['n_gps']}, YuNet faces {diag['n_with_face']}.")
    say(f"Nonzero-variance features {diag['n_nonzero_variance']} / 59. Condition number {diag['condition_number_nonzero']:.1f}.")
    say(f"Histogram-quarter sum error {diag['histogram_mass_sum_max_error']:.2e}.")
    say(f"Sharpness ratio p50 {diag['sharpness_ratio']['p50']:.2f}, p99 {diag['sharpness_ratio']['p99']:.1f}, max {diag['sharpness_ratio']['max']:.1f}, fraction >100 {diag['sharpness_ratio']['frac_above_100']:.3f}.")
    say("")
    say("## Strongest single features")
    ranked = sorted(diag["features"], key=lambda row: -row["strength"])
    for row in ranked[:15]:
        say(
            f"  {row['name']:<36} strength {row['strength']:.3f}  auc {row['auc']:.3f}  {row['direction']}  nonzero {row['nonzero']:.2f}"
        )
    say("")
    say("## Dead or near-constant features")
    say("  " + ", ".join(diag["dead_or_near_constant"]))
    say("")
    say("## Category occupancy")
    for name, info in diag["category_redundancy"].items():
        say(
            f"  {name:<12} present {info['frac_present']:.3f}  multi {info['frac_of_present_with_multiple']:.2f}  corr(max,sum) {info['corr_max_sum']:.3f}"
        )
    say("")
    say("## Correlations at least 0.9")
    for pair in diag["high_correlations"][:20]:
        say(f"  {pair['a']} vs {pair['b']}: {pair['corr']:.3f}")
    say("")

    header = f"{'model':<28} {'features':<22} {'auc':>12}  {'logloss':>7}  {'acc':>6}  {'bal':>6}  {'p@20':>6}  {'overfit_auc':>6}"
    say("## Horse race on the 59-d vector, imgsz 640, conf 0.25")
    say(header)

    results = []

    def add(result: dict) -> None:
        results.append(result)
        say(leaderboard_line(result))

    add(run_folds("base_rate", "none", splits, x, y, constant_baseline))
    add(run_folds("screenshot_rule", "screenshot", splits, x, y, screenshot_baseline))

    coefs = []

    def logistic_runner(C, balanced, max_iter):
        def trainer(xtr, ytr, xte, yte, seed):
            row, coef = fit_logistic(xtr, ytr, xte, yte, seed, C, balanced, max_iter)
            if C == 1.0 and balanced and max_iter >= 500:
                coefs.append(coef)
            return row

        return trainer

    add(run_folds("logistic_l2_plan15", "full", splits, x, y, logistic_runner(1.0, True, 15)))
    add(run_folds("logistic_l2", "full", splits, x, y, logistic_runner(1.0, True, 500)))
    add(run_folds("logistic_unbalanced", "full", splits, x, y, logistic_runner(1.0, False, 500)))
    add(run_folds("logistic_C0.1", "full", splits, x, y, logistic_runner(0.1, True, 500)))
    add(run_folds("logistic_C10", "full", splits, x, y, logistic_runner(10.0, True, 500)))

    def gbdt_plan_runner(xtr, ytr, xte, yte, seed):
        row, _ = fit_gbdt_plan(xtr, ytr, xte, yte, seed, 40)
        return row

    def gbdt_clean_runner(xtr, ytr, xte, yte, seed):
        row, _ = fit_gbdt_clean(xtr, ytr, xte, yte, seed, 40)
        return row

    def gbdt_fixed_runner(xtr, ytr, xte, yte, seed):
        model = gbdt_model(seed, 40)
        model.fit(xtr, ytr)
        return both_sides(ytr, positive_proba(model, xtr), yte, positive_proba(model, xte), {"chosen_trees": 40.0})

    def gbdt_balanced_runner(xtr, ytr, xte, yte, seed):
        row, _ = fit_gbdt_clean(xtr, ytr, xte, yte, seed, 40, balanced=True)
        return row

    add(run_folds("gbdt_plan", "full", splits, x, y, gbdt_plan_runner))
    add(run_folds("gbdt_clean", "full", splits, x, y, gbdt_clean_runner))
    add(run_folds("gbdt_fixed40", "full", splits, x, y, gbdt_fixed_runner))
    add(run_folds("gbdt_balanced_clean", "full", splits, x, y, gbdt_balanced_runner))
    add(run_folds("hist_gbdt", "full", splits, x, y, fit_hist_gbdt))
    add(run_folds("random_forest", "full", splits, x, y, fit_forest))
    add(run_folds("linear_svm_calibrated", "full", splits, x, y, fit_svm))
    add(run_folds("gaussian_nb", "full", splits, x, y, fit_nb))

    def mlp_runner(hidden, protocol):
        def trainer(xtr, ytr, xte, yte, seed):
            return fit_mlp(xtr, ytr, xte, yte, seed, hidden, protocol)

        return trainer

    say("fitting MLPs...")
    add(run_folds("mlp_small_plan", "full", splits, x, y, mlp_runner((16,), "plan")))
    add(run_folds("mlp_small_clean", "full", splits, x, y, mlp_runner((16,), "clean")))
    add(run_folds("mlp_big_plan", "full", splits, x, y, mlp_runner((64, 32), "plan")))
    add(run_folds("mlp_big_clean", "full", splits, x, y, mlp_runner((64, 32), "clean")))

    if coefs:
        mean_coef = np.mean(np.stack(coefs), axis=0)
        order = np.argsort(-np.abs(mean_coef))
        say("")
        say("## Mean scaled logistic coefficients (positive predicts delete)")
        for idx in order[:15]:
            say(f"  {FEATURE_NAMES[idx]:<36} {mean_coef[idx]:+.3f}")

    say("")
    say("## Grouped permutation AUC drop")
    groups = feature_groups()
    perm_rows = {"logistic_l2": [], "gbdt_clean": []}
    for seed, (train_idx, test_idx) in enumerate(splits):
        xtr, xte, _ = scale_pair(x[train_idx], x[test_idx])
        clf = LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs", max_iter=500)
        clf.fit(xtr, y[train_idx])
        perm_rows["logistic_l2"].append(
            permutation_auc_drop(clf, xte, y[test_idx], positive_proba, groups, seed)
        )
        _, gbdt = fit_gbdt_clean(x[train_idx], y[train_idx], x[test_idx], y[test_idx], seed, 40)
        perm_rows["gbdt_clean"].append(
            permutation_auc_drop(gbdt, x[test_idx], y[test_idx], positive_proba, groups, seed)
        )
    perm_summary = {}
    for model_name, folds in perm_rows.items():
        perm_summary[model_name] = {}
        say(f"  {model_name}")
        for key in folds[0]:
            val = float(np.mean([fold[key] for fold in folds]))
            perm_summary[model_name][key] = val
            say(f"    {key:<20} {val:+.3f}")

    say("")
    say("## Ablations (logistic_l2 and gbdt_clean)")
    ablations: dict[str, np.ndarray] = {
        "photometric": x[:, PHOTO_SLICE],
        "yolo_geometry": x[:, GEOM_SLICE],
        "yolo_categories": x[:, CAT_SLICE],
        "yolo_all": x[:, GEOM_SLICE.start : CAT_SLICE.stop],
        "yunet_faces": x[:, FACE_SLICE],
        "metadata": x[:, META_SLICE],
        "screenshot_only": x[:, [FEATURE_NAMES.index("screenshot")]],
        "no_metadata": np.concatenate([x[:, :META_SLICE.start]], axis=1),
        "no_faces": build_matrix(data, 640, 0.25, zero_faces=True),
        "full": x,
    }
    per_class, expanded = expanded_coco(data, 0.25)
    ablations["coco80_maxconf"] = per_class
    ablations["photo_geom_coco80_rest"] = expanded
    ablations["log1p_sharpness"] = log1p_scale_sensitive(x)

    # Drop features that are constant on the whole set. Descriptive, not a nested selection.
    active = np.std(x, axis=0) > 1e-8
    ablations["drop_constant"] = x[:, active]

    ablation_results = []
    say(header)
    for feature_name, matrix in ablations.items():
        for model_name, trainer in (
            ("logistic_l2", lambda xtr, ytr, xte, yte, seed: fit_logistic(xtr, ytr, xte, yte, seed, 1.0, True, 500)[0]),
            ("gbdt_clean", lambda xtr, ytr, xte, yte, seed: fit_gbdt_clean(xtr, ytr, xte, yte, seed, 40)[0]),
        ):
            result = run_folds(model_name, feature_name, splits, matrix, y, trainer)
            ablation_results.append(result)
            say(leaderboard_line(result))

    say("")
    say("## Confidence threshold and detector input size")
    say(header)
    threshold_results = []
    for conf in (0.10, 0.25, 0.50):
        matrix = build_matrix(data, 640, conf)
        for model_name, trainer in (
            ("logistic_l2", lambda xtr, ytr, xte, yte, seed: fit_logistic(xtr, ytr, xte, yte, seed, 1.0, True, 500)[0]),
            ("gbdt_clean", lambda xtr, ytr, xte, yte, seed: fit_gbdt_clean(xtr, ytr, xte, yte, seed, 40)[0]),
        ):
            result = run_folds(model_name, f"full640_t{conf:.2f}", splits, matrix, y, trainer)
            threshold_results.append(result)
            say(leaderboard_line(result))
    matrix416 = build_matrix(data, 416, 0.25)
    for model_name, trainer in (
        ("logistic_l2", lambda xtr, ytr, xte, yte, seed: fit_logistic(xtr, ytr, xte, yte, seed, 1.0, True, 500)[0]),
        ("gbdt_clean", lambda xtr, ytr, xte, yte, seed: fit_gbdt_clean(xtr, ytr, xte, yte, seed, 40)[0]),
    ):
        result = run_folds(model_name, "full416_t0.25", splits, matrix416, y, trainer)
        threshold_results.append(result)
        say(leaderboard_line(result))

    say("")
    say("## GBDT depth / leaf grid, clean protocol, up to 80 trees, lr 0.1")
    grid = []
    say(header)
    for depth in (2, 3, 4):
        for leaf in (4, 8, 16):
            def trainer(xtr, ytr, xte, yte, seed, depth=depth, leaf=leaf):
                row, _ = fit_gbdt_clean(
                    xtr, ytr, xte, yte, seed, n_estimators=80, max_depth=depth, min_leaf=leaf
                )
                return row

            result = run_folds("gbdt_clean", f"depth{depth}_leaf{leaf}", splits, x, y, trainer)
            grid.append(result)
            say(leaderboard_line(result))

    curve = []
    if not args.skip_curve:
        say("")
        say("## Learning curve (clean protocol)")
        curve = learning_curve(x, y, [20, 40, 80, 150, 300, 500, len(y)], repeats=5)
        for point in curve:
            bits = [f"n={point['n']}"]
            for model in ("logistic_l2", "gbdt_clean", "mlp_small_clean", "mlp_big_clean"):
                info = point[model]
                if info is None:
                    bits.append(f"{model}=failed")
                else:
                    bits.append(
                        f"{model} auc {info['auc_mean']:.3f}±{info['auc_std']:.3f} gap {info['overfit_auc_mean']:.3f}"
                    )
            say("  " + " | ".join(bits))

    # Mistake sketch on the first split, clean GBDT.
    train_idx, test_idx = splits[0]
    gbdt_row, gbdt = fit_gbdt_clean(x[train_idx], y[train_idx], x[test_idx], y[test_idx], 0, 40)
    p = positive_proba(gbdt, x[test_idx])
    y_test = y[test_idx]
    filenames = data["filenames"]
    ids = data["ids"]
    say("")
    say("## Split 0 clean GBDT mistakes")
    say(f"test auc {gbdt_row['test_roc_auc']:.3f} log loss {gbdt_row['test_log_loss']:.3f}")
    order = np.argsort(-p)
    say("high P(delete) keeps:")
    shown = 0
    for index in order:
        if y_test[index] == 0:
            src = test_idx[index]
            say(
                f"  p={p[index]:.2f} {filenames[src]} ss={x[src, FEATURE_NAMES.index('screenshot')]:.0f} "
                f"person={x[src, FEATURE_NAMES.index('person_max_conf')]:.2f} "
                f"lap={x[src, FEATURE_NAMES.index('laplacian_var')]:.4f}"
            )
            shown += 1
            if shown == 8:
                break
    say("low P(delete) deletes:")
    shown = 0
    for index in order[::-1]:
        if y_test[index] == 1:
            src = test_idx[index]
            say(
                f"  p={p[index]:.2f} {filenames[src]} ss={x[src, FEATURE_NAMES.index('screenshot')]:.0f} "
                f"person={x[src, FEATURE_NAMES.index('person_max_conf')]:.2f} "
                f"lap={x[src, FEATURE_NAMES.index('laplacian_var')]:.4f}"
            )
            shown += 1
            if shown == 8:
                break
    del ids

    payload = {
        "diagnostics": diag,
        "horse_race": results,
        "permutation": perm_summary,
        "ablations": ablation_results,
        "thresholds": threshold_results,
        "gbdt_grid": grid,
        "learning_curve": curve,
        "detector_meta": {
            "detector": meta_json.get("detector"),
            "face_detector": meta_json.get("face_detector"),
            "elapsed_sec": meta_json.get("elapsed_sec"),
            "class_image_hits": meta_json.get("class_image_hits_640_conf0.25"),
        },
    }
    (args.out / "results.json").write_text(json.dumps(payload, indent=2))
    write_report(args.out / "summary.md", lines)
    say("")
    say(f"wrote {args.out / 'results.json'}")


if __name__ == "__main__":
    main()
