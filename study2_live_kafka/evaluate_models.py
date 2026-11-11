"""Study 2 Predictive Evaluation: Telemetry-Only (Model A) vs. Change-Aware (Model B).

Compares three strictly causal feature sets across 1-Hz operational windows:
  1. Model A (Telemetry-Only): 12 standard broker/service telemetry series
  2. Model B-Coarse (Telemetry + Naive Deployment Flags): adds 4 coarse binary flags
  3. Model B-Semantic (Telemetry + Compatibility & Queueing-Capacity Descriptors): adds 7 semantic descriptors
"""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize

if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

TELEMETRY_FEATURES = [
    "offered_rate",
    "fulfilled_rate",
    "obs_mean_lat",
    "obs_p95_lat",
    "orders_lag_total",
    "orders_lag_max_part",
    "orders_lag_skew",
    "downstream_lag",
    "inv_err_rate",
    "bill_err_rate",
    "retry_rate",
    "overdue_rate",
]

COARSE_CHANGE_FEATURES = [
    "change_active",
    "coarse_schema_flag",
    "coarse_routing_flag",
    "coarse_service_deploy_flag",
]

SEMANTIC_CHANGE_FEATURES = [
    "sem_schema_deficit",
    "sem_unsubscribed_topic",
    "sem_dep_probe_fail",
    "sem_compat_deficit",
    "sem_part_concentration",
    "sem_max_part_util",
    "sem_canary_overload",
]

FEATURE_SETS = {
    "Model_A_Telemetry": TELEMETRY_FEATURES,
    "Model_B_Coarse": TELEMETRY_FEATURES + COARSE_CHANGE_FEATURES,
    "Model_B_Semantic": TELEMETRY_FEATURES + SEMANTIC_CHANGE_FEATURES,
}


class L2LogisticClassifier:
    def __init__(self, reg_lambda: float = 1.0):
        self.reg_lambda = reg_lambda
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.w_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "L2LogisticClassifier":
        self.mean_ = X.mean(axis=0)
        self.std_ = np.where(X.std(axis=0) > 1e-8, X.std(axis=0), 1.0)
        Xs = (X - self.mean_) / self.std_
        Xb = np.hstack([np.ones((Xs.shape[0], 1)), Xs])
        d = Xb.shape[1]

        pos_ratio = max(1e-4, float(y.mean()))
        weights = np.where(y == 1, 0.5 / pos_ratio, 0.5 / max(1e-4, 1.0 - pos_ratio))

        def loss_and_grad(w: np.ndarray) -> Tuple[float, np.ndarray]:
            z = np.clip(Xb @ w, -30.0, 30.0)
            p = 1.0 / (1.0 + np.exp(-z))
            eps = 1e-12
            nll = -np.mean(weights * (y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps)))
            reg = 0.5 * self.reg_lambda * np.sum(w[1:] ** 2) / Xs.shape[0]
            grad = (Xb.T @ (weights * (p - y))) / Xs.shape[0]
            grad[1:] += self.reg_lambda * w[1:] / Xs.shape[0]
            return float(nll + reg), grad

        res = minimize(loss_and_grad, np.zeros(d), jac=True, method="L-BFGS-B")
        self.w_ = res.x
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = (X - self.mean_) / self.std_
        Xb = np.hstack([np.ones((Xs.shape[0], 1)), Xs])
        z = np.clip(Xb @ self.w_, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-z))


class SimpleDecisionTree:
    def __init__(self, max_depth: int = 5, min_samples_leaf: int = 6, max_features: int | None = None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.tree_: dict | None = None

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int, rng: np.random.Generator) -> dict:
        n, d = X.shape
        prob = float(y.mean()) if n > 0 else 0.0
        if depth >= self.max_depth or n < 2 * self.min_samples_leaf or prob in (0.0, 1.0):
            return {"leaf": True, "prob": prob}

        k = self.max_features or max(3, int(math.ceil(0.6 * d)))
        feat_idxs = rng.choice(d, size=min(d, k), replace=False)
        parent_gini = 2.0 * prob * (1.0 - prob)
        best_gain = 1e-7
        best_feat = -1
        best_thr = 0.0

        for f in feat_idxs:
            col = X[:, f]
            qs = np.unique(np.quantile(col, [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9]))
            for thr in qs:
                left = col <= thr
                n_l = int(left.sum())
                n_r = n - n_l
                if n_l < self.min_samples_leaf or n_r < self.min_samples_leaf:
                    continue
                p_l = float(y[left].mean())
                p_r = float(y[~left].mean())
                gini = (n_l / n) * 2.0 * p_l * (1.0 - p_l) + (n_r / n) * 2.0 * p_r * (1.0 - p_r)
                gain = parent_gini - gini
                if gain > best_gain:
                    best_gain = gain
                    best_feat = int(f)
                    best_thr = float(thr)

        if best_feat == -1:
            return {"leaf": True, "prob": prob}

        mask = X[:, best_feat] <= best_thr
        return {
            "leaf": False,
            "feat": best_feat,
            "thr": best_thr,
            "left": self._build(X[mask], y[mask], depth + 1, rng),
            "right": self._build(X[~mask], y[~mask], depth + 1, rng),
        }

    def fit(self, X: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> "SimpleDecisionTree":
        self.tree_ = self._build(X, y, 0, rng)
        return self

    def _pred_row(self, row: np.ndarray, node: dict) -> float:
        while not node["leaf"]:
            node = node["left"] if row[node["feat"]] <= node["thr"] else node["right"]
        return node["prob"]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._pred_row(X[i], self.tree_) for i in range(X.shape[0])], dtype=float)


class RandomForestClassifierNP:
    def __init__(self, n_estimators: int = 50, max_depth: int = 6, seed: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.seed = seed
        self.trees_: List[SimpleDecisionTree] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifierNP":
        rng = np.random.default_rng(self.seed)
        self.trees_ = []
        n = X.shape[0]
        for _ in range(self.n_estimators):
            idx = rng.integers(0, n, size=n)
            tree = SimpleDecisionTree(max_depth=self.max_depth, min_samples_leaf=5)
            tree.fit(X[idx], y[idx], rng)
            self.trees_.append(tree)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        preds = np.column_stack([t.predict_proba(X) for t in self.trees_])
        return preds.mean(axis=1)


def pr_auc_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    order = np.argsort(-y_prob, kind="mergesort")
    y_sorted = y_true[order]
    total_pos = float(y_true.sum())
    if total_pos == 0:
        return 0.0
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1.0 - y_sorted)
    precisions = np.concatenate([[1.0], tps / np.maximum(1.0, tps + fps)])
    recalls = np.concatenate([[0.0], tps / total_pos])
    return float(np.sum((recalls[1:] - recalls[:-1]) * precisions[1:]))


def roc_auc_score_np(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    order = np.argsort(-y_prob, kind="mergesort")
    y_sorted = y_true[order]
    pos = float(y_true.sum())
    neg = float(len(y_true) - pos)
    if pos == 0 or neg == 0:
        return 0.5
    tps = np.concatenate([[0.0], np.cumsum(y_sorted) / pos])
    fps = np.concatenate([[0.0], np.cumsum(1.0 - y_sorted) / neg])
    return float(np.trapezoid(tps, fps))


def pick_threshold_on_train(y_train: np.ndarray, p_train: np.ndarray) -> float:
    best_f1 = -1.0
    best_thr = 0.5
    for thr in np.linspace(0.20, 0.80, 61):
        pred = (p_train >= thr).astype(int)
        tp = float(np.sum((pred == 1) & (y_train == 1)))
        fp = float(np.sum((pred == 1) & (y_train == 0)))
        fn = float(np.sum((pred == 0) & (y_train == 1)))
        f1 = (2.0 * tp) / max(1.0, 2.0 * tp + fp + fn)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr


def compute_metrics(
    rows: List[dict],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    run_onset_lookup: Dict[str, Tuple[float, float | None]],
) -> dict:
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))

    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = (2.0 * prec * rec) / max(1e-12, prec + rec)
    prauc = pr_auc_score(y_true, y_prob)
    rocauc = roc_auc_score_np(y_true, y_prob)
    brier = float(np.mean((y_prob - y_true) ** 2))

    benign_change_sc = {"benign_schema_additive", "benign_routing_dual_sub", "benign_minor_slowdown"}
    benign_rollout_idx = [
        i for i, r in enumerate(rows)
        if r["scenario"] in benign_change_sc and (float(r["onset_t"]) - 1.0) <= float(r["t"]) <= (float(r["onset_t"]) + 5.0)
    ]
    benign_far = float(np.mean(y_pred[benign_rollout_idx])) if benign_rollout_idx else 0.0
    overall_fpr = fp / max(1, fp + tn)

    by_run: Dict[str, List[Tuple[float, int]]] = {}
    run_sc: Dict[str, str] = {}
    for i, r in enumerate(rows):
        rid = r["run_id"]
        by_run.setdefault(rid, []).append((float(r["t"]), int(y_pred[i])))
        run_sc[rid] = r["scenario"]

    lead_sla_all: List[float] = []
    lead_by_sc: Dict[str, List[float]] = {}
    detected_harmful_runs = 0
    total_harmful_runs = 0

    for rid, seq in by_run.items():
        sc = run_sc[rid]
        onset_t, deg_t = run_onset_lookup[rid]
        if deg_t is None or sc.startswith("benign_") or sc == "baseline_normal":
            continue
        total_harmful_runs += 1
        seq.sort(key=lambda x: x[0])
        alerts = [t for t, pred in seq if pred == 1 and t >= math.floor(onset_t - 1.0)]
        if alerts:
            detected_harmful_runs += 1
            t_alert = alerts[0]
            sla_manifest_t = deg_t + 2.0
            lead_s = sla_manifest_t - t_alert
            lead_sla_all.append(lead_s)
            lead_by_sc.setdefault(sc, []).append(lead_s)

    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "pr_auc": round(prauc, 4),
        "roc_auc": round(rocauc, 4),
        "brier": round(brier, 4),
        "overall_fpr": round(overall_fpr, 4),
        "benign_rollout_far": round(benign_far, 4),
        "harmful_run_detection_rate": round(detected_harmful_runs / max(1, total_harmful_runs), 4),
        "mean_lead_before_sla_s": round(statistics.mean(lead_sla_all), 3) if lead_sla_all else 0.0,
        "lead_by_scenario_s": {k: round(statistics.mean(v), 3) for k, v in sorted(lead_by_sc.items())},
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def evaluate_all(data_dir: str) -> dict:
    windows_path = os.path.join(data_dir, "study2_windows.csv")
    summary_path = os.path.join(data_dir, "study2_runs_summary.json")

    with open(windows_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)
    run_onset_lookup = {
        r["run_id"]: (float(r["onset_t"]), r["degradation_onset_t"])
        for r in summary["runs"]
    }

    y_all = np.array([int(r["y_degradation"]) for r in rows], dtype=int)

    fold_of_run: Dict[str, int] = {}
    max_reps = max(1, max(int(x["run_id"].rsplit("-r", 1)[1]) + 1 for x in summary["runs"]))
    n_folds = min(5, max(2, max_reps))
    for r in summary["runs"]:
        rid = r["run_id"]
        rep_num = int(rid.rsplit("-r", 1)[1])
        fold_of_run[rid] = rep_num % n_folds
    fold_arr = np.array([fold_of_run[r["run_id"]] for r in rows], dtype=int)

    cv_results: Dict[str, dict] = {}
    pr_curves: Dict[str, dict] = {}

    for model_family in ("RandomForest", "LogisticRegression"):
        cv_results[model_family] = {}
        for fset_name, cols in FEATURE_SETS.items():
            X_all = np.array([[float(r[c]) for c in cols] for r in rows], dtype=float)
            oof_prob = np.zeros(len(rows), dtype=float)
            oof_pred = np.zeros(len(rows), dtype=int)
            fold_f1s: List[float] = []
            fold_praucs: List[float] = []

            n_folds_actual = int(fold_arr.max()) + 1
            for k in range(n_folds_actual):
                tr = fold_arr != k
                te = fold_arr == k
                if model_family == "RandomForest":
                    clf = RandomForestClassifierNP(n_estimators=50, max_depth=6, seed=100 + k)
                else:
                    clf = L2LogisticClassifier(reg_lambda=1.0)
                clf.fit(X_all[tr], y_all[tr])
                p_tr = clf.predict_proba(X_all[tr])
                thr = pick_threshold_on_train(y_all[tr], p_tr)
                p_te = clf.predict_proba(X_all[te])
                pred_te = (p_te >= thr).astype(int)
                oof_prob[te] = p_te
                oof_pred[te] = pred_te

                tp_k = np.sum((pred_te == 1) & (y_all[te] == 1))
                fp_k = np.sum((pred_te == 1) & (y_all[te] == 0))
                fn_k = np.sum((pred_te == 0) & (y_all[te] == 1))
                fold_f1s.append(float((2.0 * tp_k) / max(1.0, 2.0 * tp_k + fp_k + fn_k)))
                fold_praucs.append(pr_auc_score(y_all[te], p_te))

            agg = compute_metrics(rows, y_all, oof_prob, oof_pred, run_onset_lookup)
            agg["fold_f1_mean"] = round(statistics.mean(fold_f1s), 4)
            agg["fold_f1_sd"] = round(statistics.stdev(fold_f1s), 4) if len(fold_f1s) > 1 else 0.0
            agg["fold_prauc_mean"] = round(statistics.mean(fold_praucs), 4)
            agg["fold_prauc_sd"] = round(statistics.stdev(fold_praucs), 4) if len(fold_praucs) > 1 else 0.0
            cv_results[model_family][fset_name] = agg

            if model_family == "RandomForest":
                order = np.argsort(-oof_prob, kind="mergesort")
                ys = y_all[order]
                tps = np.cumsum(ys)
                fps = np.cumsum(1.0 - ys)
                prec_c = np.concatenate([[1.0], tps / np.maximum(1.0, tps + fps)])
                rec_c = np.concatenate([[0.0], tps / max(1.0, float(y_all.sum()))])
                step = max(1, len(prec_c) // 150)
                pr_curves[fset_name] = {
                    "recall": [round(float(x), 4) for x in rec_c[::step]] + [round(float(rec_c[-1]), 4)],
                    "precision": [round(float(x), 4) for x in prec_c[::step]] + [round(float(prec_c[-1]), 4)],
                }

    harmful_scenarios = [
        "harmful_consumer_slowdown",
        "harmful_partition_hotspot",
        "harmful_dependency_outage",
        "harmful_routing_unsubscribed",
        "harmful_schema_breaking",
    ]
    sc_arr = np.array([r["scenario"] for r in rows])
    healthy_mask = np.array([int(r["is_harmful_scenario"]) == 0 for r in rows])

    loso_results: Dict[str, dict] = {}
    for model_family in ("LogisticRegression", "RandomForest"):
        loso_results[model_family] = {}
        for fset_name, cols in FEATURE_SETS.items():
            X_all = np.array([[float(r[c]) for c in cols] for r in rows], dtype=float)
            per_heldout: Dict[str, dict] = {}
            for held_sc in harmful_scenarios:
                te = (sc_arr == held_sc) | (healthy_mask & (fold_arr == int(fold_arr.max())))
                tr = ~te & (sc_arr != held_sc)
                if model_family == "LogisticRegression":
                    clf = L2LogisticClassifier(reg_lambda=1.0)
                else:
                    clf = RandomForestClassifierNP(n_estimators=50, max_depth=6, seed=303)
                clf.fit(X_all[tr], y_all[tr])
                thr = pick_threshold_on_train(y_all[tr], clf.predict_proba(X_all[tr]))
                p_te = clf.predict_proba(X_all[te])
                pred_te = (p_te >= thr).astype(int)
                sub_rows = [rows[i] for i in range(len(rows)) if te[i]]
                m = compute_metrics(sub_rows, y_all[te], p_te, pred_te, run_onset_lookup)
                per_heldout[held_sc] = {
                    "pr_auc": m["pr_auc"],
                    "f1": m["f1"],
                    "recall": m["recall"],
                    "precision": m["precision"],
                    "mean_lead_before_sla_s": m["mean_lead_before_sla_s"],
                }
            loso_results[model_family][fset_name] = {
                "mean_pr_auc": round(statistics.mean(v["pr_auc"] for v in per_heldout.values()), 4),
                "mean_f1": round(statistics.mean(v["f1"] for v in per_heldout.values()), 4),
                "per_scenario": per_heldout,
            }

    out_payload = {
        "run_grouped_5fold_cv": cv_results,
        "leave_one_scenario_out": loso_results,
        "pr_curves_rf": pr_curves,
    }
    out_path = os.path.join(data_dir, "study2_eval_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)
    return out_payload


if __name__ == "__main__":
    ddir = os.path.join(os.path.dirname(__file__), "data")
    res = evaluate_all(ddir if os.path.exists(ddir) else os.path.dirname(__file__))
    print("=== 5-Fold Run-Grouped CV ===")
    for mf, mdict in res["run_grouped_5fold_cv"].items():
        print(f"\n[{mf}]")
        for fn, m in mdict.items():
            print(
                f"  {fn:20s} | PR-AUC={m['pr_auc']:.4f} | F1={m['f1']:.4f} (Prec={m['precision']:.4f}, Rec={m['recall']:.4f}) "
                f"| BenignRolloutFAR={m['benign_rollout_far']:.4f} | Lead={m['mean_lead_before_sla_s']:.2f}s"
            )
            print("     Lead by sc:", m["lead_by_scenario_s"])
    print("\n=== Leave-One-Harmful-Scenario-Out (LOSO) ===")
    for mf, mdict in res["leave_one_scenario_out"].items():
        print(f"[{mf}]", {k: (v["mean_pr_auc"], v["mean_f1"]) for k, v in mdict.items()})
