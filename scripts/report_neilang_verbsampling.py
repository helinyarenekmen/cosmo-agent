"""
Generates a markdown report from neilang verbalized-sampling prediction results.
Usage: python3 report_neilang_verbsampling.py [csv_file]
If no file is given, the latest neilang_verbsampling_*.csv in exports/ is used.

Extra plots vs. the standard report:
  - Side-by-side comparison: sampled vs argmax confusion matrices
  - Mean predicted probability per GT class (calibration view)
  - Prediction entropy distribution
"""

import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

BASE_DIR     = Path(__file__).parent.parent
EXPORTS_DIR  = BASE_DIR / "exports"
REPORTS_DIR  = BASE_DIR / "reports"
FIGURES_DIR  = BASE_DIR / "figures"

LABEL_MAP   = {1: "Not bothered (1)", 2: "A little (2)", 3: "Very bothered (3)"}
LABEL_SHORT = {1: "None (1)", 2: "A bit (2)", 3: "Very (3)"}
COLORS      = {"gt": "#4C72B0", "sampled": "#DD8452", "argmax": "#55A868",
               "match": "#55A868", "no_match": "#C44E52"}

PLOTS_DIR: Path = None  # set in main()


# ─── Data ────────────────────────────────────────────────────────────────────

def load_latest_csv() -> Path:
    candidates = sorted(EXPORTS_DIR.glob("neilang_verbsampling_*.csv"))
    if not candidates:
        raise FileNotFoundError("No neilang_verbsampling_*.csv found in exports/")
    return candidates[-1]


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in ["gt_neilang", "sampled_pred", "argmax_pred",
                "match_sampled", "match_argmax", "p1", "p2", "p3", "age"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(df: pd.DataFrame, pred_col: str, match_col: str) -> dict:
    valid  = df.dropna(subset=["gt_neilang", pred_col, match_col])
    labels = sorted(set(valid["gt_neilang"].astype(int)) | set(valid[pred_col].astype(int)))
    gt     = valid["gt_neilang"].astype(int).tolist()
    pred   = valid[pred_col].astype(int).tolist()

    cm = {(g, p): sum(1 for a, b in zip(gt, pred) if a == g and b == p)
          for g in labels for p in labels}

    per_class = {}
    for lbl in labels:
        tp = cm.get((lbl, lbl), 0)
        fp = sum(cm.get((g, lbl), 0) for g in labels if g != lbl)
        fn = sum(cm.get((lbl, p), 0) for p in labels if p != lbl)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[lbl] = {"tp": tp, "fp": fp, "fn": fn,
                          "precision": prec, "recall": rec, "f1": f1,
                          "support": sum(1 for g in gt if g == lbl)}

    n        = len(valid)
    acc      = valid[match_col].sum() / n if n else 0.0
    macro_f1 = sum(per_class[l]["f1"] for l in labels) / len(labels)
    w_f1     = sum(per_class[l]["f1"] * per_class[l]["support"] for l in labels) / n if n else 0.0

    return {
        "total": len(df), "valid": n,
        "parse_fail": int(df[pred_col].isna().sum()),
        "accuracy": acc, "macro_f1": macro_f1, "weighted_f1": w_f1,
        "cm": cm, "per_class": per_class, "labels": labels,
        "gt_dist":   dict(Counter(gt)),
        "pred_dist": dict(Counter(pred)),
    }


# ─── Plot helpers ─────────────────────────────────────────────────────────────

def save_fig(fig, name: str) -> str:
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return f"../figures/{PLOTS_DIR.name}/{name}"


def img_tag(rel_path: str, alt: str = "") -> str:
    return f"![{alt}]({rel_path})"


# ─── Plots ───────────────────────────────────────────────────────────────────

def plot_confusion_matrix(metrics: dict, title: str, fname: str) -> str:
    labels = metrics["labels"]
    cm     = metrics["cm"]
    matrix = np.array([[cm.get((g, p), 0) for p in labels] for g in labels])
    total  = matrix.sum()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    plt.colorbar(im, ax=ax)
    tick_labels = [LABEL_SHORT[l] for l in labels]
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(tick_labels, fontsize=10)
    ax.set_yticklabels(tick_labels, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Ground Truth", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    for i in range(len(labels)):
        for j in range(len(labels)):
            val   = matrix[i, j]
            color = "white" if matrix.max() and val > matrix.max() * 0.6 else "black"
            ax.text(j, i, f"{val}\n({val/total*100:.1f}%)",
                    ha="center", va="center", fontsize=10, color=color, fontweight="bold")
    plt.tight_layout()
    return save_fig(fig, fname)


def plot_confusion_normalised(metrics: dict, title: str, fname: str) -> str:
    labels = metrics["labels"]
    cm     = metrics["cm"]
    matrix = np.array([[cm.get((g, p), 0) for p in labels] for g in labels], dtype=float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    norm = np.where(row_sums != 0, matrix / np.where(row_sums != 0, row_sums, 1), 0.0)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(norm, cmap="YlOrRd", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Row-normalised proportion")
    tick_labels = [LABEL_SHORT[l] for l in labels]
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(tick_labels, fontsize=10)
    ax.set_yticklabels(tick_labels, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Ground Truth", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{norm[i,j]:.2f}", ha="center", va="center", fontsize=11,
                    color="white" if norm[i, j] > 0.6 else "black", fontweight="bold")
    plt.tight_layout()
    return save_fig(fig, fname)


def plot_distribution(df: pd.DataFrame, m_smp: dict, m_arg: dict) -> str:
    labels  = m_smp["labels"]
    gt_dist = m_smp["gt_dist"]
    x, w = np.arange(len(labels)), 0.25

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w,   [gt_dist.get(l, 0)            for l in labels], w,
           label="Ground Truth", color=COLORS["gt"],      alpha=0.85)
    ax.bar(x,       [m_smp["pred_dist"].get(l, 0) for l in labels], w,
           label="Sampled",      color=COLORS["sampled"], alpha=0.85)
    ax.bar(x + w,   [m_arg["pred_dist"].get(l, 0) for l in labels], w,
           label="Argmax",       color=COLORS["argmax"],  alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL_MAP[l] for l in labels], fontsize=9)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Ground Truth vs Sampled vs Argmax Distribution", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout()
    return save_fig(fig, "distribution.png")


def plot_mean_probs_by_gt(df: pd.DataFrame) -> str:
    """Mean predicted probability for each class, grouped by GT label."""
    valid  = df.dropna(subset=["gt_neilang", "p1", "p2", "p3"])
    labels = sorted(valid["gt_neilang"].astype(int).unique())
    p_cols = ["p1", "p2", "p3"]
    means  = valid.groupby("gt_neilang")[p_cols].mean()

    x, w = np.arange(len(p_cols)), 0.2
    fig, ax = plt.subplots(figsize=(7, 4))
    palette = ["#4C72B0", "#DD8452", "#55A868"]
    for i, lbl in enumerate(labels):
        vals = [means.loc[lbl, c] if lbl in means.index else 0 for c in p_cols]
        ax.bar(x + (i - 1) * w, vals, w,
               label=f"GT: {LABEL_SHORT[lbl]}", color=palette[i % 3], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"p({LABEL_SHORT[j+1]})" for j in range(3)], fontsize=10)
    ax.set_ylabel("Mean predicted probability", fontsize=11)
    ax.set_title("Mean Predicted Probabilities by GT Class", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    return save_fig(fig, "mean_probs_by_gt.png")


def plot_entropy(df: pd.DataFrame) -> str:
    """Distribution of prediction entropy per GT class."""
    valid = df.dropna(subset=["gt_neilang", "p1", "p2", "p3"]).copy()
    eps   = 1e-9
    valid["entropy"] = -(
        valid["p1"] * np.log2(valid["p1"] + eps) +
        valid["p2"] * np.log2(valid["p2"] + eps) +
        valid["p3"] * np.log2(valid["p3"] + eps)
    )

    labels  = sorted(valid["gt_neilang"].astype(int).unique())
    palette = ["#4C72B0", "#DD8452", "#55A868"]

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, lbl in enumerate(labels):
        subset = valid[valid["gt_neilang"] == lbl]["entropy"]
        ax.hist(subset, bins=20, alpha=0.55,
                label=LABEL_SHORT[lbl], color=palette[i % 3], edgecolor="white")

    ax.axvline(valid["entropy"].mean(), color="black", linestyle="--", linewidth=1.2,
               label=f"Overall mean ({valid['entropy'].mean():.2f} bits)")
    ax.set_xlabel("Prediction Entropy (bits)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Prediction Entropy Distribution by GT Class", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return save_fig(fig, "entropy.png")


def plot_accuracy_by_group(df: pd.DataFrame, col: str, match_col: str,
                           title: str, fname: str) -> str:
    valid  = df.dropna(subset=[match_col, col]).copy()
    valid[col] = valid[col].astype(str)
    groups = (valid.groupby(col)[match_col]
              .agg(["mean", "count"])
              .reset_index()
              .rename(columns={"mean": "accuracy", "count": "n"})
              .sort_values("accuracy", ascending=True))

    fig, ax = plt.subplots(figsize=(7, max(3, len(groups) * 0.45)))
    colors = [COLORS["match"] if v >= 0.5 else COLORS["no_match"] if v < 0.35 else "#F0A500"
              for v in groups["accuracy"]]
    bars = ax.barh(groups[col], groups["accuracy"], color=colors, alpha=0.85)
    for bar, (_, row) in zip(bars, groups.iterrows()):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{row['accuracy']:.3f}  (n={int(row['n'])})", va="center", fontsize=9)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Accuracy", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axvline(groups["accuracy"].mean(), color="gray", linestyle="--",
               linewidth=1, label="Mean")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return save_fig(fig, fname)


def plot_accuracy_by_age(df: pd.DataFrame, match_col: str, title: str, fname: str) -> str:
    valid = df.dropna(subset=[match_col, "age"]).copy()
    bins       = [18, 25, 35, 45, 55, 65, 100]
    age_labels = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    valid["age_group"] = pd.cut(valid["age"], bins=bins, labels=age_labels, right=False)
    groups = valid.groupby("age_group", observed=True)[match_col].agg(["mean", "count"]).reset_index()

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [COLORS["match"] if v >= 0.5 else COLORS["no_match"] if v < 0.35 else "#F0A500"
              for v in groups["mean"]]
    bars = ax.bar(groups["age_group"].astype(str), groups["mean"], color=colors, alpha=0.85)
    for bar, (_, row) in zip(bars, groups.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{row['mean']:.3f}\n(n={int(row['count'])})",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("Age Group", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axhline(valid[match_col].mean(), color="gray", linestyle="--",
               linewidth=1, label="Overall mean")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return save_fig(fig, fname)


# ─── Report ──────────────────────────────────────────────────────────────────

def build_report(df: pd.DataFrame, m_smp: dict, m_arg: dict,
                 csv_path: Path, ts: str) -> str:

    print("  confusion matrix (sampled)...")
    p_cm_smp  = plot_confusion_matrix(m_smp, "Confusion Matrix — Sampled",
                                      "confusion_matrix_sampled.png")
    print("  confusion matrix (argmax)...")
    p_cm_arg  = plot_confusion_matrix(m_arg, "Confusion Matrix — Argmax",
                                      "confusion_matrix_argmax.png")
    print("  normalised matrix (sampled)...")
    p_norm_smp = plot_confusion_normalised(m_smp,
                                           "Normalised Confusion Matrix — Sampled\n(row = GT class)",
                                           "confusion_matrix_normalised_sampled.png")
    print("  normalised matrix (argmax)...")
    p_norm_arg = plot_confusion_normalised(m_arg,
                                           "Normalised Confusion Matrix — Argmax\n(row = GT class)",
                                           "confusion_matrix_normalised_argmax.png")
    print("  distribution...")
    p_dist    = plot_distribution(df, m_smp, m_arg)
    print("  mean probs by GT...")
    p_probs   = plot_mean_probs_by_gt(df)
    print("  entropy...")
    p_entropy = plot_entropy(df)
    print("  accuracy by region (argmax)...")
    p_region  = plot_accuracy_by_group(df, "region", "match_argmax",
                                       "Accuracy by Region (Argmax)",
                                       "accuracy_by_region.png")
    print("  accuracy by gender (argmax)...")
    p_gender  = plot_accuracy_by_group(df, "gender", "match_argmax",
                                       "Accuracy by Gender (Argmax)",
                                       "accuracy_by_gender.png")
    print("  accuracy by age (argmax)...")
    p_age     = plot_accuracy_by_age(df, "match_argmax",
                                     "Accuracy by Age Group (Argmax)",
                                     "accuracy_by_age.png")

    def _cm_table(m: dict) -> str:
        labels = m["labels"]
        header = "| | " + " | ".join(f"**Pred {LABEL_SHORT[l]}**" for l in labels) + " |"
        sep    = "|---|" + "|".join("---" for _ in labels) + "|"
        rows   = "\n".join(
            "| **GT " + LABEL_SHORT[g] + "** | " +
            " | ".join(str(m["cm"].get((g, p), 0)) for p in labels) + " |"
            for g in labels
        )
        return f"{header}\n{sep}\n{rows}"

    def _pc_table(m: dict) -> str:
        labels = m["labels"]
        pc = m["per_class"]
        n  = m["valid"]
        macro_p = sum(pc[l]["precision"] for l in labels) / len(labels)
        macro_r = sum(pc[l]["recall"]    for l in labels) / len(labels)
        w_p     = sum(pc[l]["precision"] * pc[l]["support"] for l in labels) / n
        w_r     = sum(pc[l]["recall"]    * pc[l]["support"] for l in labels) / n
        rows = "\n".join(
            f"| {LABEL_MAP[l]} | {pc[l]['support']} | "
            f"{pc[l]['precision']:.4f} | {pc[l]['recall']:.4f} | {pc[l]['f1']:.4f} |"
            for l in labels
        )
        return (
            "| Class | Support | Precision | Recall | F1 |\n"
            "|---|---|---|---|---|\n"
            f"{rows}\n"
            f"| **Macro avg** | {n} | {macro_p:.4f} | {macro_r:.4f} | {m['macro_f1']:.4f} |\n"
            f"| **Weighted avg** | {n} | {w_p:.4f} | {w_r:.4f} | {m['weighted_f1']:.4f} |"
        )

    region_rows = "\n".join(
        f"| {r['region']} | {int(r['count'])} | {r['mean']:.4f} |"
        for _, r in (df.dropna(subset=["match_argmax", "region"])
                     .groupby("region")["match_argmax"]
                     .agg(["mean", "count"]).reset_index()
                     .sort_values("mean", ascending=False)).iterrows()
    )
    gender_rows = "\n".join(
        f"| {r['gender']} | {int(r['count'])} | {r['mean']:.4f} |"
        for _, r in (df.dropna(subset=["match_argmax", "gender"])
                     .groupby("gender")["match_argmax"]
                     .agg(["mean", "count"]).reset_index()
                     .sort_values("mean", ascending=False)).iterrows()
    )

    # Average probability distribution
    vp = df.dropna(subset=["p1", "p2", "p3"])
    avg_p1 = vp["p1"].mean()
    avg_p2 = vp["p2"].mean()
    avg_p3 = vp["p3"].mean()

    # Entropy stats
    eps = 1e-9
    ent = -(vp["p1"] * np.log2(vp["p1"] + eps) +
            vp["p2"] * np.log2(vp["p2"] + eps) +
            vp["p3"] * np.log2(vp["p3"] + eps))
    high_conf = (ent < 0.5).sum()

    return f"""# neilang Prediction Report — Verbalized Sampling

**Model:** gpt-5.4-mini | **Temperature:** 0.8 | **Method:** Verbalized Sampling | **Date:** {ts}
**Source:** `{csv_path.name}`
**Prompt cleaning:** sentences revealing `neilang` (language-neighbor) and `neirelg` (religion-neighbor) attitudes removed.

> **Verbalized Sampling:** instead of predicting a single label, the model outputs a probability
> distribution over all classes (p1, p2, p3). Two predictions are derived:
> **Argmax** (deterministic — highest probability wins) and
> **Sampled** (stochastic — label drawn from the distribution).

---

## 1. Overall Performance

| Metric | Sampled | Argmax |
|---|---|---|
| Total personas | {m_smp['total']} | {m_arg['total']} |
| Valid predictions | {m_smp['valid']} | {m_arg['valid']} |
| Parse failures | {m_smp['parse_fail']} | {m_arg['parse_fail']} |
| **Accuracy** | **{m_smp['accuracy']:.4f}** | **{m_arg['accuracy']:.4f}** |
| Macro F1 | {m_smp['macro_f1']:.4f} | {m_arg['macro_f1']:.4f} |
| Weighted F1 | {m_smp['weighted_f1']:.4f} | {m_arg['weighted_f1']:.4f} |

---

## 2. Distribution: Ground Truth vs Sampled vs Argmax

{img_tag(p_dist, "Distribution")}

---

## 3. Predicted Probability Analysis

### 3a. Mean Predicted Probabilities by GT Class

{img_tag(p_probs, "Mean Probs by GT")}

> If the model is well-calibrated, the mean p(k) should be highest when GT = k.

| | Mean p(None=1) | Mean p(A little=2) | Mean p(Very=3) |
|---|---|---|---|
| Overall | {avg_p1:.4f} | {avg_p2:.4f} | {avg_p3:.4f} |

### 3b. Prediction Entropy

{img_tag(p_entropy, "Entropy Distribution")}

> Low entropy = high-confidence prediction. High entropy = uncertain.
> Max entropy for 3 classes = log₂(3) ≈ 1.585 bits.

| Metric | Value |
|---|---|
| Mean entropy | {ent.mean():.4f} bits |
| Median entropy | {ent.median():.4f} bits |
| High-confidence predictions (entropy < 0.5) | {high_conf} ({high_conf/len(vp)*100:.1f}%) |

---

## 4. Confusion Matrices

### 4a. Sampled Prediction

{img_tag(p_cm_smp, "Confusion Matrix Sampled")}

{_cm_table(m_smp)}

### 4b. Argmax Prediction

{img_tag(p_cm_arg, "Confusion Matrix Argmax")}

{_cm_table(m_arg)}

---

## 5. Normalised Confusion Matrices

### 5a. Sampled

{img_tag(p_norm_smp, "Normalised Confusion Matrix Sampled")}

### 5b. Argmax

{img_tag(p_norm_arg, "Normalised Confusion Matrix Argmax")}

> Row-normalised: shows what the model predicts *given* the true class.

---

## 6. Per-class Metrics

### 6a. Sampled

{_pc_table(m_smp)}

### 6b. Argmax

{_pc_table(m_arg)}

---

## 7. Accuracy by Region (Argmax)

{img_tag(p_region, "Accuracy by Region")}

| Region | N | Accuracy |
|---|---|---|
{region_rows}

---

## 8. Accuracy by Gender (Argmax)

{img_tag(p_gender, "Accuracy by Gender")}

| Gender | N | Accuracy |
|---|---|---|
{gender_rows}

---

## 9. Accuracy by Age Group (Argmax)

{img_tag(p_age, "Accuracy by Age Group")}

---

## 10. Notes

- Verbalized sampling yields a richer output than direct prediction: the probability
  distribution captures model uncertainty rather than a single hard label.
- **Argmax** is typically more accurate than **Sampled** since it always picks the
  most probable class, while sampling introduces stochastic noise.
- **Mean entropy** of {ent.mean():.3f} bits (max = 1.585) indicates the model's average
  confidence level across all personas.
- Parse failures: **{m_smp['parse_fail']}** personas
  (`{m_smp['parse_fail']/m_smp['total']*100:.1f}%`) — model did not return valid p1/p2/p3 JSON.
"""


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else load_latest_csv()
    print(f"Source: {csv_path}")

    df    = load_data(csv_path)
    m_smp = compute_metrics(df, "sampled_pred", "match_sampled")
    m_arg = compute_metrics(df, "argmax_pred",  "match_argmax")
    ts      = datetime.now().strftime("%Y-%m-%d %H:%M")
    ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")

    global PLOTS_DIR
    REPORTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR = FIGURES_DIR / f"neilang_verbsampling_plots_{ts_file}"
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating plots...")
    report = build_report(df, m_smp, m_arg, csv_path, ts)

    report_path = REPORTS_DIR / f"neilang_verbsampling_report_{ts_file}.md"
    report_path.write_text(report, encoding="utf-8")

    print(f"\nReport → {report_path}")
    print(f"Plots  → {PLOTS_DIR}/")
    print(f"Sampled  — Accuracy: {m_smp['accuracy']:.4f} | Macro F1: {m_smp['macro_f1']:.4f}")
    print(f"Argmax   — Accuracy: {m_arg['accuracy']:.4f} | Macro F1: {m_arg['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
