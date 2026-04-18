"""
Generates a markdown report from womenwork prediction results.
Usage: python3 report_womenwork.py [csv_file]
If no file is given, the latest womenwork_predictions_*.csv in exports/ is used.
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

LABEL_MAP = {
    1: "Strongly disagree (1)",
    2: "Disagree (2)",
    3: "Neither (3)",
    4: "Agree (4)",
    5: "Strongly agree (5)",
}
LABEL_SHORT = {1: "SD(1)", 2: "D(2)", 3: "N(3)", 4: "A(4)", 5: "SA(5)"}
COLORS = {"gt": "#4C72B0", "pred": "#DD8452", "match": "#55A868", "no_match": "#C44E52"}

PLOTS_DIR: Path = None  # set in main()


# ─── Data ────────────────────────────────────────────────────────────────────

def load_latest_csv() -> Path:
    candidates = sorted(EXPORTS_DIR.glob("womenwork_predictions_*.csv"))
    if not candidates:
        raise FileNotFoundError("No womenwork_predictions_*.csv found in exports/")
    return candidates[-1]


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in ["gt_womenwork", "pred_womenwork", "match", "age"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(df: pd.DataFrame) -> dict:
    valid  = df.dropna(subset=["gt_womenwork", "pred_womenwork", "match"])
    labels = sorted(set(valid["gt_womenwork"].astype(int)) | set(valid["pred_womenwork"].astype(int)))
    gt     = valid["gt_womenwork"].astype(int).tolist()
    pred   = valid["pred_womenwork"].astype(int).tolist()

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
    acc      = valid["match"].sum() / n if n else 0.0
    macro_f1 = sum(per_class[l]["f1"] for l in labels) / len(labels)
    w_f1     = sum(per_class[l]["f1"] * per_class[l]["support"] for l in labels) / n if n else 0.0

    return {
        "total": len(df), "valid": n,
        "parse_fail": int(df["pred_womenwork"].isna().sum()),
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

def plot_confusion_matrix(metrics: dict) -> str:
    labels = metrics["labels"]
    cm     = metrics["cm"]
    matrix = np.array([[cm.get((g, p), 0) for p in labels] for g in labels])
    total  = matrix.sum()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="Blues")
    plt.colorbar(im, ax=ax)
    tick_labels = [LABEL_SHORT[l] for l in labels]
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(tick_labels, fontsize=10)
    ax.set_yticklabels(tick_labels, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Ground Truth", fontsize=11)
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
    for i in range(len(labels)):
        for j in range(len(labels)):
            val   = matrix[i, j]
            color = "white" if matrix.max() and val > matrix.max() * 0.6 else "black"
            ax.text(j, i, f"{val}\n({val/total*100:.1f}%)",
                    ha="center", va="center", fontsize=9, color=color, fontweight="bold")
    plt.tight_layout()
    return save_fig(fig, "confusion_matrix.png")


def plot_confusion_normalised(metrics: dict) -> str:
    labels = metrics["labels"]
    cm     = metrics["cm"]
    matrix = np.array([[cm.get((g, p), 0) for p in labels] for g in labels], dtype=float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    norm = np.where(row_sums != 0, matrix / np.where(row_sums != 0, row_sums, 1), 0.0)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(norm, cmap="YlOrRd", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Row-normalised proportion")
    tick_labels = [LABEL_SHORT[l] for l in labels]
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(tick_labels, fontsize=10)
    ax.set_yticklabels(tick_labels, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Ground Truth", fontsize=11)
    ax.set_title("Normalised Confusion Matrix\n(row = GT class)", fontsize=12, fontweight="bold")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{norm[i,j]:.2f}", ha="center", va="center", fontsize=10,
                    color="white" if norm[i, j] > 0.6 else "black", fontweight="bold")
    plt.tight_layout()
    return save_fig(fig, "confusion_matrix_normalised.png")


def plot_distribution(metrics: dict) -> str:
    labels    = metrics["labels"]
    gt_dist   = metrics["gt_dist"]
    pred_dist = metrics["pred_dist"]
    x, w = np.arange(len(labels)), 0.35

    fig, ax = plt.subplots(figsize=(9, 4))
    bars_gt   = ax.bar(x - w/2, [gt_dist.get(l, 0)   for l in labels], w,
                       label="Ground Truth", color=COLORS["gt"],   alpha=0.85)
    bars_pred = ax.bar(x + w/2, [pred_dist.get(l, 0) for l in labels], w,
                       label="Prediction",   color=COLORS["pred"], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL_MAP[l] for l in labels], fontsize=8)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Ground Truth vs Prediction Distribution", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    for bar in list(bars_gt) + list(bars_pred):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                str(int(bar.get_height())), ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    return save_fig(fig, "distribution.png")


def plot_accuracy_by_group(df: pd.DataFrame, col: str, title: str) -> str:
    valid  = df.dropna(subset=["match", col]).copy()
    valid[col] = valid[col].astype(str)
    groups = (valid.groupby(col)["match"]
              .agg(["mean", "count"])
              .reset_index()
              .rename(columns={"mean": "accuracy", "count": "n"})
              .sort_values("accuracy", ascending=True))

    fig, ax = plt.subplots(figsize=(7, max(3, len(groups) * 0.45)))
    colors = [COLORS["match"] if v >= 0.35 else COLORS["no_match"] if v < 0.20 else "#F0A500"
              for v in groups["accuracy"]]
    bars = ax.barh(groups[col], groups["accuracy"], color=colors, alpha=0.85)
    for bar, (_, row) in zip(bars, groups.iterrows()):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{row['accuracy']:.3f}  (n={int(row['n'])})", va="center", fontsize=9)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Accuracy", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axvline(groups["accuracy"].mean(), color="gray", linestyle="--", linewidth=1, label="Mean")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return save_fig(fig, f"accuracy_by_{col}.png")


def plot_accuracy_by_age(df: pd.DataFrame) -> str:
    valid = df.dropna(subset=["match", "age"]).copy()
    bins       = [18, 25, 35, 45, 55, 65, 100]
    age_labels = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    valid["age_group"] = pd.cut(valid["age"], bins=bins, labels=age_labels, right=False)
    groups = valid.groupby("age_group", observed=True)["match"].agg(["mean", "count"]).reset_index()

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [COLORS["match"] if v >= 0.35 else COLORS["no_match"] if v < 0.20 else "#F0A500"
              for v in groups["mean"]]
    bars = ax.bar(groups["age_group"].astype(str), groups["mean"], color=colors, alpha=0.85)
    for bar, (_, row) in zip(bars, groups.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{row['mean']:.3f}\n(n={int(row['count'])})",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("Age Group", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("Accuracy by Age Group", fontsize=13, fontweight="bold")
    ax.axhline(valid["match"].mean(), color="gray", linestyle="--", linewidth=1, label="Overall mean")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return save_fig(fig, "accuracy_by_age.png")


def plot_mean_pred_by_gt(df: pd.DataFrame, metrics: dict) -> str:
    """For each GT class, show the mean predicted class — reveals systematic bias direction."""
    valid  = df.dropna(subset=["gt_womenwork", "pred_womenwork"])
    labels = metrics["labels"]
    means  = valid.groupby("gt_womenwork")["pred_womenwork"].mean()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([str(l) for l in labels],
           [means.get(l, 0) for l in labels],
           color=COLORS["gt"], alpha=0.85)
    ax.plot([str(l) for l in labels], labels, "o--",
            color=COLORS["pred"], label="Perfect prediction (GT=Pred)", linewidth=1.5)
    ax.set_xlabel("Ground Truth Class", fontsize=11)
    ax.set_ylabel("Mean Predicted Class", fontsize=11)
    ax.set_title("Mean Predicted Class per GT Class\n(bars above line = over-prediction, below = under-prediction)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0.5, 5.5)
    plt.tight_layout()
    return save_fig(fig, "mean_pred_by_gt.png")


# ─── Report ──────────────────────────────────────────────────────────────────

def build_report(df: pd.DataFrame, metrics: dict, csv_path: Path, ts: str) -> str:
    m, pc, labels = metrics, metrics["per_class"], metrics["labels"]

    print("  confusion matrix...")
    p_cm     = plot_confusion_matrix(metrics)
    print("  normalised confusion matrix...")
    p_norm   = plot_confusion_normalised(metrics)
    print("  distribution...")
    p_dist   = plot_distribution(metrics)
    print("  mean pred by GT...")
    p_bias   = plot_mean_pred_by_gt(df, metrics)
    print("  accuracy by region...")
    p_region = plot_accuracy_by_group(df, "region", "Accuracy by Region")
    print("  accuracy by gender...")
    p_gender = plot_accuracy_by_group(df, "gender", "Accuracy by Gender")
    print("  accuracy by age...")
    p_age    = plot_accuracy_by_age(df)

    cm_header = "| | " + " | ".join(f"**Pred {LABEL_SHORT[l]}**" for l in labels) + " |"
    cm_sep    = "|---|" + "|".join("---" for _ in labels) + "|"
    cm_rows   = "\n".join(
        "| **GT " + LABEL_SHORT[g] + "** | " +
        " | ".join(str(m["cm"].get((g, p), 0)) for p in labels) + " |"
        for g in labels
    )

    n       = m["valid"]
    macro_p = sum(pc[l]["precision"] for l in labels) / len(labels)
    macro_r = sum(pc[l]["recall"]    for l in labels) / len(labels)
    w_p     = sum(pc[l]["precision"] * pc[l]["support"] for l in labels) / n
    w_r     = sum(pc[l]["recall"]    * pc[l]["support"] for l in labels) / n

    pc_rows = "\n".join(
        f"| {LABEL_MAP[l]} | {pc[l]['support']} | "
        f"{pc[l]['precision']:.4f} | {pc[l]['recall']:.4f} | {pc[l]['f1']:.4f} |"
        for l in labels
    )

    dist_rows = "\n".join(
        f"| {LABEL_MAP[l]} | {m['gt_dist'].get(l,0)} "
        f"({m['gt_dist'].get(l,0)/n*100:.1f}%) | "
        f"{m['pred_dist'].get(l,0)} ({m['pred_dist'].get(l,0)/n*100:.1f}%) |"
        for l in labels
    )

    region_rows = "\n".join(
        f"| {r['region']} | {int(r['count'])} | {r['mean']:.4f} |"
        for _, r in (df.dropna(subset=["match", "region"])
                     .groupby("region")["match"]
                     .agg(["mean", "count"]).reset_index()
                     .sort_values("mean", ascending=False)).iterrows()
    )
    gender_rows = "\n".join(
        f"| {r['gender']} | {int(r['count'])} | {r['mean']:.4f} |"
        for _, r in (df.dropna(subset=["match", "gender"])
                     .groupby("gender")["match"]
                     .agg(["mean", "count"]).reset_index()
                     .sort_values("mean", ascending=False)).iterrows()
    )

    return f"""# womenwork Prediction Report

**Model:** gpt-5.4-mini | **Temperature:** 0.8 | **Date:** {ts}
**Source:** `{csv_path.name}`
**Question:** *"A man's job is to earn money; a woman's job is to look after the home and family."*
(1 = Strongly disagree → 5 = Strongly agree)
**Prompt cleaning:** sentences revealing gender-role attitudes (`geleneksel cinsiyet rolleri`, `evin reisinin erkek`) removed before inference.

---

## 1. Overall Performance

| Metric | Value |
|---|---|
| Total personas | {m['total']} |
| Valid predictions | {m['valid']} |
| Parse failures | {m['parse_fail']} |
| **Accuracy** | **{m['accuracy']:.4f}** |
| Macro F1 | {m['macro_f1']:.4f} |
| Weighted F1 | {m['weighted_f1']:.4f} |

> Note: 5-class prediction — random baseline accuracy ≈ 0.20.

---

## 2. Ground Truth vs Prediction Distribution

{img_tag(p_dist, "Distribution")}

| Class | Ground Truth | Prediction |
|---|---|---|
{dist_rows}

---

## 3. Confusion Matrix

{img_tag(p_cm, "Confusion Matrix")}

{cm_header}
{cm_sep}
{cm_rows}

---

## 4. Normalised Confusion Matrix

{img_tag(p_norm, "Normalised Confusion Matrix")}

> Row-normalised: shows what the model predicts *given* the true class.

---

## 5. Prediction Bias by GT Class

{img_tag(p_bias, "Mean Predicted Class per GT Class")}

> Bars above the dashed line = model over-predicts (biased toward agreement);
> bars below = model under-predicts (biased toward disagreement).

---

## 6. Per-class Metrics

| Class | Support | Precision | Recall | F1 |
|---|---|---|---|---|
{pc_rows}
| **Macro avg** | {n} | {macro_p:.4f} | {macro_r:.4f} | {m['macro_f1']:.4f} |
| **Weighted avg** | {n} | {w_p:.4f} | {w_r:.4f} | {m['weighted_f1']:.4f} |

---

## 7. Accuracy by Region

{img_tag(p_region, "Accuracy by Region")}

| Region | N | Accuracy |
|---|---|---|
{region_rows}

---

## 8. Accuracy by Gender

{img_tag(p_gender, "Accuracy by Gender")}

| Gender | N | Accuracy |
|---|---|---|
{gender_rows}

---

## 9. Accuracy by Age Group

{img_tag(p_age, "Accuracy by Age Group")}

---

## 10. Notes

- **5-class** prediction with ordinal scale — adjacent-class errors are less costly than cross-scale errors.
- Parse failures: **{m['parse_fail']}** personas (`{m['parse_fail']/m['total']*100:.1f}%`).
- Ground truth distribution: majority class = {max(m['gt_dist'], key=m['gt_dist'].get)} ({m['gt_dist'].get(max(m['gt_dist'], key=m['gt_dist'].get),0)} personas, {m['gt_dist'].get(max(m['gt_dist'], key=m['gt_dist'].get),0)/n*100:.1f}%).
"""


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else load_latest_csv()
    print(f"Source: {csv_path}")

    df      = load_data(csv_path)
    metrics = compute_metrics(df)
    ts      = datetime.now().strftime("%Y-%m-%d %H:%M")
    ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")

    global PLOTS_DIR
    REPORTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR = FIGURES_DIR / f"womenwork_plots_{ts_file}"
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating plots...")
    report = build_report(df, metrics, csv_path, ts)

    report_path = REPORTS_DIR / f"womenwork_report_{ts_file}.md"
    report_path.write_text(report, encoding="utf-8")

    print(f"\nReport → {report_path}")
    print(f"Plots  → {PLOTS_DIR}/")
    print(f"Accuracy: {metrics['accuracy']:.4f} | Macro F1: {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
