"""
Generates a GitHub-friendly markdown report from pacdemons prediction results.

Usage:
    python3 report_pacdemons.py [csv_file]

If no file is given, the latest pacdemons_predictions_*.csv in exports/ is used.

Output:
    - Markdown report: exports/pacdemons_report_<timestamp>.md
    - PNG plots:       exports/report_plots/<report_stem>/*.png
"""

import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

BASE_DIR     = Path(__file__).parent.parent.resolve()
EXPORTS_DIR  = BASE_DIR / "exports"
REPORTS_DIR  = BASE_DIR / "reports"
FIGURES_DIR  = BASE_DIR / "figures"

LABEL_MAP = {1: "Yes (participated)", 2: "No (did not participate)"}
LABEL_SHORT = {1: "Yes (1)", 2: "No (2)"}
COLORS = {
    "gt": "#4C72B0",
    "pred": "#DD8452",
    "match": "#55A868",
    "no_match": "#C44E52",
    "mid": "#F0A500",
}


# ─── Data ────────────────────────────────────────────────────────────────────

def load_latest_csv() -> Path:
    candidates = sorted(EXPORTS_DIR.glob("pacdemons_predictions_*.csv"))
    if not candidates:
        raise FileNotFoundError("No pacdemons_predictions_*.csv found in exports/")
    return candidates[-1]


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_cols = ["gt_pacdemons", "pred_pacdemons", "match"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in CSV: {missing}")

    df["gt_pacdemons"] = pd.to_numeric(df["gt_pacdemons"], errors="coerce")
    df["pred_pacdemons"] = pd.to_numeric(df["pred_pacdemons"], errors="coerce")
    df["match"] = pd.to_numeric(df["match"], errors="coerce")

    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")

    return df


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(df: pd.DataFrame) -> dict:
    valid = df.dropna(subset=["gt_pacdemons", "pred_pacdemons", "match"]).copy()
    labels = [1, 2]

    gt = valid["gt_pacdemons"].astype(int).tolist()
    pred = valid["pred_pacdemons"].astype(int).tolist()

    cm = {
        (g, p): sum(1 for a, b in zip(gt, pred) if a == g and b == p)
        for g in labels for p in labels
    }

    per_class = {}
    for lbl in labels:
        tp = cm.get((lbl, lbl), 0)
        fp = sum(cm.get((g, lbl), 0) for g in labels if g != lbl)
        fn = sum(cm.get((lbl, p), 0) for p in labels if p != lbl)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        support = sum(1 for g in gt if g == lbl)

        per_class[lbl] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    n = len(valid)
    acc = valid["match"].sum() / n if n else 0.0
    macro_f1 = sum(per_class[l]["f1"] for l in labels) / len(labels)
    weighted_f1 = (
        sum(per_class[l]["f1"] * per_class[l]["support"] for l in labels) / n if n else 0.0
    )

    return {
        "total": len(df),
        "valid": n,
        "parse_fail": int(df["pred_pacdemons"].isna().sum()),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "cm": cm,
        "per_class": per_class,
        "labels": labels,
        "gt_dist": dict(Counter(gt)),
        "pred_dist": dict(Counter(pred)),
    }


# ─── Plot helpers ────────────────────────────────────────────────────────────

def save_fig(fig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def md_img(relative_path: str, alt: str = "") -> str:
    return f"![{alt}]({relative_path})"


def sort_group_values(series: pd.Series) -> pd.Series:
    # Keeps strings stable and avoids weird NaN ordering issues
    return series.fillna("Unknown").astype(str)


# ─── Plots ───────────────────────────────────────────────────────────────────

def plot_confusion_matrix(metrics: dict, output_path: Path) -> None:
    labels = metrics["labels"]
    cm = metrics["cm"]
    matrix = np.array([[cm.get((g, p), 0) for p in labels] for g in labels])
    total = matrix.sum()

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap="Blues")
    plt.colorbar(im, ax=ax)

    tick_labels = [LABEL_SHORT[l] for l in labels]
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(tick_labels, fontsize=10)
    ax.set_yticklabels(tick_labels, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Ground Truth", fontsize=11)
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")

    max_val = matrix.max() if matrix.size else 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = matrix[i, j]
            pct = (val / total * 100) if total else 0
            color = "white" if max_val and val > max_val * 0.6 else "black"
            ax.text(
                j, i, f"{val}\n({pct:.1f}%)",
                ha="center", va="center",
                fontsize=11, color=color, fontweight="bold"
            )

    plt.tight_layout()
    save_fig(fig, output_path)


def plot_distribution(metrics: dict, output_path: Path) -> None:
    labels = metrics["labels"]
    gt_dist = metrics["gt_dist"]
    pred_dist = metrics["pred_dist"]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    bars_gt = ax.bar(
        x - width / 2,
        [gt_dist.get(l, 0) for l in labels],
        width,
        label="Ground Truth",
        color=COLORS["gt"],
        alpha=0.85,
    )
    bars_pred = ax.bar(
        x + width / 2,
        [pred_dist.get(l, 0) for l in labels],
        width,
        label="Prediction",
        color=COLORS["pred"],
        alpha=0.85,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([LABEL_SHORT[l] for l in labels], fontsize=10)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Ground Truth vs Prediction Distribution", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    for bar in list(bars_gt) + list(bars_pred):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            str(int(bar.get_height())),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    save_fig(fig, output_path)


def plot_accuracy_by_group(
    df: pd.DataFrame,
    col: str,
    title: str,
    output_path: Path,
) -> bool:
    if col not in df.columns:
        return False

    valid = df.dropna(subset=["match", col]).copy()
    if valid.empty:
        return False

    valid[col] = sort_group_values(valid[col])

    groups = (
        valid.groupby(col)["match"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "accuracy", "count": "n"})
        .sort_values("accuracy", ascending=True)
    )

    if groups.empty:
        return False

    fig, ax = plt.subplots(figsize=(7, max(3, len(groups) * 0.45)))
    colors = [
        COLORS["match"] if v >= 0.9 else COLORS["no_match"] if v < 0.8 else COLORS["mid"]
        for v in groups["accuracy"]
    ]
    bars = ax.barh(groups[col].astype(str), groups["accuracy"], color=colors, alpha=0.85)

    for bar, (_, row) in zip(bars, groups.iterrows()):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{row['accuracy']:.3f}  (n={int(row['n'])})",
            va="center",
            fontsize=9,
        )

    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Accuracy", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axvline(groups["accuracy"].mean(), color="gray", linestyle="--", linewidth=1, label="Mean")
    ax.legend(fontsize=9)

    plt.tight_layout()
    save_fig(fig, output_path)
    return True


def plot_accuracy_by_age(df: pd.DataFrame, output_path: Path) -> bool:
    if "age" not in df.columns:
        return False

    valid = df.dropna(subset=["match", "age"]).copy()
    if valid.empty:
        return False

    bins = [18, 25, 35, 45, 55, 65, 100]
    age_labels = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    valid["age_group"] = pd.cut(valid["age"], bins=bins, labels=age_labels, right=False)

    groups = (
        valid.dropna(subset=["age_group"])
        .groupby("age_group", observed=True)["match"]
        .agg(["mean", "count"])
        .reset_index()
    )

    if groups.empty:
        return False

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [
        COLORS["match"] if v >= 0.9 else COLORS["no_match"] if v < 0.8 else COLORS["mid"]
        for v in groups["mean"]
    ]
    bars = ax.bar(groups["age_group"].astype(str), groups["mean"], color=colors, alpha=0.85)

    for bar, (_, row) in zip(bars, groups.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{row['mean']:.3f}\n(n={int(row['count'])})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylim(0, 1.15)
    ax.set_xlabel("Age Group", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("Accuracy by Age Group", fontsize=13, fontweight="bold")
    ax.axhline(valid["match"].mean(), color="gray", linestyle="--", linewidth=1, label="Overall mean")
    ax.legend(fontsize=9)

    plt.tight_layout()
    save_fig(fig, output_path)
    return True


# ─── Report ──────────────────────────────────────────────────────────────────

def build_report(
    df: pd.DataFrame,
    metrics: dict,
    csv_path: Path,
    timestamp_human: str,
    report_path: Path,
    plot_dir: Path,
) -> str:
    m = metrics
    pc = metrics["per_class"]
    labels = metrics["labels"]

    plot_dir.mkdir(parents=True, exist_ok=True)

    cm_file = plot_dir / "confusion_matrix.png"
    dist_file = plot_dir / "gt_vs_pred_distribution.png"
    region_file = plot_dir / "accuracy_by_region.png"
    gender_file = plot_dir / "accuracy_by_gender.png"
    age_file = plot_dir / "accuracy_by_age_group.png"

    print("Generating plots...")
    plot_confusion_matrix(metrics, cm_file)
    plot_distribution(metrics, dist_file)
    has_region = plot_accuracy_by_group(df, "region", "Accuracy by Region", region_file)
    has_gender = plot_accuracy_by_group(df, "gender", "Accuracy by Gender", gender_file)
    has_age = plot_accuracy_by_age(df, age_file)

    # Relative paths from reports/ to figures/<plot_dir>/
    def _rel(f: Path) -> str:
        return f"../figures/{plot_dir.name}/{f.name}"

    rel_cm     = _rel(cm_file)
    rel_dist   = _rel(dist_file)
    rel_region = _rel(region_file)
    rel_gender = _rel(gender_file)
    rel_age    = _rel(age_file)

    cm_header = "| | " + " | ".join(f"**Pred {LABEL_SHORT[l]}**" for l in labels) + " |"
    cm_sep = "|---|" + "|".join("---" for _ in labels) + "|"
    cm_rows = "\n".join(
        "| **GT " + LABEL_SHORT[g] + "** | " +
        " | ".join(str(m["cm"].get((g, p), 0)) for p in labels) + " |"
        for g in labels
    )

    pc_rows = "\n".join(
        f"| {LABEL_SHORT[l]} | {pc[l]['support']} | "
        f"{pc[l]['precision']:.4f} | {pc[l]['recall']:.4f} | {pc[l]['f1']:.4f} |"
        for l in labels
    )

    yes_gt_pct = m["gt_dist"].get(1, 0) / m["valid"] * 100 if m["valid"] else 0
    yes_pred_pct = m["pred_dist"].get(1, 0) / m["valid"] * 100 if m["valid"] else 0
    macro_prec = sum(pc[l]["precision"] for l in labels) / len(labels)
    macro_rec = sum(pc[l]["recall"] for l in labels) / len(labels)
    w_prec = sum(pc[l]["precision"] * pc[l]["support"] for l in labels) / m["valid"] if m["valid"] else 0
    w_rec = sum(pc[l]["recall"] * pc[l]["support"] for l in labels) / m["valid"] if m["valid"] else 0

    region_section = ""
    if has_region:
        region_acc = (
            df.dropna(subset=["match", "region"])
            .copy()
            .assign(region=lambda x: x["region"].fillna("Unknown").astype(str))
            .groupby("region")["match"]
            .agg(["mean", "count"])
            .reset_index()
            .sort_values("mean", ascending=False)
        )
        region_rows = "\n".join(
            f"| {row['region']} | {int(row['count'])} | {row['mean']:.4f} |"
            for _, row in region_acc.iterrows()
        )
        region_section = f"""
## 5. Accuracy by Region

{md_img(rel_region, "Accuracy by Region")}

| Region | N | Accuracy |
|---|---|---|
{region_rows}

---
"""
    else:
        region_section = """
## 5. Accuracy by Region

Region column not available.

---
"""

    gender_section = ""
    if has_gender:
        gender_acc = (
            df.dropna(subset=["match", "gender"])
            .copy()
            .assign(gender=lambda x: x["gender"].fillna("Unknown").astype(str))
            .groupby("gender")["match"]
            .agg(["mean", "count"])
            .reset_index()
            .sort_values("mean", ascending=False)
        )
        gender_rows = "\n".join(
            f"| {row['gender']} | {int(row['count'])} | {row['mean']:.4f} |"
            for _, row in gender_acc.iterrows()
        )
        gender_section = f"""
## 6. Accuracy by Gender

{md_img(rel_gender, "Accuracy by Gender")}

| Gender | N | Accuracy |
|---|---|---|
{gender_rows}

---
"""
    else:
        gender_section = """
## 6. Accuracy by Gender

Gender column not available.

---
"""

    age_section = ""
    if has_age:
        age_section = f"""
## 7. Accuracy by Age Group

{md_img(rel_age, "Accuracy by Age Group")}

---
"""
    else:
        age_section = """
## 7. Accuracy by Age Group

Age column not available.

---
"""

    return f"""# pacdemons Prediction Report

**Model:** gpt-5.4-mini  
**Temperature:** 0.8  
**Date:** {timestamp_human}  
**Source:** `{csv_path.name}`

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

---

## 2. Ground Truth vs Prediction Distribution

{md_img(rel_dist, "Ground Truth vs Prediction Distribution")}

| | Ground Truth | Prediction |
|---|---|---|
| Yes (1) | {m['gt_dist'].get(1, 0)} | {m['pred_dist'].get(1, 0)} |
| No (2) | {m['gt_dist'].get(2, 0)} | {m['pred_dist'].get(2, 0)} |

> Ground truth "Yes" rate: **{yes_gt_pct:.1f}%**  
> Model "Yes" rate: **{yes_pred_pct:.1f}%**

---

## 3. Confusion Matrix

{md_img(rel_cm, "Confusion Matrix")}

{cm_header}
{cm_sep}
{cm_rows}

---

## 4. Per-class Metrics

| Class | Support | Precision | Recall | F1 |
|---|---|---|---|---|
{pc_rows}
| **Macro avg** | {m['valid']} | {macro_prec:.4f} | {macro_rec:.4f} | {m['macro_f1']:.4f} |
| **Weighted avg** | {m['valid']} | {w_prec:.4f} | {w_rec:.4f} | {m['weighted_f1']:.4f} |

---

{region_section}
{gender_section}
{age_section}
## 8. Notes

- The model correctly reflects class imbalance: the ground truth "Yes" rate is only **{yes_gt_pct:.1f}%**, and the model predominantly predicts **No**.
- High overall accuracy is largely driven by the majority class.
- Recall for the "Yes" class is only **{pc[1]['recall']:.4f}**.
- Parse failures: **{m['parse_fail']}** personas (**{(m['parse_fail'] / m['total'] * 100) if m['total'] else 0:.1f}%**).
"""


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    csv_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else load_latest_csv().resolve()
    print(f"Source CSV: {csv_path}")

    df = load_data(csv_path)
    metrics = compute_metrics(df)

    timestamp_human = datetime.now().strftime("%Y-%m-%d %H:%M")
    timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")

    REPORTS_DIR.mkdir(exist_ok=True)
    report_path = REPORTS_DIR / f"pacdemons_report_{timestamp_file}.md"
    plot_dir = FIGURES_DIR / f"pacdemons_plots_{timestamp_file}"
    plot_dir.mkdir(parents=True, exist_ok=True)

    report = build_report(
        df=df,
        metrics=metrics,
        csv_path=csv_path,
        timestamp_human=timestamp_human,
        report_path=report_path,
        plot_dir=plot_dir,
    )

    report_path.write_text(report, encoding="utf-8")

    print(f"\nReport written to: {report_path}")
    print(f"Plots written to:  {plot_dir}")
    print(f"Accuracy: {metrics['accuracy']:.4f} | Macro F1: {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()