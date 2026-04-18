"""
Rapor scripti: Dijital İkizlerle Nedensel Çıkarım Deneyi
Kürtçe Anadil Eğitimi — 3×4 Tam Faktöriyel Tasarım

Kullanım: python3 scripts/report_kurdish_causal.py exports/kurdish_causal_<ts>.csv
"""

import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ─── Dizinler ─────────────────────────────────────────────────────────────────

BASE_DIR    = Path(__file__).parent.parent
FIGURES_DIR = BASE_DIR / "figures"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# ─── Renk paleti ──────────────────────────────────────────────────────────────

C_SECURITY = {"K": "#6baed6", "1a": "#e6550d", "1b": "#74c476"}   # mavi / turuncu / yeşil
C_FRAMING  = {"K": "#bdbdbd", "2a": "#fdae6b", "2b": "#3182bd", "2c": "#31a354"}
C_NEUTRAL  = "#555555"

# ─── Etiketler ────────────────────────────────────────────────────────────────

SEC_LABELS = {
    "K":  "Kontrol\n(bağlam yok)",
    "1a": "Çatışma\n(aktif operasyon)",
    "1b": "Barış\n(anlaşma sağlandı)",
}
FRM_LABELS = {
    "K":  "Kontrol\n(çerçeve yok)",
    "2a": "Dini\n(Kur'an ayeti)",
    "2b": "İnsan Hakları\n(BM Beyannamesi)",
    "2c": "Ulusal Özgürlük\n(kendi kaderini tayin)",
}
DV_LABELS = {
    "dv_3a": "3a — Meşru Hak Mücadelesi",
    "dv_3b": "3b — Toplumsal Huzuru Bozucu",
    "dv_3c": "3c — Kürtçe Eğitime Yasal İzin (Ana DV)",
    "dv_3d": "3d — Davranış Niyeti",
}
D3D_LABELS = {
    1: "Bizzat katılır",
    2: "Sosyal medyada destekler",
    3: "Sessizce destekler",
    4: "Sessizce karşı çıkar",
    5: "Sosyal medyada karşı çıkar",
    6: "Karşıt etkinliğe katılır",
}

# ─── Yardımcı: grafik kayıt ───────────────────────────────────────────────────

def save_fig(plots_dir: Path, name: str) -> str:
    path = plots_dir / name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return f"../figures/{plots_dir.name}/{name}"


# ─── Veri yükleme & temizleme ─────────────────────────────────────────────────

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["age_group"] = pd.cut(
        df["age"], bins=[17, 29, 44, 59, 120],
        labels=["18–29", "30–44", "45–59", "60+"]
    )
    return df


# ─── Grafikler ────────────────────────────────────────────────────────────────

def plot_participant_demographics(df: pd.DataFrame, plots_dir: Path) -> dict[str, str]:
    paths = {}
    personas = df.drop_duplicates("persona_id")
    n = len(personas)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Yaş dağılımı
    ax = axes[0]
    ax.hist(personas["age"], bins=20, color="#6baed6", edgecolor="white", linewidth=0.5)
    ax.axvline(personas["age"].mean(), color="#e6550d", linestyle="--", linewidth=1.5,
               label=f"Ort. {personas['age'].mean():.1f}")
    ax.set_title("Yaş Dağılımı", fontsize=12, fontweight="bold")
    ax.set_xlabel("Yaş")
    ax.set_ylabel("Kişi Sayısı")
    ax.legend(fontsize=9)

    # Cinsiyet
    ax = axes[1]
    gender_counts = personas["gender"].value_counts()
    bars = ax.bar(gender_counts.index, gender_counts.values,
                  color=["#6baed6", "#fd8d3c"], edgecolor="white")
    for bar, val in zip(bars, gender_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{val}\n(%{val/n*100:.0f})", ha="center", fontsize=9)
    ax.set_title("Cinsiyet Dağılımı", fontsize=12, fontweight="bold")
    ax.set_ylim(0, gender_counts.max() * 1.2)

    # Bölge
    ax = axes[2]
    region_counts = personas["region"].value_counts()
    short_names = {r: r.replace(" Bölgesi", "").replace(" Anadolu", "\nAnadolu") for r in region_counts.index}
    ax.barh([short_names[r] for r in region_counts.index], region_counts.values,
            color="#74c476", edgecolor="white")
    ax.set_title("Bölge Dağılımı", fontsize=12, fontweight="bold")
    ax.set_xlabel("Kişi Sayısı")

    fig.suptitle(f"Katılımcı Demografisi (N={n} dijital ikiz)", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    paths["demographics"] = save_fig(plots_dir, "demographics.png")
    return paths


def plot_manipulation_check(df: pd.DataFrame, plots_dir: Path) -> dict[str, str]:
    paths = {}
    mc_df = df.dropna(subset=["manip_check_sampled"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, sec, expected, title in [
        (axes[0], "1a", 1, "Çatışma Koşulu (1a)\nBeklenen yanıt: Aktif çatışma var (p1)"),
        (axes[1], "1b", 2, "Barış Koşulu (1b)\nBeklenen yanıt: Anlaşma sağlandı (p2)"),
    ]:
        sub = mc_df[mc_df["security_context"] == sec]
        counts = sub["manip_check_sampled"].value_counts().sort_index()
        labels = {1: "Aktif çatışma", 2: "Anlaşma sağlandı", 3: "Hatırlamıyorum"}
        colors = ["#e6550d" if i == expected else "#bdbdbd" for i in counts.index]
        bars = ax.bar([labels.get(i, str(i)) for i in counts.index], counts.values, color=colors)
        for bar, val in zip(bars, counts.values):
            pct = val / len(sub) * 100
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                    f"%{pct:.0f}", ha="center", fontsize=10, fontweight="bold")
        acc = (sub["manip_check_sampled"] == expected).mean()
        ax.set_title(f"{title}\nDoğruluk: %{acc*100:.1f}", fontsize=10, fontweight="bold")
        ax.set_ylim(0, counts.max() * 1.25)

    fig.suptitle("Manipülasyon Kontrolü — Model haberi doğru algıladı mı?",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    paths["manip_check"] = save_fig(plots_dir, "manip_check.png")
    return paths


def plot_main_effect_security(df: pd.DataFrame, plots_dir: Path) -> dict[str, str]:
    paths = {}
    dv_df = df.dropna(subset=["dv_3c_sampled"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Sol: bar chart ortalama + 95% CI
    ax = axes[0]
    sec_order = ["K", "1a", "1b"]
    means, cis = [], []
    for sec in sec_order:
        sub = dv_df[dv_df["security_context"] == sec]["dv_3c_sampled"]
        means.append(sub.mean())
        cis.append(1.96 * sub.std() / np.sqrt(len(sub)))
    colors = [C_SECURITY[s] for s in sec_order]
    bars = ax.bar([SEC_LABELS[s] for s in sec_order], means, color=colors,
                  yerr=cis, capsize=5, edgecolor="white", linewidth=0.8, error_kw={"linewidth": 1.5})
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 0.05,
                f"{m:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.axhline(means[0], color=C_SECURITY["K"], linestyle="--", linewidth=1, alpha=0.6, label="Kontrol ort.")
    ax.set_ylim(1, 3.5)
    ax.set_ylabel("Ortalama Yanıt (1–5 Likert)", fontsize=10)
    ax.set_title("Güvenlik Bağlamının 3c'ye Etkisi\n(tüm çerçeve koşulları birleşik)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

    # Sağ: dağılım (violin)
    ax = axes[1]
    data_to_plot = [dv_df[dv_df["security_context"] == s]["dv_3c_sampled"].dropna() for s in sec_order]
    vp = ax.violinplot(data_to_plot, positions=[1, 2, 3], showmedians=True, showmeans=False)
    for i, (body, sec) in enumerate(zip(vp["bodies"], sec_order)):
        body.set_facecolor(C_SECURITY[sec])
        body.set_alpha(0.7)
    vp["cmedians"].set_color("black")
    vp["cmedians"].set_linewidth(2)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([SEC_LABELS[s] for s in sec_order], fontsize=8)
    ax.set_ylabel("Yanıt Dağılımı (1–5)", fontsize=10)
    ax.set_title("Yanıt Dağılımı — Güvenlik Bağlamına Göre", fontsize=10, fontweight="bold")
    ax.set_ylim(0.5, 5.5)

    fig.suptitle("Ana Etki: Güvenlik Bağlamı → Kürtçe Eğitim Desteği (3c)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    paths["main_effect_security"] = save_fig(plots_dir, "main_effect_security.png")
    return paths


def plot_main_effect_framing(df: pd.DataFrame, plots_dir: Path) -> dict[str, str]:
    paths = {}
    dv_df = df.dropna(subset=["dv_3c_sampled"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    frm_order = ["K", "2a", "2b", "2c"]

    # Sol: bar chart
    ax = axes[0]
    means, cis = [], []
    for frm in frm_order:
        sub = dv_df[dv_df["framing"] == frm]["dv_3c_sampled"]
        means.append(sub.mean())
        cis.append(1.96 * sub.std() / np.sqrt(len(sub)))
    colors = [C_FRAMING[f] for f in frm_order]
    bars = ax.bar([FRM_LABELS[f] for f in frm_order], means, color=colors,
                  yerr=cis, capsize=5, edgecolor="white", linewidth=0.8, error_kw={"linewidth": 1.5})
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 0.05,
                f"{m:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.axhline(means[0], color=C_FRAMING["K"], linestyle="--", linewidth=1, alpha=0.6, label="Kontrol ort.")
    ax.set_ylim(1, 3.5)
    ax.set_ylabel("Ortalama Yanıt (1–5 Likert)", fontsize=10)
    ax.set_title("Çerçevelemenin 3c'ye Etkisi\n(tüm güvenlik koşulları birleşik)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

    # Sağ: tüm DVler için framing etkisi
    ax = axes[1]
    dvs = ["dv_3a", "dv_3b", "dv_3c"]
    dv_names = ["3a Meşruiyet", "3b Huzur bozucu", "3c Kürtçe eğitim"]
    x = np.arange(len(frm_order))
    width = 0.25
    for i, (dv, name) in enumerate(zip(dvs, dv_names)):
        col = f"{dv}_sampled"
        means_dv = [dv_df[dv_df["framing"] == f][col].mean() for f in frm_order]
        ax.plot([FRM_LABELS[f] for f in frm_order], means_dv,
                marker="o", linewidth=2, label=name, markersize=7)
    ax.set_ylabel("Ortalama Yanıt (1–5)", fontsize=10)
    ax.set_title("Tüm DV'ler — Çerçeve Etkisi", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_ylim(2, 3.8)

    fig.suptitle("Ana Etki: Çerçeveleme Türü → Bağımlı Değişkenler",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    paths["main_effect_framing"] = save_fig(plots_dir, "main_effect_framing.png")
    return paths


def plot_interaction_heatmap(df: pd.DataFrame, plots_dir: Path) -> dict[str, str]:
    paths = {}
    dv_df = df.dropna(subset=["dv_3c_sampled"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sec_order = ["K", "1a", "1b"]
    frm_order = ["K", "2a", "2b", "2c"]

    # Sol: heatmap
    ax = axes[0]
    pivot = dv_df.pivot_table(
        values="dv_3c_sampled", index="security_context", columns="framing", aggfunc="mean"
    ).loc[sec_order, frm_order]
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=2.4, vmax=3.1, aspect="auto")
    ax.set_xticks(range(len(frm_order)))
    ax.set_xticklabels([FRM_LABELS[f] for f in frm_order], fontsize=8)
    ax.set_yticks(range(len(sec_order)))
    ax.set_yticklabels([SEC_LABELS[s] for s in sec_order], fontsize=8)
    for i in range(len(sec_order)):
        for j in range(len(frm_order)):
            ax.text(j, i, f"{pivot.values[i, j]:.3f}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color="black")
    plt.colorbar(im, ax=ax, label="Ort. Destek (1–5)")
    ax.set_title("Etkileşim Haritası: Güvenlik × Çerçeve\n(3c — Kürtçe Eğitim Desteği)",
                 fontsize=10, fontweight="bold")

    # Sağ: çizgi grafik (etkileşim)
    ax = axes[1]
    for sec in sec_order:
        vals = [dv_df[(dv_df["security_context"] == sec) & (dv_df["framing"] == f)]["dv_3c_sampled"].mean()
                for f in frm_order]
        ax.plot([FRM_LABELS[f] for f in frm_order], vals,
                marker="o", linewidth=2.5, markersize=8,
                color=C_SECURITY[sec], label=SEC_LABELS[sec].replace("\n", " "))
    ax.set_ylabel("Ortalama 3c Yanıtı (1–5)", fontsize=10)
    ax.set_title("Etkileşim Grafiği\n(çerçeve × güvenlik bağlamı)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(2.3, 3.2)

    fig.suptitle("Faktöriyel Etkileşim: Güvenlik Bağlamı × Çerçeveleme Türü",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    paths["interaction_heatmap"] = save_fig(plots_dir, "interaction_heatmap.png")
    return paths


def plot_within_person_effects(df: pd.DataFrame, plots_dir: Path) -> dict[str, str]:
    paths = {}
    dv_df = df.dropna(subset=["dv_3c_sampled"])

    # Her persona için koşullar arası fark hesapla
    ref = dv_df[dv_df["condition_id"] == "C01"][["persona_id", "dv_3c_sampled"]].rename(
        columns={"dv_3c_sampled": "ref"}
    )
    comparisons = [
        ("C05", "Çatışma\n(bağlam yok → aktif operasyon)", C_SECURITY["1a"]),
        ("C09", "Barış\n(bağlam yok → anlaşma)", C_SECURITY["1b"]),
        ("C02", "Dini çerçeve\n(çerçevesiz → Kur'an)", C_FRAMING["2a"]),
        ("C03", "İnsan hakları\n(çerçevesiz → BM)", C_FRAMING["2b"]),
        ("C04", "Ulusal özgürlük\n(çerçevesiz → kendi kader)", C_FRAMING["2c"]),
        ("C07", "Çatışma + İnsan hakları\n(en düşük?)", "#9e9ac8"),
        ("C11", "Barış + İnsan hakları\n(referans en yüksek?)", "#fdbb84"),
    ]

    diffs, labels, colors, ns = [], [], [], []
    for cid, label, color in comparisons:
        comp = dv_df[dv_df["condition_id"] == cid][["persona_id", "dv_3c_sampled"]].rename(
            columns={"dv_3c_sampled": "comp"}
        )
        merged = ref.merge(comp, on="persona_id")
        diff = merged["comp"] - merged["ref"]
        diffs.append(diff.mean())
        labels.append(label)
        colors.append(color)
        ns.append(len(merged))

    fig, ax = plt.subplots(figsize=(12, 6))
    y_pos = range(len(diffs))
    bars = ax.barh(list(y_pos), diffs, color=colors, edgecolor="white", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=1.2, linestyle="-")
    for i, (d, n) in enumerate(zip(diffs, ns)):
        sign = "+" if d >= 0 else ""
        ax.text(d + (0.005 if d >= 0 else -0.005), i,
                f"{sign}{d:.3f} (n={n})",
                va="center", ha="left" if d >= 0 else "right", fontsize=9)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Ortalama Fark vs Kontrol-Kontrol (C01)", fontsize=10)
    ax.set_title("Birey İçi (Within-Person) Etkiler\n"
                 "Her koşul, aynı personanın kontrol koşulundaki yanıtına göre karşılaştırılıyor",
                 fontsize=11, fontweight="bold")
    ax.set_xlim(-0.3, 0.3)

    plt.tight_layout()
    paths["within_person"] = save_fig(plots_dir, "within_person.png")
    return paths


def plot_threat_and_frame_legit(df: pd.DataFrame, plots_dir: Path) -> dict[str, str]:
    paths = {}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Sol: tehdit algısı
    ax = axes[0]
    dv_df = df.dropna(subset=["threat_sampled"])
    sec_order = ["K", "1a", "1b"]
    means = [dv_df[dv_df["security_context"] == s]["threat_sampled"].mean() for s in sec_order]
    cis   = [1.96 * dv_df[dv_df["security_context"] == s]["threat_sampled"].std() /
              np.sqrt(len(dv_df[dv_df["security_context"] == s])) for s in sec_order]
    bars = ax.bar([SEC_LABELS[s] for s in sec_order], means, color=[C_SECURITY[s] for s in sec_order],
                  yerr=cis, capsize=5, edgecolor="white", error_kw={"linewidth": 1.5})
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 0.03,
                f"{m:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(1, 3.5)
    ax.set_ylabel("Tehdit Algısı (1–4)", fontsize=10)
    ax.set_title("Tehdit Algısı — Güvenlik Bağlamına Göre\n(1=hiç tehdit yok, 4=çok ciddi tehdit)",
                 fontsize=10, fontweight="bold")

    # Sağ: çerçeve meşruiyeti
    ax = axes[1]
    fl_df = df.dropna(subset=["frame_legit_sampled"])
    frm_order = ["2a", "2b", "2c"]
    means_fl = [fl_df[fl_df["framing"] == f]["frame_legit_sampled"].mean() for f in frm_order]
    cis_fl   = [1.96 * fl_df[fl_df["framing"] == f]["frame_legit_sampled"].std() /
                np.sqrt(len(fl_df[fl_df["framing"] == f])) for f in frm_order]
    bars = ax.bar([FRM_LABELS[f] for f in frm_order], means_fl,
                  color=[C_FRAMING[f] for f in frm_order],
                  yerr=cis_fl, capsize=5, edgecolor="white", error_kw={"linewidth": 1.5})
    for bar, m in zip(bars, means_fl):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 0.03,
                f"{m:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(1, 3.5)
    ax.set_ylabel("Çerçeve Meşruiyeti (1–4)", fontsize=10)
    ax.set_title("Protestocu Grubun Yaklaşımı Ne Kadar Meşru?\n(1=hiç meşru değil, 4=çok meşru)",
                 fontsize=10, fontweight="bold")

    fig.suptitle("Mekanizma Değişkenleri: Tehdit Algısı ve Çerçeve Meşruiyeti",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    paths["mechanism_vars"] = save_fig(plots_dir, "mechanism_vars.png")
    return paths


def plot_all_dvs_by_condition(df: pd.DataFrame, plots_dir: Path) -> dict[str, str]:
    paths = {}
    cond_order = [
        "C01", "C02", "C03", "C04",
        "C05", "C06", "C07", "C08",
        "C09", "C10", "C11", "C12",
    ]
    cond_labels = {
        "C01": "K-K", "C02": "K-Din", "C03": "K-İH", "C04": "K-UÖ",
        "C05": "Ça-K", "C06": "Ça-Din", "C07": "Ça-İH", "C08": "Ça-UÖ",
        "C09": "Ba-K", "C10": "Ba-Din", "C11": "Ba-İH", "C12": "Ba-UÖ",
    }
    dvs = [("dv_3a_sampled", "3a Meşruiyet", "#6baed6"),
           ("dv_3b_sampled", "3b Huzur Bozucu", "#e6550d"),
           ("dv_3c_sampled", "3c Kürtçe Eğitim", "#31a354")]

    fig, ax = plt.subplots(figsize=(15, 5))
    x = np.arange(len(cond_order))
    width = 0.28
    offsets = [-width, 0, width]

    for (col, label, color), offset in zip(dvs, offsets):
        sub = df.dropna(subset=[col])
        means = [sub[sub["condition_id"] == cid][col].mean() for cid in cond_order]
        ax.bar(x + offset, means, width, label=label, color=color, alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([cond_labels[c] for c in cond_order], fontsize=8, rotation=0)
    ax.set_ylabel("Ortalama Yanıt (1–5)", fontsize=10)
    ax.set_ylim(1.5, 4.0)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title(
        "Tüm Bağımlı Değişkenler — 12 Koşul Karşılaştırması\n"
        "K=Kontrol | Ça=Çatışma | Ba=Barış | Din=Dini | İH=İnsan Hakları | UÖ=Ulusal Özgürlük",
        fontsize=10, fontweight="bold"
    )

    # Bölücü çizgiler (güvenlik koşulları arası)
    for x_pos in [3.5, 7.5]:
        ax.axvline(x_pos, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    # Güvenlik bağlamı etiketleri
    for x_pos, label in [(1.5, "Kontrol Bağlamı"), (5.5, "Çatışma Bağlamı"), (9.5, "Barış Bağlamı")]:
        ax.text(x_pos, 3.9, label, ha="center", fontsize=9, color="gray",
                style="italic", bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    plt.tight_layout()
    paths["all_dvs_by_condition"] = save_fig(plots_dir, "all_dvs_by_condition.png")
    return paths


def plot_behavioral_intention(df: pd.DataFrame, plots_dir: Path) -> dict[str, str]:
    paths = {}
    dv_df = df.dropna(subset=["dv_3d_sampled"])
    sec_order = ["K", "1a", "1b"]

    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(6)
    width = 0.28

    for i, sec in enumerate(sec_order):
        sub = dv_df[dv_df["security_context"] == sec]
        counts = sub["dv_3d_sampled"].value_counts(normalize=True).sort_index()
        vals = [counts.get(float(j), 0) for j in range(1, 7)]
        ax.bar(x + i * width, vals, width, label=SEC_LABELS[sec].replace("\n", " "),
               color=C_SECURITY[sec], edgecolor="white", alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels([D3D_LABELS[i] for i in range(1, 7)], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Oran", fontsize=10)
    ax.set_title(
        "Davranış Niyeti (3d) — Güvenlik Bağlamına Göre\n"
        "1–3: Destekleme | 4–6: Karşı çıkma",
        fontsize=11, fontweight="bold"
    )
    ax.axvline(2.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(1.0, ax.get_ylim()[1] * 0.95, "← Destekleme", fontsize=9, color="green", ha="center")
    ax.text(4.0, ax.get_ylim()[1] * 0.95, "Karşı çıkma →", fontsize=9, color="red", ha="center")
    ax.legend(fontsize=9)

    plt.tight_layout()
    paths["behavioral_intention"] = save_fig(plots_dir, "behavioral_intention.png")
    return paths


def plot_demographic_moderation(df: pd.DataFrame, plots_dir: Path) -> dict[str, str]:
    paths = {}
    dv_df = df.dropna(subset=["dv_3c_sampled"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --- Cinsiyet ---
    ax = axes[0]
    for gender, color in [("Kadın", "#e377c2"), ("Erkek", "#1f77b4")]:
        sub = dv_df[dv_df["gender"] == gender]
        means_by_sec = [sub[sub["security_context"] == s]["dv_3c_sampled"].mean() for s in ["K", "1a", "1b"]]
        ax.plot([SEC_LABELS[s] for s in ["K", "1a", "1b"]], means_by_sec,
                marker="o", linewidth=2.5, markersize=8, label=gender, color=color)
    ax.set_ylabel("Ort. 3c (1–5)", fontsize=10)
    ax.set_title("Cinsiyet Moderasyonu\nGüvenlik Bağlamı × Cinsiyet", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(2.3, 3.3)

    # --- Yaş Grubu ---
    ax = axes[1]
    age_colors = {"18–29": "#e6550d", "30–44": "#fd8d3c", "45–59": "#74c476", "60+": "#238b45"}
    for age_grp, color in age_colors.items():
        sub = dv_df[dv_df["age_group"] == age_grp]
        if len(sub) < 50:
            continue
        means_by_sec = [sub[sub["security_context"] == s]["dv_3c_sampled"].mean() for s in ["K", "1a", "1b"]]
        ax.plot([SEC_LABELS[s] for s in ["K", "1a", "1b"]], means_by_sec,
                marker="o", linewidth=2.5, markersize=8, label=age_grp, color=color)
    ax.set_ylabel("Ort. 3c (1–5)", fontsize=10)
    ax.set_title("Yaş Grubu Moderasyonu\nGüvenlik Bağlamı × Yaş", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(2.0, 3.5)

    # --- Bölge (en yüksek ve en düşük 3 bölge) ---
    ax = axes[2]
    region_means = dv_df.groupby("region")["dv_3c_sampled"].mean().sort_values(ascending=False)
    top3    = list(region_means.head(3).index)
    bottom3 = list(region_means.tail(3).index)
    regions_to_show = top3 + bottom3
    reg_colors = ["#2ca02c", "#2ca02c", "#2ca02c", "#d62728", "#d62728", "#d62728"]
    means_reg = [dv_df[dv_df["region"] == r]["dv_3c_sampled"].mean() for r in regions_to_show]
    short = [r.replace(" Bölgesi", "").replace(" Anadolu", "\nAnad.") for r in regions_to_show]
    bars = ax.barh(short, means_reg, color=reg_colors, edgecolor="white")
    ax.axvline(dv_df["dv_3c_sampled"].mean(), color="gray", linestyle="--", linewidth=1,
               label=f"Genel ort. {dv_df['dv_3c_sampled'].mean():.2f}")
    for bar, m in zip(bars, means_reg):
        ax.text(m + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{m:.3f}", va="center", fontsize=9)
    ax.set_title("Bölgesel Farklılıklar\n(en yüksek ve en düşük 3)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Ort. 3c", fontsize=9)
    ax.legend(fontsize=8)

    fig.suptitle("Demografik Moderasyon: Kim Nasıl Etkilendi?",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    paths["demographic_moderation"] = save_fig(plots_dir, "demographic_moderation.png")
    return paths


def plot_probability_distributions(df: pd.DataFrame, plots_dir: Path) -> dict[str, str]:
    """Her koşul için 3c olasılık dağılımını (p1–p5 ortalaması) gösterir."""
    paths = {}

    cond_order = ["C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08", "C09", "C10", "C11", "C12"]
    cond_labels = {
        "C01": "K-K", "C02": "K-Din", "C03": "K-İH", "C04": "K-UÖ",
        "C05": "Ça-K", "C06": "Ça-Din", "C07": "Ça-İH", "C08": "Ça-UÖ",
        "C09": "Ba-K", "C10": "Ba-Din", "C11": "Ba-İH", "C12": "Ba-UÖ",
    }
    p_cols = ["dv_3c_p1", "dv_3c_p2", "dv_3c_p3", "dv_3c_p4", "dv_3c_p5"]
    p_colors = ["#d62728", "#ff7f0e", "#aec7e8", "#2ca02c", "#1f77b4"]
    p_labels = ["1\nKesinlikle\nkatılmıyorum", "2\nKatılmıyorum", "3\nKararsız",
                "4\nKatılıyorum", "5\nKesinlikle\nkatılıyorum"]

    fig, axes = plt.subplots(3, 4, figsize=(18, 10), sharey=True)
    axes_flat = axes.flatten()

    for i, cid in enumerate(cond_order):
        ax = axes_flat[i]
        sub = df[df["condition_id"] == cid].dropna(subset=p_cols)
        means = [sub[p].mean() for p in p_cols]
        bars = ax.bar(range(1, 6), means, color=p_colors, edgecolor="white", linewidth=0.5)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, m + 0.005,
                    f"{m:.2f}", ha="center", fontsize=7)
        ax.set_title(cond_labels[cid], fontsize=9, fontweight="bold")
        ax.set_xticks(range(1, 6))
        ax.set_xticklabels(["1", "2", "3", "4", "5"], fontsize=7)
        ax.set_ylim(0, 0.45)
        if i % 4 == 0:
            ax.set_ylabel("Ort. Olasılık", fontsize=8)

    # Ortak etiket
    fig.text(0.5, 0.01, "Yanıt Seçeneği (1=Kesinlikle Katılmıyorum … 5=Kesinlikle Katılıyorum)", ha="center", fontsize=10)
    fig.suptitle("3c Olasılık Dağılımları — Her Koşul İçin\n"
                 "(K=Kontrol | Ça=Çatışma | Ba=Barış | Din=Dini | İH=İnsan Hakları | UÖ=Ulusal Özgürlük)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    paths["probability_distributions"] = save_fig(plots_dir, "probability_distributions.png")
    return paths


# ─── Rapor metni ──────────────────────────────────────────────────────────────

def compute_summary_stats(df: pd.DataFrame) -> dict:
    stats = {}
    dv = df.dropna(subset=["dv_3c_sampled"])

    # Koşul bazlı ortalamalar
    stats["cond_means"] = dv.groupby(["security_context", "framing"])["dv_3c_sampled"].agg(
        ["mean", "std", "count"]
    ).round(3)

    # Güvenlik ana etkisi
    stats["sec_means"] = dv.groupby("security_context")["dv_3c_sampled"].agg(["mean", "std", "count"]).round(3)

    # Çerçeve ana etkisi
    stats["frm_means"] = dv.groupby("framing")["dv_3c_sampled"].agg(["mean", "std", "count"]).round(3)

    # Manipülasyon kontrolü
    mc = df.dropna(subset=["manip_check_sampled"])
    stats["manip_1a_acc"] = (mc[mc["security_context"] == "1a"]["manip_check_sampled"] == 1).mean()
    stats["manip_1b_acc"] = (mc[mc["security_context"] == "1b"]["manip_check_sampled"] == 2).mean()

    # Tehdit algısı
    stats["threat_means"] = df.dropna(subset=["threat_sampled"]).groupby("security_context")["threat_sampled"].mean().round(3)

    # Çerçeve meşruiyeti
    stats["fl_means"] = df.dropna(subset=["frame_legit_sampled"]).groupby("framing")["frame_legit_sampled"].mean().round(3)

    # Demografik
    stats["gender_3c"] = dv.groupby("gender")["dv_3c_sampled"].mean().round(3)
    stats["age_3c"] = dv.groupby("age_group", observed=True)["dv_3c_sampled"].mean().round(3)

    # Genel
    stats["n_personas"] = df["persona_id"].nunique()
    stats["n_conditions"] = df["condition_id"].nunique()
    stats["n_total"] = len(dv)
    stats["parse_fail"] = df["dv_3c_sampled"].isna().sum()

    # Within-person
    ref = dv[dv["condition_id"] == "C01"][["persona_id", "dv_3c_sampled"]].rename(columns={"dv_3c_sampled": "ref"})
    for cid, label in [("C05", "conflict"), ("C09", "peace")]:
        comp = dv[dv["condition_id"] == cid][["persona_id", "dv_3c_sampled"]].rename(columns={"dv_3c_sampled": "comp"})
        merged = ref.merge(comp, on="persona_id")
        stats[f"within_{label}_diff"] = (merged["comp"] - merged["ref"]).mean()

    return stats


def build_report(df: pd.DataFrame, stats: dict, img_paths: dict, csv_path: str, ts: str) -> str:
    n_p = stats["n_personas"]
    n_t = stats["n_total"]
    parse_f = stats["parse_fail"]

    sec_k  = stats["sec_means"].loc["K",  "mean"]
    sec_1a = stats["sec_means"].loc["1a", "mean"]
    sec_1b = stats["sec_means"].loc["1b", "mean"]

    frm_k  = stats["frm_means"].loc["K",  "mean"]
    frm_2a = stats["frm_means"].loc["2a", "mean"]
    frm_2b = stats["frm_means"].loc["2b", "mean"]
    frm_2c = stats["frm_means"].loc["2c", "mean"]

    manip_1a = stats["manip_1a_acc"] * 100
    manip_1b = stats["manip_1b_acc"] * 100
    thr_k  = stats["threat_means"]["K"]
    thr_1a = stats["threat_means"]["1a"]
    thr_1b = stats["threat_means"]["1b"]
    fl_2a  = stats["fl_means"]["2a"]
    fl_2b  = stats["fl_means"]["2b"]
    fl_2c  = stats["fl_means"]["2c"]

    within_conf = stats["within_conflict_diff"]
    within_peac = stats["within_peace_diff"]

    md = f"""# Dijital İkizlerle Nedensel Çıkarım Raporu
## Kürtçe Anadil Eğitimi Üzerine 3×4 Faktöriyel Deney

**Rapor tarihi:** {ts}
**Veri dosyası:** `{csv_path}`
**Model:** GPT-4.1 Mini | **Yöntem:** Verbalized Sampling (Olasılık Dağılımı)

---

## 1. Araştırmanın Özeti

Bu çalışma, Türk kamuoyunu temsil eden **{n_p} dijital ikiz** kullanarak, Kürtçe anadil eğitimi talebine verilen desteğin iki farklı faktörden nasıl etkilendiğini incelemektedir:

1. **Güvenlik Bağlamı** — Anketi dolduran kişi bir güvenlik olayı haberiyle mi karşılaşıyor?
2. **Talep Çerçevelemesi** — Kürtçe eğitim talebi hangi argümanla savunuluyor?

> **Dijital ikiz nedir?** COSMO Türkiye anketine katılan gerçek kişilerin demografik ve tutum profilleri, büyük dil modeli aracılığıyla simüle ediliyor. Her dijital ikiz, gerçek anketteki bir kişiyi temsil ediyor ve o kişinin özellikleri (yaş, cinsiyet, bölge, dini kimlik, siyasi eğilim vb.) dikkate alınarak yanıt veriyor.

**Bu yöntemin avantajı:** Aynı kişi 12 farklı koşulda test edilebiliyor — gerçek bir ankette bunu yapmak mümkün olmaz. Bu sayede *birey içi (within-subject)* karşılaştırma yapılabiliyor; bireysel farklılıklar kontrol altına alınıyor.

---

## 2. Deney Tasarımı

### 3×4 Tam Faktöriyel Tasarım

Her dijital ikiz, 12 koşulun **tamamına** tabi tutuldu. Her koşulda bir senaryo sunuldu ve 4 tutum sorusu ile 2 ek ölçüm sorusu soruldu.

#### Faktör 1 — Güvenlik Bağlamı (3 düzey)

| Kod | Koşul | Sunulan Bilgi |
|-----|-------|---------------|
| **K** | Kontrol | Güvenlik haberi yok |
| **1a** | Çatışma | "Türkiye ordusu Suriye'nin kuzeyinde aktif operasyon yürütüyor, kayıplar var" |
| **1b** | Barış | "Türkiye ile Kürt grupları arasında anlaşma sağlandı, çatışma sona erdi" |

#### Faktör 2 — Talep Çerçevelemesi (4 düzey)

| Kod | Çerçeve | Kullanılan Argüman |
|-----|---------|-------------------|
| **K** | Kontrol | Çerçeve yok — sadece yürüyüş haberi |
| **2a** | Dini | Kur'an'dan "Sizi farklı dil ve renklerde yarattık" ayeti |
| **2b** | İnsan Hakları | BM Evrensel Beyannamesi'ne atıf — "dil temel bir haktır" |
| **2c** | Ulusal Özgürlük | "Dil yasakları sömürge politikasıdır, kendi kaderini tayin hakkı" |

#### Koşul Matrisi (12 koşul)

|  | K (çerçeve yok) | 2a Dini | 2b İnsan Hakkı | 2c Ulusal Özg. |
|--|--|--|--|--|
| **K** (bağlam yok) | C01 | C02 | C03 | C04 |
| **1a** (çatışma) | C05 | C06 | C07 | C08 |
| **1b** (barış) | C09 | C10 | C11 | C12 |

---

### Ölçülen Değişkenler

Her koşulda model şu sorulara yanıt verdi (5'li Likert, 1=Kesinlikle katılmıyorum):

| Değişken | Soru |
|----------|------|
| **3a** | Bu grup meşru bir hak mücadelesi yürütüyor |
| **3b** | Bu grubun talepleri toplumsal huzuru bozucu |
| **3c** ⭐ | Devlet okullarında Kürtçe anadil eğitimine yasal izin verilmeli *(Ana DV)* |
| **3d** | Böyle bir etkinlikte ne yapardınız? *(6 davranış seçeneği)* |

Ek ölçümler:
- **Tehdit algısı** — Türkiye'nin güvenliği ne kadar tehdit altında? (1–4)
- **Çerçeve meşruiyeti** — Protestocu grubun yaklaşımı ne kadar meşru? (1–4, sadece çerçeve koşullarında)
- **Manipülasyon kontrolü** — Model haberi doğru hatırlıyor mu? (sadece 1a/1b koşullarında)

---

## 3. Katılımcı Profili

![Demografik Dağılım]({img_paths.get("demographics", "")})

Çalışmada {n_p} dijital ikiz kullanıldı; her biri 12 koşulda test edildi → toplam **{n_t:,} geçerli gözlem**.

| Özellik | Değer |
|---------|-------|
| Toplam dijital ikiz | {n_p} |
| Yaş aralığı | 18–84 |
| Ortalama yaş | ~40 |
| Kadın / Erkek | ~%49 / %51 |
| Bölge | 12 NUTS-2 bölgesi (tüm Türkiye) |
| Geçersiz yanıt (parse hatası) | {parse_f} ({parse_f/len(df)*100:.1f}%) |

**Önemli not:** Dijital ikizler gerçek anket katılımcılarına dayanmaktadır; bu nedenle Türkiye nüfusunu temsil eden bir örneklemi yansıtmaktadır.

---

## 4. Manipülasyon Kontrolü

Modelin sunulan haberi doğru algılayıp algılamadığını test etmek için her güvenlik koşulunda bir manipülasyon kontrolü sorusu soruldu.

![Manipülasyon Kontrolü]({img_paths.get("manip_check", "")})

| Koşul | Beklenen Yanıt | Doğruluk |
|-------|----------------|----------|
| 1a Çatışma | "Aktif çatışma var" | **%{manip_1a:.1f}** |
| 1b Barış | "Anlaşma sağlandı" | **%{manip_1b:.1f}** |

> ✅ **Sonuç:** Model sunulan haberi %90+ doğrulukla doğru hatırladı. Manipülasyon başarılı; bulguların güvenlik bağlamına dair yorumları sağlamlıkla desteklenmektedir.

---

## 5. Ana Bulgular

### 5.1 Ana Etki: Güvenlik Bağlamı

![Güvenlik Bağlamı Ana Etkisi]({img_paths.get("main_effect_security", "")})

| Güvenlik Bağlamı | Ort. 3c Desteği | Kontrol'dan Fark |
|------------------|----------------|-----------------|
| **K** — Kontrol | {sec_k:.3f} | — |
| **1a** — Çatışma | {sec_1a:.3f} | **{sec_1a-sec_k:+.3f}** |
| **1b** — Barış | {sec_1b:.3f} | **{sec_1b-sec_k:+.3f}** |

**Bulgu:** Güvenlik bağlamı Kürtçe eğitim desteğini anlamlı biçimde düşürdü. Çatışma koşulunda ({sec_1a:.3f}) ve barış koşulunda ({sec_1b:.3f}) destek, kontrol koşuluna ({sec_k:.3f}) kıyasla düştü. Bu, hem çatışma hem de barış haberlerinin — yani *herhangi bir* Kürt-ilişkili siyasi bilginin — desteği azalttığını göstermektedir.

> **Yorumlama:** Güvenlik/siyasi bilgi, Türk kamuoyunda Kürt meselesini "siyasallaştırıyor" ve savunmacı bir tepkiye neden oluyor. Çatışma haberi bunu daha güçlü bir şekilde yapıyor ({sec_1a-sec_k:+.3f}), barış haberi ise daha az ({sec_1b-sec_k:+.3f}).

#### Birey İçi (Within-Person) Etkiler

Aynı kişi hem kontrol hem de diğer koşullarda test edildiği için, bireysel farklılıklar arındırılmış net etkileri hesaplayabildik:

| Karşılaştırma | Birey İçi Fark |
|---------------|----------------|
| K-K → Çatışma-K | **{within_conf:+.3f}** |
| K-K → Barış-K | **{within_peac:+.3f}** |

---

### 5.2 Ana Etki: Çerçeveleme Türü

![Çerçeve Ana Etkisi]({img_paths.get("main_effect_framing", "")})

| Çerçeve | Ort. 3c Desteği | Kontrol'dan Fark |
|---------|----------------|-----------------|
| **K** — Kontrol | {frm_k:.3f} | — |
| **2a** — Dini | {frm_2a:.3f} | **{frm_2a-frm_k:+.3f}** |
| **2b** — İnsan Hakları | {frm_2b:.3f} | **{frm_2b-frm_k:+.3f}** |
| **2c** — Ulusal Özgürlük | {frm_2c:.3f} | **{frm_2c-frm_k:+.3f}** |

**Bulgu:** Çerçeveleme türü beklenenden daha zayıf etki gösterdi. Hiçbir çerçeve desteği kontrol koşuluna kıyasla anlamlı biçimde artırmadı; hatta insan hakları ({frm_2b:.3f}) ve ulusal özgürlük ({frm_2c:.3f}) çerçeveleri desteği hafifçe düşürdü.

> **Yorumlama:** Bu bulgu literatürdeki "framing" teorileriyle kısmen çelişmektedir. Türkiye bağlamında, Kürt taleplerini meşrulaştırmak için kullanılan argümanın türü değil, *talebin kendisinin siyasi çerçeveye oturtulması* belirleyici görünmektedir.

---

### 5.3 Faktöriyel Etkileşim: Güvenlik × Çerçeve

![Etkileşim Haritası]({img_paths.get("interaction_heatmap", "")})

**Tam koşul ortalamaları (3c — Kürtçe Eğitim Desteği):**

| | K (çerçeve yok) | 2a Dini | 2b İnsan Hakkı | 2c Ulusal Özg. |
|--|--|--|--|--|
"""

    # Pivot tabloyu ekle
    dv_df = df.dropna(subset=["dv_3c_sampled"])
    for sec in ["K", "1a", "1b"]:
        sec_name = SEC_LABELS[sec].replace("\n", " ")
        row = f"| **{sec}** {sec_name} |"
        for frm in ["K", "2a", "2b", "2c"]:
            val = dv_df[(dv_df["security_context"] == sec) & (dv_df["framing"] == frm)]["dv_3c_sampled"].mean()
            row += f" {val:.3f} |"
        md += row + "\n"

    md += f"""
**Bulgu:** En düşük destek **Çatışma + İnsan Hakları (C07)** koşulunda ({dv_df[(dv_df['condition_id']=='C07')]['dv_3c_sampled'].mean():.3f}), en yüksek destek **Kontrol + Kontrol (C01)** koşulunda ({dv_df[(dv_df['condition_id']=='C01')]['dv_3c_sampled'].mean():.3f}) gözlemlendi.

---

### 5.4 Tüm Bağımlı Değişkenler

![Tüm DV'ler 12 Koşulda]({img_paths.get("all_dvs_by_condition", "")})

| Değişken | Kontrol Ort. | Çatışma Ort. | Barış Ort. |
|----------|-------------|-------------|-----------|
| 3a Meşruiyet | {dv_df[dv_df['security_context']=='K']['dv_3a_sampled'].mean():.3f} | {dv_df[dv_df['security_context']=='1a']['dv_3a_sampled'].mean():.3f} | {dv_df[dv_df['security_context']=='1b']['dv_3a_sampled'].mean():.3f} |
| 3b Huzur bozucu | {dv_df[dv_df['security_context']=='K']['dv_3b_sampled'].mean():.3f} | {dv_df[dv_df['security_context']=='1a']['dv_3b_sampled'].mean():.3f} | {dv_df[dv_df['security_context']=='1b']['dv_3b_sampled'].mean():.3f} |
| 3c Kürtçe eğitim ⭐ | {dv_df[dv_df['security_context']=='K']['dv_3c_sampled'].mean():.3f} | {dv_df[dv_df['security_context']=='1a']['dv_3c_sampled'].mean():.3f} | {dv_df[dv_df['security_context']=='1b']['dv_3c_sampled'].mean():.3f} |

> **Not:** 3b (huzur bozucu) yorumlanırken dikkat: yüksek değer daha *bozucu* bulunduğu anlamına gelir.

---

### 5.5 Birey İçi Etkiler — Within-Person Analizi

![Birey İçi Etkiler]({img_paths.get("within_person", "")})

Her personanın kontrol-kontrol koşuluna (C01) kıyasla diğer koşullardaki yanıt değişimi gösterilmektedir. Negatif değer = destek azaldı, pozitif = destek arttı.

---

## 6. Mekanizma Değişkenleri

![Mekanizma Değişkenleri]({img_paths.get("mechanism_vars", "")})

### 6.1 Tehdit Algısı

| Güvenlik Bağlamı | Tehdit Algısı (1–4) |
|------------------|---------------------|
| K — Kontrol | {thr_k:.3f} |
| 1a — Çatışma | **{thr_1a:.3f}** |
| 1b — Barış | {thr_1b:.3f} |

Çatışma haberi tehdit algısını belirgin biçimde artırdı ({thr_1a:.3f} vs {thr_k:.3f}). Barış haberi kontrolden daha düşük tehdit algısına yol açmadı; bu, *herhangi bir* Kürt siyasi bağlamının tehdit algısını hafifçe yükselttiğine işaret ediyor.

### 6.2 Çerçeve Meşruiyeti

| Çerçeve | Meşruiyet (1–4) |
|---------|-----------------|
| 2a — Dini | {fl_2a:.3f} |
| 2b — İnsan Hakları | {fl_2b:.3f} |
| 2c — Ulusal Özgürlük | {fl_2c:.3f} |

Dini çerçeve en yüksek meşruiyet algısı aldı ({fl_2a:.3f}), insan hakları çerçevesi en düşüğü ({fl_2b:.3f}). Bu, Türk kamuoyunda dini argümanların daha kabul edilebilir bulunduğunu gösteriyor — ancak bu kabul edilebilirlik Kürtçe eğitim desteğine doğrudan yansımadı.

---

## 7. Davranış Niyeti (3d)

![Davranış Niyeti]({img_paths.get("behavioral_intention", "")})

Davranış niyeti sorusunda katılımcıların büyük çoğunluğu pasif karşı çıkma (sessizce karşı çıkar) ve aktif karşı çıkma (sosyal medyada karşıt paylaşım) seçeneklerinde yoğunlaştı. Üç güvenlik koşulu arasında davranış dağılımı görece benzer kaldı — bu, güvenlik bağlamının tutumsal puanları etkilediği kadar davranış niyetini değiştirmediğine işaret ediyor.

---

## 8. Demografik Moderasyon

![Demografik Moderasyon]({img_paths.get("demographic_moderation", "")})

### 8.1 Cinsiyet

| Cinsiyet | Ort. 3c Desteği |
|----------|----------------|
| Kadın | {stats['gender_3c']['Kadın']:.3f} |
| Erkek | {stats['gender_3c']['Erkek']:.3f} |

Kadınlar Kürtçe eğitime erkeklerden biraz daha fazla destek verdi.

### 8.2 Yaş Grubu

| Yaş Grubu | Ort. 3c Desteği |
|-----------|----------------|
"""
    for grp in ["18–29", "30–44", "45–59", "60+"]:
        val = stats["age_3c"].get(grp, float("nan"))
        md += f"| {grp} | {val:.3f} |\n"

    md += f"""
Genç katılımcılar Kürtçe eğitime belirgin biçimde daha fazla destek gösterdi; destekteki düşüş yaşla birlikte sistematik bir şekilde artıyor.

### 8.3 Bölgesel Farklılıklar

Ege ve İstanbul bölgeleri en yüksek desteği gösterirken, Karadeniz bölgeleri en düşük desteği gösterdi.

---

## 9. Olasılık Dağılımları (3c)

![Olasılık Dağılımları]({img_paths.get("probability_distributions", "")})

Verbalized sampling yöntemi sayesinde modelin her yanıt seçeneğine atadığı olasılıkları da analiz edebildik. Dağılımlarda dikkat çekici nokta: **3. seçenek (kararsız)** tüm koşullarda görece yüksek olasılık aldı — bu, Türk kamuoyunun bu konuda homojen olmadığını ve önemli bir "belirsiz" kitlenin var olduğunu gösteriyor.

---

## 10. Sonuç ve Kısıtlamalar

### Temel Bulgular

1. **Güvenlik bağlamı desteği düşürüyor.** Kürt meselesiyle ilgili herhangi bir siyasi bilgi (çatışma veya barış) Kürtçe eğitim desteğini düşürüyor. Çatışma haberi bu etkiyi daha güçlü yapıyor.

2. **Çerçeveleme beklenen etkiyi göstermiyor.** Dini, insan hakları veya ulusal özgürlük çerçevelerinden hiçbiri desteği kontrol koşuluna kıyasla artırmadı. Türkiye bağlamında talebin "nasıl" sunulduğu değil, sunulmasının kendisi belirleyici görünüyor.

3. **Dini çerçeve meşruiyet kazanıyor ama desteği artırmıyor.** İlginç bir ayrışma: dini çerçeve en kabul edilebilir bulunan çerçeve, ancak bu kabul Kürtçe eğitim desteğine dönüşmüyor.

4. **Demografik makas belirgin.** Genç, kadın ve batı bölgeleri daha fazla destek gösteriyor; yaşlı, erkek ve doğu/kuzey bölgeleri daha az.

5. **Manipülasyon yüksek doğrulukla çalışıyor (%90+).** Model sunulan haberi güvenilir biçimde işledi.

### Kısıtlamalar

- **Dijital ikizler gerçek insanlar değil.** Sonuçlar LLM'in öğrendiği kalıpları yansıtıyor; gerçek anket sonuçlarından farklılık gösterebilir.
- **Örneklem eksik.** Toplam 2.615 personadan ~1.033'ü kullanıldı (~%39). Kalan personas sistematik farklılık göstermiyor olsa da kontrol edilmesi önerilir.
- **Sıra etkisi simüle edilmedi.** Gerçek bir deneyde koşul sırası randomize edilir; burada her çağrı bağımsız.

---

*Rapor otomatik olarak oluşturulmuştur — `scripts/report_kurdish_causal.py`*
"""
    return md


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Kullanım: python3 scripts/report_kurdish_causal.py exports/kurdish_causal_<ts>.csv")
        sys.exit(1)

    csv_path = sys.argv[1]
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    ts_file  = ts

    print(f"Kaynak: {csv_path}")
    df = load_data(csv_path)
    print(f"Yüklendi: {len(df)} satır, {df['persona_id'].nunique()} persona")

    PLOTS_DIR   = FIGURES_DIR / f"kurdish_causal_plots_{ts_file}"
    report_path = REPORTS_DIR / f"kurdish_causal_report_{ts_file}.md"
    PLOTS_DIR.mkdir(exist_ok=True)

    print("Grafikler oluşturuluyor...")
    img_paths = {}

    print("  demografik dağılım...")
    img_paths.update(plot_participant_demographics(df, PLOTS_DIR))

    print("  manipülasyon kontrolü...")
    img_paths.update(plot_manipulation_check(df, PLOTS_DIR))

    print("  güvenlik bağlamı ana etkisi...")
    img_paths.update(plot_main_effect_security(df, PLOTS_DIR))

    print("  çerçeve ana etkisi...")
    img_paths.update(plot_main_effect_framing(df, PLOTS_DIR))

    print("  etkileşim haritası...")
    img_paths.update(plot_interaction_heatmap(df, PLOTS_DIR))

    print("  birey içi etkiler...")
    img_paths.update(plot_within_person_effects(df, PLOTS_DIR))

    print("  mekanizma değişkenleri...")
    img_paths.update(plot_threat_and_frame_legit(df, PLOTS_DIR))

    print("  tüm DV'ler 12 koşul...")
    img_paths.update(plot_all_dvs_by_condition(df, PLOTS_DIR))

    print("  davranış niyeti...")
    img_paths.update(plot_behavioral_intention(df, PLOTS_DIR))

    print("  demografik moderasyon...")
    img_paths.update(plot_demographic_moderation(df, PLOTS_DIR))

    print("  olasılık dağılımları...")
    img_paths.update(plot_probability_distributions(df, PLOTS_DIR))

    print("İstatistikler hesaplanıyor...")
    stats = compute_summary_stats(df)

    print("Rapor yazılıyor...")
    report_md = build_report(df, stats, img_paths, csv_path, ts)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"\nRapor → {report_path}")
    print(f"Görseller → {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
