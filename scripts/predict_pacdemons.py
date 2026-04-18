"""
Batch script: pacdemons tahmini
- Model: gpt-5.4-mini, temperature: 0.8
- Veri: personas_rewritten_v2.json (pacdemons olan tüm personalar, ~1707)
- Çıktı: exports/pacdemons_predictions_<timestamp>.csv + .json
- Kaldığı yerden devam (checkpoint) destekli
"""

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

# ─── Sabitler ────────────────────────────────────────────────────────────────

MODEL = "gpt-5.4-mini"
TEMPERATURE = 0.8
MAX_TOKENS = 80
RETRY_LIMIT = 3
RETRY_DELAY = 5   # saniye
MAX_WORKERS = 10  # paralel istek sayısı

BASE_DIR = Path(__file__).parent.parent
PERSONAS_PATH = BASE_DIR / "personas_rewritten_v2.json"
EXPORTS_DIR = BASE_DIR / "exports"
EXPORTS_DIR.mkdir(exist_ok=True)

TURKEY_CONTEXT = (
    "Türkiye'nin ideolojik yapısı iki temel eksen üzerinde şekillenir: sol-sağ siyasi ekseni ve "
    "seküler-dindar kültürel ekseni. Sol tarafta sosyalizm, feminizm, sosyal demokrasi ve Kürt ulusal "
    "hareketi birbirine yakın konumlanır. Sağ tarafta Türk milliyetçiliği ve Kemalizm yer alır. "
    "Muhafazakârlık ve İslamcılık kültürel eksenin dindar ucunu oluşturur. "
    "Türkiye'de bireylerin sokak eylemlerine katılım kararı, yalnızca bireysel tercihlerin değil, "
"aynı zamanda bölgesel bağlamın, siyasi parti aidiyetinin ve kimlik temelli aidiyetlerin etkisi "
"altında şekillenebilir. Farklı bölgelerde yaşayan bireyler, içinde bulundukları sosyal ve politik "
"çevreye bağlı olarak protesto eylemlerine farklı düzeylerde maruz kalabilir ve bu da katılım "
"olasılıklarını etkileyebilir. Benzer şekilde, bireylerin siyasi kimlikleri ve kendilerini yakın "
"hissettikleri politik hareketler, sokak eylemlerine katılım konusundaki tutumlarını "
"şekillendirebilir."
)

QUESTION = (
    "Aşağıdaki soru son 12 ay içindeki siyasi katılımınla ilgilidir. "
    "Kendi gerçek deneyimine ve değerlerine göre dürüstçe yanıtla.\n\n"
    "Yanıtını SADECE şu JSON formatında ver, başka hiçbir şey yazma:\n"
    '{"pacdemons": <1 veya 2>}\n\n'
    "Soru (son 12 ay içinde Türkiye'de işlerin düzelmesi için):\n"
    "Sokak eylemlerine (gösteri, yürüyüş vb.) katıldın mı?  1=Evet  2=Hayır"
)

# ─── API ─────────────────────────────────────────────────────────────────────

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

def get_client() -> OpenAI:
    key = OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise ValueError("API anahtarı bulunamadı.")
    return OpenAI(api_key=key)


def parse_response(text: str) -> int | None:
    if not text:
        return None
    text = text.strip()
    for src in [text, re.search(r"\{[^{}]*\}", text, re.DOTALL).group(0) if re.search(r"\{[^{}]*\}", text) else ""]:
        try:
            obj = json.loads(src)
            val = obj.get("pacdemons")
            if val in (1, 2):
                return int(val)
            if str(val) in ("1", "2"):
                return int(val)
        except Exception:
            pass
    m = re.search(r'"pacdemons"\s*:\s*([12])', text)
    if m:
        return int(m.group(1))
    return None


def call_api(client: OpenAI, system_prompt: str) -> tuple[int | None, str, int, int]:
    full_system = f"{TURKEY_CONTEXT.strip()}\n\n{system_prompt}"
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": full_system},
                    {"role": "user", "content": QUESTION},
                ],
                temperature=TEMPERATURE,
                max_completion_tokens=MAX_TOKENS,
            )
            raw = resp.choices[0].message.content or ""
            tok_p = getattr(resp.usage, "prompt_tokens", 0)
            tok_c = getattr(resp.usage, "completion_tokens", 0)
            return parse_response(raw), raw, tok_p, tok_c
        except Exception as exc:
            print(f"  [Hata] deneme {attempt}/{RETRY_LIMIT}: {exc}")
            if attempt < RETRY_LIMIT:
                time.sleep(RETRY_DELAY * attempt)
    return None, "API_ERROR", 0, 0


# ─── Ana akış ────────────────────────────────────────────────────────────────

def load_personas() -> list[dict]:
    with open(PERSONAS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return [p for p in data if "pacdemons" in p.get("ground_truth", {})]


def run(checkpoint_path: Path | None = None):
    personas = load_personas()
    print(f"pacdemons olan persona sayısı: {len(personas)}")

    done: dict[str, dict] = {}
    if checkpoint_path and checkpoint_path.exists():
        with open(checkpoint_path, encoding="utf-8") as f:
            for rec in json.load(f):
                done[rec["persona_id"]] = rec
        print(f"Checkpoint yüklendi: {len(done)} persona zaten tamamlanmış, atlanıyor.")

    client = get_client()
    results: list[dict] = list(done.values())
    remaining = [p for p in personas if p["persona_id"] not in done]
    total = len(personas)
    checkpoint_lock = threading.Lock()
    completed_count = len(done)

    print(f"İşlenecek: {len(remaining)} persona | workers: {MAX_WORKERS}\n")

    def process(persona: dict) -> dict:
        pid = persona["persona_id"]
        gt_val = persona["ground_truth"].get("pacdemons")
        pred, raw, tok_p, tok_c = call_api(client, persona.get("system_prompt", ""))
        return {
            "persona_id": pid,
            "qform": persona.get("qform"),
            "age": persona.get("age"),
            "gender": persona.get("gender"),
            "region": persona.get("region"),
            "pacdemons": gt_val,
            "gt_pacdemons": gt_val,
            "pred_pacdemons": pred,
            "match": int(pred == gt_val) if pred is not None and gt_val is not None else None,
            "raw_response": raw,
            "token_prompt": tok_p,
            "token_completion": tok_c,
        }

    pbar = tqdm(total=len(remaining), desc="Tahmin", unit="persona")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process, p): p for p in remaining}
        for future in as_completed(futures):
            rec = future.result()
            match_sym = "✓" if rec["match"] == 1 else ("✗" if rec["match"] == 0 else "?")
            pbar.set_postfix({"id": rec["persona_id"], "gt": rec["gt_pacdemons"],
                              "pred": rec["pred_pacdemons"], "match": match_sym})
            pbar.update(1)

            with checkpoint_lock:
                results.append(rec)
                completed_count += 1
                if checkpoint_path and completed_count % 50 == 0:
                    with open(checkpoint_path, "w", encoding="utf-8") as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
                    tqdm.write(f"  → Checkpoint kaydedildi ({completed_count}/{total})")

    pbar.close()

    if checkpoint_path:
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    return results


def save_results(results: list[dict], ts: str):
    csv_path = EXPORTS_DIR / f"pacdemons_predictions_{ts}.csv"
    json_path = EXPORTS_DIR / f"pacdemons_predictions_{ts}.json"

    # JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # CSV (basit, pandas gerektirmez)
    if results:
        headers = list(results[0].keys())
        with open(csv_path, "w", encoding="utf-8-sig") as f:
            f.write(",".join(headers) + "\n")
            for rec in results:
                row = []
                for h in headers:
                    val = rec.get(h, "")
                    s = "" if val is None else str(val)
                    if "," in s or '"' in s or "\n" in s:
                        s = '"' + s.replace('"', '""') + '"'
                    row.append(s)
                f.write(",".join(row) + "\n")

    return csv_path, json_path


def print_summary(results: list[dict]):
    from collections import Counter

    valid = [r for r in results if r["match"] is not None]
    if not valid:
        print("Geçerli sonuç yok.")
        return

    gt_vals   = [r["gt_pacdemons"]   for r in valid]
    pred_vals = [r["pred_pacdemons"] for r in valid]
    parse_fail = sum(1 for r in results if r["pred_pacdemons"] is None)

    accuracy = sum(r["match"] for r in valid) / len(valid)
    labels = sorted(set(gt_vals) | set(pred_vals))  # [1, 2]

    # Confusion matrix
    cm: dict[tuple, int] = {}
    for gt, pr in zip(gt_vals, pred_vals):
        cm[(gt, pr)] = cm.get((gt, pr), 0) + 1

    # Per-class metrics
    metrics = {}
    for lbl in labels:
        tp = cm.get((lbl, lbl), 0)
        fp = sum(cm.get((g, lbl), 0) for g in labels if g != lbl)
        fn = sum(cm.get((lbl, p), 0) for p in labels if p != lbl)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        metrics[lbl] = {"tp": tp, "fp": fp, "fn": fn,
                        "precision": precision, "recall": recall, "f1": f1}

    label_names = {1: "Evet (1)", 2: "Hayır (2)"}

    print("\n" + "=" * 56)
    print("ÖZET")
    print("=" * 56)
    print(f"Toplam persona      : {len(results)}")
    print(f"Geçerli tahmin      : {len(valid)}")
    print(f"Ayrıştırma hatası   : {parse_fail}")
    print(f"Accuracy            : {accuracy:.4f}  ({sum(r['match'] for r in valid)}/{len(valid)})")

    print(f"\nGround truth dağılımı : {dict(Counter(gt_vals))}")
    print(f"Tahmin dağılımı       : {dict(Counter(pred_vals))}")

    # Confusion matrix tablosu
    col_w = 12
    print("\n── Confusion Matrix ──────────────────────────────────")
    header = " " * 16 + "".join(f"Pred {label_names.get(l, l):>{col_w-5}}" for l in labels)
    print(header)
    print(" " * 16 + "-" * (col_w * len(labels)))
    for gt in labels:
        row = f"GT {label_names.get(gt, gt):<13}"
        for pr in labels:
            row += f"{cm.get((gt, pr), 0):>{col_w}}"
        print(row)

    # Per-class metrics tablosu
    print("\n── Per-class Metrics ─────────────────────────────────")
    print(f"{'Sınıf':<14} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 56)
    for lbl in labels:
        support = sum(1 for g in gt_vals if g == lbl)
        m = metrics[lbl]
        print(f"{label_names.get(lbl, lbl):<14} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {support:>10}")

    # Macro / weighted avg
    macro_p  = sum(metrics[l]["precision"] for l in labels) / len(labels)
    macro_r  = sum(metrics[l]["recall"]    for l in labels) / len(labels)
    macro_f1 = sum(metrics[l]["f1"]        for l in labels) / len(labels)
    total_support = len(valid)
    w_p  = sum(metrics[l]["precision"] * sum(1 for g in gt_vals if g == l) for l in labels) / total_support
    w_r  = sum(metrics[l]["recall"]    * sum(1 for g in gt_vals if g == l) for l in labels) / total_support
    w_f1 = sum(metrics[l]["f1"]        * sum(1 for g in gt_vals if g == l) for l in labels) / total_support
    print("-" * 56)
    print(f"{'Macro avg':<14} {macro_p:>10.4f} {macro_r:>10.4f} {macro_f1:>10.4f} {total_support:>10}")
    print(f"{'Weighted avg':<14} {w_p:>10.4f} {w_r:>10.4f} {w_f1:>10.4f} {total_support:>10}")
    print("=" * 56)


if __name__ == "__main__":
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = EXPORTS_DIR / "pacdemons_checkpoint.json"

    print(f"Model: {MODEL} | Temperature: {TEMPERATURE}")
    print(f"Checkpoint: {checkpoint_path}\n")

    results = run(checkpoint_path=checkpoint_path)

    csv_path, json_path = save_results(results, ts)
    print(f"\nCSV  → {csv_path}")
    print(f"JSON → {json_path}")

    print_summary(results)

    # Tamamlandıktan sonra checkpoint'i temizle
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print("Checkpoint temizlendi.")
