"""
Batch script: neilang tahmini (farklı dil konuşan komşuya yaklaşım)
- Model: gpt-5.4-mini, temperature: 0.8
- Veri: personas_rewritten_v2.json (neilang olan tüm personalar, ~1692)
- Prompt temizleme: neilang'ı sızdıran "farklı dil/dilden komşu" cümleleri kaldırılır
- Çıktı: exports/neilang_predictions_<timestamp>.csv + .json
- Kaldığı yerden devam (checkpoint) destekli, paralel istek
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

MODEL       = "gpt-5.4-mini"
TEMPERATURE = 0.8
MAX_TOKENS  = 80
RETRY_LIMIT = 3
RETRY_DELAY = 5
MAX_WORKERS = 10

BASE_DIR      = Path(__file__).parent.parent
PERSONAS_PATH = BASE_DIR / "personas_rewritten_v2.json"
EXPORTS_DIR   = BASE_DIR / "exports"
EXPORTS_DIR.mkdir(exist_ok=True)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

TURKEY_CONTEXT = (
    "Türkiye'nin ideolojik yapısı iki temel eksen üzerinde şekillenir: sol-sağ siyasi ekseni ve "
    "seküler-dindar kültürel ekseni. Sol tarafta sosyalizm, feminizm, sosyal demokrasi ve Kürt ulusal "
    "hareketi birbirine yakın konumlanır. Sağ tarafta Türk milliyetçiliği ve Kemalizm yer alır. "
    "Muhafazakârlık ve İslamcılık kültürel eksenin dindar ucunu oluşturur."
)

QUESTION = (
    "Aşağıdaki soru komşuluk ilişkilerine dair bir tutum sorusudur. "
    "Kendi gerçek değerlerine ve bakış açına göre dürüstçe yanıtla.\n\n"
    "Yanıtını SADECE şu JSON formatında ver, başka hiçbir şey yazma:\n"
    '{"neilang": <1, 2 veya 3>}\n\n'
    "Soru:\n"
    "Seninle farklı bir dil konuşan biri komşun olsaydı nasıl hissederdin?\n"
    "  1 = Hiç rahatsız olmazdım\n"
    "  2 = Biraz rahatsız olurdum\n"
    "  3 = Çok rahatsız olurdum"
)

# ─── Prompt temizleme ─────────────────────────────────────────────────────────
#
# Kaldırılan bilgiler:
#   - neilang : "farklı dil konuşan / farklı dilden" komşu ifadeleri  (~20 persona)
#   - neirelg : "farklı dinî görüşten / farklı dinden" komşu ifadeleri (~967 persona)
#
# Strateji: komşu-tutum bağlamındaki cümleleri tamamen kaldır.

# neilang: dil+komşu cümleleri
_FULL_SENT_LANG = re.compile(
    r'[^.;!?\n]*(?:farklı\s+dil(?:\s+konuşan)?|farklı\s+dilden)[^.;!?\n]*[.;!?]?',
    re.IGNORECASE,
)

# neirelg: din+komşu cümleleri
# "farklı dinî görüş", "farklı dinden", "farklı inanç ... komşu"
_FULL_SENT_REL = re.compile(
    r'[^.;!?\n]*(?:farklı\s+din[îi]?\s+görüş|farklı\s+dinden|farklı\s+inanç)[^.;!?\n]*[.;!?]?',
    re.IGNORECASE,
)

_DOUBLE_SPACE = re.compile(r'  +')


def clean_prompt(prompt: str) -> str:
    cleaned = _FULL_SENT_LANG.sub('', prompt)
    cleaned = _FULL_SENT_REL.sub('', cleaned)
    cleaned = _DOUBLE_SPACE.sub(' ', cleaned)
    lines = [l.strip() for l in cleaned.splitlines()]
    return '\n'.join(l for l in lines if l)


# ─── API ─────────────────────────────────────────────────────────────────────

def get_client() -> OpenAI:
    key = OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise ValueError("API anahtarı bulunamadı.")
    return OpenAI(api_key=key)


def parse_response(text: str) -> int | None:
    if not text:
        return None
    for src in [text.strip(),
                (re.search(r'\{[^{}]*\}', text, re.DOTALL) or type('', (), {'group': lambda s, x: ''})()).group(0)]:
        try:
            obj = json.loads(src)
            val = obj.get("neilang")
            if val in (1, 2, 3):
                return int(val)
            if str(val) in ("1", "2", "3"):
                return int(val)
        except Exception:
            pass
    m = re.search(r'"neilang"\s*:\s*([123])', text)
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
                    {"role": "user",   "content": QUESTION},
                ],
                temperature=TEMPERATURE,
                max_completion_tokens=MAX_TOKENS,
            )
            raw   = resp.choices[0].message.content or ""
            tok_p = getattr(resp.usage, "prompt_tokens",      0)
            tok_c = getattr(resp.usage, "completion_tokens",  0)
            return parse_response(raw), raw, tok_p, tok_c
        except Exception as exc:
            tqdm.write(f"  [Hata] deneme {attempt}/{RETRY_LIMIT}: {exc}")
            if attempt < RETRY_LIMIT:
                time.sleep(RETRY_DELAY * attempt)
    return None, "API_ERROR", 0, 0


# ─── Ana akış ────────────────────────────────────────────────────────────────

def load_personas() -> list[dict]:
    with open(PERSONAS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return [p for p in data if "neilang" in p.get("ground_truth", {})]


def run(checkpoint_path: Path | None = None):
    personas = load_personas()
    print(f"neilang olan persona sayısı: {len(personas)}")

    done: dict[str, dict] = {}
    if checkpoint_path and checkpoint_path.exists():
        with open(checkpoint_path, encoding="utf-8") as f:
            for rec in json.load(f):
                done[rec["persona_id"]] = rec
        print(f"Checkpoint yüklendi: {len(done)} persona atlanıyor.")

    client    = get_client()
    results: list[dict] = list(done.values())
    remaining = [p for p in personas if p["persona_id"] not in done]
    total     = len(personas)
    lock      = threading.Lock()
    completed_count = len(done)

    print(f"İşlenecek: {len(remaining)} persona | workers: {MAX_WORKERS}\n")

    def process(persona: dict) -> dict:
        pid    = persona["persona_id"]
        gt_val = persona["ground_truth"].get("neilang")
        cleaned_prompt = clean_prompt(persona.get("system_prompt", ""))
        pred, raw, tok_p, tok_c = call_api(client, cleaned_prompt)
        return {
            "persona_id":      pid,
            "qform":           persona.get("qform"),
            "age":             persona.get("age"),
            "gender":          persona.get("gender"),
            "region":          persona.get("region"),
            "neilang":         gt_val,
            "gt_neilang":      gt_val,
            "pred_neilang":    pred,
            "match":           int(pred == gt_val) if pred is not None and gt_val is not None else None,
            "raw_response":    raw,
            "token_prompt":    tok_p,
            "token_completion": tok_c,
        }

    pbar = tqdm(total=len(remaining), desc="Tahmin", unit="persona")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process, p): p for p in remaining}
        for future in as_completed(futures):
            rec = future.result()
            match_sym = "✓" if rec["match"] == 1 else ("✗" if rec["match"] == 0 else "?")
            pbar.set_postfix({"id": rec["persona_id"], "gt": rec["gt_neilang"],
                              "pred": rec["pred_neilang"], "match": match_sym})
            pbar.update(1)

            with lock:
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


# ─── Kayıt ───────────────────────────────────────────────────────────────────

def save_results(results: list[dict], ts: str):
    csv_path  = EXPORTS_DIR / f"neilang_predictions_{ts}.csv"
    json_path = EXPORTS_DIR / f"neilang_predictions_{ts}.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

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


# ─── Özet ────────────────────────────────────────────────────────────────────

def print_summary(results: list[dict]):
    from collections import Counter

    valid      = [r for r in results if r["match"] is not None]
    parse_fail = sum(1 for r in results if r["pred_neilang"] is None)
    if not valid:
        print("Geçerli sonuç yok.")
        return

    gt_vals   = [r["gt_neilang"]   for r in valid]
    pred_vals = [r["pred_neilang"] for r in valid]
    accuracy  = sum(r["match"] for r in valid) / len(valid)
    labels    = sorted(set(gt_vals) | set(pred_vals))
    label_names = {1: "Hiç (1)", 2: "Biraz (2)", 3: "Çok (3)"}

    cm: dict[tuple, int] = {}
    for g, p in zip(gt_vals, pred_vals):
        cm[(g, p)] = cm.get((g, p), 0) + 1

    metrics = {}
    for lbl in labels:
        tp = cm.get((lbl, lbl), 0)
        fp = sum(cm.get((g, lbl), 0) for g in labels if g != lbl)
        fn = sum(cm.get((lbl, p), 0) for p in labels if p != lbl)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        metrics[lbl] = {"precision": prec, "recall": rec, "f1": f1}

    col_w = 10
    print("\n" + "=" * 60)
    print("ÖZET")
    print("=" * 60)
    print(f"Toplam      : {len(results)}")
    print(f"Geçerli     : {len(valid)}")
    print(f"Parse hata  : {parse_fail}")
    print(f"Accuracy    : {accuracy:.4f}  ({sum(r['match'] for r in valid)}/{len(valid)})")
    print(f"\nGT dağılımı  : {dict(Counter(gt_vals))}")
    print(f"Pred dağılımı: {dict(Counter(pred_vals))}")

    print("\n── Confusion Matrix ─────────────────────────────────────")
    header = " " * 14 + "".join(f"Pred {label_names.get(l,l):>{col_w-4}}" for l in labels)
    print(header)
    print(" " * 14 + "-" * (col_w * len(labels)))
    for g in labels:
        row = f"GT {label_names.get(g,g):<11}"
        for p in labels:
            row += f"{cm.get((g,p),0):>{col_w}}"
        print(row)

    print("\n── Per-class Metrics ────────────────────────────────────")
    print(f"{'Sınıf':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 54)
    for lbl in labels:
        support = sum(1 for g in gt_vals if g == lbl)
        m = metrics[lbl]
        print(f"{label_names.get(lbl,lbl):<12} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {support:>10}")

    n = len(valid)
    macro_p  = sum(metrics[l]["precision"] for l in labels) / len(labels)
    macro_r  = sum(metrics[l]["recall"]    for l in labels) / len(labels)
    macro_f1 = sum(metrics[l]["f1"]        for l in labels) / len(labels)
    w_p  = sum(metrics[l]["precision"] * sum(1 for g in gt_vals if g == l) for l in labels) / n
    w_r  = sum(metrics[l]["recall"]    * sum(1 for g in gt_vals if g == l) for l in labels) / n
    w_f1 = sum(metrics[l]["f1"]        * sum(1 for g in gt_vals if g == l) for l in labels) / n
    print("-" * 54)
    print(f"{'Macro avg':<12} {macro_p:>10.4f} {macro_r:>10.4f} {macro_f1:>10.4f} {n:>10}")
    print(f"{'Weighted avg':<12} {w_p:>10.4f} {w_r:>10.4f} {w_f1:>10.4f} {n:>10}")
    print("=" * 60)


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ts              = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = EXPORTS_DIR / "neilang_checkpoint.json"

    print(f"Model: {MODEL} | Temperature: {TEMPERATURE}")
    print(f"Checkpoint: {checkpoint_path}\n")

    # Prompt temizleme önizlemesi
    print("── Prompt temizleme testi ──")
    _leak_kw = re.compile(
        r'farklı\s+dil|farklı\s+dilden|dil\s+konuşan'
        r'|farklı\s+din[îi]?\s+görüş|farklı\s+dinden|farklı\s+inanç',
        re.IGNORECASE,
    )
    with open(PERSONAS_PATH, encoding="utf-8") as f:
        _sample_data = json.load(f)
    _leaky = [p for p in _sample_data if _leak_kw.search(p.get("system_prompt", ""))]
    _still = [p for p in _leaky if _leak_kw.search(clean_prompt(p.get("system_prompt", "")))]
    print(f"  Leak içeren prompt: {len(_leaky)} → temizleme sonrası kalan: {len(_still)}")
    if _leaky:
        _ex = _leaky[0]
        _orig = _ex["system_prompt"]
        for _s in re.split(r'(?<=[.;])\s+', _orig):
            if _leak_kw.search(_s):
                print(f"  KALDIRILDI: {repr(_s.strip())}")
    print()

    results = run(checkpoint_path=checkpoint_path)

    csv_path, json_path = save_results(results, ts)
    print(f"\nCSV  → {csv_path}")
    print(f"JSON → {json_path}")

    print_summary(results)

    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print("Checkpoint temizlendi.")
