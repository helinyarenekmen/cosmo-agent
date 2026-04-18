"""
Batch script: womenwork tahmini — Verbalized Sampling yöntemi
- Model: gpt-5.4-mini, temperature: 0.8
- Yöntem: Modelden her seçenek için olasılık dağılımı istenir (verbalized sampling).
  Model {"p1":0.1,"p2":0.1,"p3":0.2,"p4":0.4,"p5":0.2} gibi bir dağılım döndürür.
  Tahmin bu dağılımdan örneklenerek (sampled_pred) ve argmax alınarak (argmax_pred) hesaplanır.
- Veri: personas_rewritten_v2.json (womenwork olan tüm personalar, ~2588)
- Prompt temizleme: womenwork'ü sızdıran cinsiyet rolü cümleleri kaldırılır
- Çıktı: exports/womenwork_verbsampling_<timestamp>.csv + .json
- Kaldığı yerden devam (checkpoint) destekli, paralel istek
"""

import json
import os
import random
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
MAX_TOKENS  = 150
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
    "Muhafazakârlık ve İslamcılık kültürel eksenin dindar ucunu oluşturur. "
    "Türkiye'de cinsiyet rolleri, kadının iş hayatındaki yeri ve aile yapısı hem geleneksel hem de "
    "modern değerlerin çatışma alanı olmaya devam etmektedir. Bireylerin bu konudaki tutumları; "
    "dini kimlik, eğitim düzeyi, yaşanılan bölge ve siyasi eğilimle yakından ilişkilidir."
)

QUESTION = (
    "Aşağıdaki soru toplumsal cinsiyet rollerine ilişkin bir tutum sorusudur. "
    "Kendi gerçek değerlerine ve bakış açına göre dürüstçe yanıtla.\n\n"
    "Bu kişinin profiline göre her yanıt seçeneğinin ne kadar olası olduğunu tahmin et. "
    "Olasılıklar 0 ile 1 arasında olmalı ve toplamları 1.0 olmalıdır.\n\n"
    "Yanıtını SADECE şu JSON formatında ver, başka hiçbir şey yazma:\n"
    '{"p1": <0-1>, "p2": <0-1>, "p3": <0-1>, "p4": <0-1>, "p5": <0-1>}\n\n'
    "Soru:\n"
    "Bir erkeğin görevi para kazanmaktır; bir kadının görevi ise eve ve aileye bakmaktır.\n"
    "  1 = Hiç katılmıyorum\n"
    "  2 = Katılmıyorum\n"
    "  3 = Ne katılıyorum ne katılmıyorum\n"
    "  4 = Katılıyorum\n"
    "  5 = Tamamen katılıyorum"
)

# ─── Prompt temizleme ─────────────────────────────────────────────────────────

_SENT_GENDER_ROLE = re.compile(
    r'[^.;!?\n]*(?:geleneksel\s+cinsiyet\s+rol'
    r'|evin\s+reis'
    r'|erkek\s+rol[üu]'
    r'|kad[iı]n[- ]erkek\s+rol'
    r'|erkeğ\w*\s+söz\s+sahibi'
    r'|aile\s+reisi\s+erkek)[^.;!?\n]*[.;!?]?',
    re.IGNORECASE,
)
_SENT_WOMEN_WORK = re.compile(
    r'[^.;!?\n]*(?:kad[iı]nlar[iı]n?\s+çalış'
    r'|çalışan\s+kad[iı]n'
    r'|kad[iı]n.*ev\s+d[iı]ş[iı]nda\s+çalış'
    r'|işsizlik.*erkek.*öncelik'
    r'|erkek.*iş.*öncelik)[^.;!?\n]*[.;!?]?',
    re.IGNORECASE,
)
_DOUBLE_SPACE = re.compile(r'  +')


def clean_prompt(prompt: str) -> str:
    cleaned = _SENT_GENDER_ROLE.sub('', prompt)
    cleaned = _SENT_WOMEN_WORK.sub('', cleaned)
    cleaned = _DOUBLE_SPACE.sub(' ', cleaned)
    lines = [l.strip() for l in cleaned.splitlines()]
    return '\n'.join(l for l in lines if l)


# ─── API & parsing ────────────────────────────────────────────────────────────

def get_client() -> OpenAI:
    key = OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise ValueError("API anahtarı bulunamadı.")
    return OpenAI(api_key=key)


def parse_response(text: str) -> tuple[dict[str, float] | None, int | None, int | None]:
    if not text:
        return None, None, None

    obj = None
    for src in [text.strip(),
                (re.search(r'\{[^{}]*\}', text, re.DOTALL) or type('', (), {'group': lambda s, x: ''})()).group(0)]:
        try:
            obj = json.loads(src)
            break
        except Exception:
            pass

    if obj is None:
        # Regex fallback
        matches = {f"p{i}": re.search(rf'"p{i}"\s*:\s*([0-9]*\.?[0-9]+)', text) for i in range(1, 6)}
        if all(matches.values()):
            obj = {k: float(m.group(1)) for k, m in matches.items()}
        else:
            return None, None, None

    try:
        vals = [float(obj.get(f"p{i}", 0)) for i in range(1, 6)]
    except (TypeError, ValueError):
        return None, None, None

    total = sum(vals)
    if total <= 0:
        return None, None, None

    vals = [v / total for v in vals]
    probs = {f"p{i+1}": round(v, 6) for i, v in enumerate(vals)}

    argmax_pred = vals.index(max(vals)) + 1

    r = random.random()
    cumsum = 0.0
    sampled_pred = 5
    for i, v in enumerate(vals):
        cumsum += v
        if r < cumsum:
            sampled_pred = i + 1
            break

    return probs, sampled_pred, argmax_pred


def call_api(client: OpenAI, system_prompt: str):
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
            tok_p = getattr(resp.usage, "prompt_tokens",     0)
            tok_c = getattr(resp.usage, "completion_tokens", 0)
            probs, sampled_pred, argmax_pred = parse_response(raw)
            return probs, sampled_pred, argmax_pred, raw, tok_p, tok_c
        except Exception as exc:
            tqdm.write(f"  [Hata] deneme {attempt}/{RETRY_LIMIT}: {exc}")
            if attempt < RETRY_LIMIT:
                time.sleep(RETRY_DELAY * attempt)
    return None, None, None, "API_ERROR", 0, 0


# ─── Ana akış ────────────────────────────────────────────────────────────────

def load_personas() -> list[dict]:
    with open(PERSONAS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return [p for p in data if "womenwork" in p.get("ground_truth", {})]


def run(checkpoint_path: Path | None = None):
    personas = load_personas()
    print(f"womenwork olan persona sayısı: {len(personas)}")

    done: dict[str, dict] = {}
    if checkpoint_path and checkpoint_path.exists():
        with open(checkpoint_path, encoding="utf-8") as f:
            for rec in json.load(f):
                done[rec["persona_id"]] = rec
        print(f"Checkpoint yüklendi: {len(done)} persona atlanıyor.")

    client   = get_client()
    results: list[dict] = list(done.values())
    remaining = [p for p in personas if p["persona_id"] not in done]
    total     = len(personas)
    lock      = threading.Lock()
    completed_count = len(done)

    print(f"İşlenecek: {len(remaining)} persona | workers: {MAX_WORKERS}\n")

    def process(persona: dict) -> dict:
        pid    = persona["persona_id"]
        gt_val = persona["ground_truth"].get("womenwork")
        cleaned_prompt = clean_prompt(persona.get("system_prompt", ""))
        probs, sampled_pred, argmax_pred, raw, tok_p, tok_c = call_api(client, cleaned_prompt)

        match_sampled = int(sampled_pred == gt_val) if sampled_pred is not None and gt_val is not None else None
        match_argmax  = int(argmax_pred  == gt_val) if argmax_pred  is not None and gt_val is not None else None

        return {
            "persona_id":       pid,
            "qform":            persona.get("qform"),
            "age":              persona.get("age"),
            "gender":           persona.get("gender"),
            "region":           persona.get("region"),
            "gt_womenwork":     gt_val,
            "p1":               probs["p1"] if probs else None,
            "p2":               probs["p2"] if probs else None,
            "p3":               probs["p3"] if probs else None,
            "p4":               probs["p4"] if probs else None,
            "p5":               probs["p5"] if probs else None,
            "sampled_pred":     sampled_pred,
            "argmax_pred":      argmax_pred,
            "match_sampled":    match_sampled,
            "match_argmax":     match_argmax,
            "raw_response":     raw,
            "token_prompt":     tok_p,
            "token_completion": tok_c,
        }

    pbar = tqdm(total=len(remaining), desc="Tahmin", unit="persona")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process, p): p for p in remaining}
        for future in as_completed(futures):
            rec = future.result()
            sym_s = "✓" if rec["match_sampled"] == 1 else ("✗" if rec["match_sampled"] == 0 else "?")
            sym_a = "✓" if rec["match_argmax"]  == 1 else ("✗" if rec["match_argmax"]  == 0 else "?")
            pbar.set_postfix({
                "id":  rec["persona_id"],
                "gt":  rec["gt_womenwork"],
                "smp": f"{rec['sampled_pred']}{sym_s}",
                "arg": f"{rec['argmax_pred']}{sym_a}",
            })
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
    csv_path  = EXPORTS_DIR / f"womenwork_verbsampling_{ts}.csv"
    json_path = EXPORTS_DIR / f"womenwork_verbsampling_{ts}.json"

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

    label_names = {
        1: "Hiç Katılm.(1)", 2: "Katılmıyor(2)", 3: "Nötr      (3)",
        4: "Katılıyor (4)",  5: "Tmnkl Katıl(5)",
    }

    for pred_key, match_key, title in [
        ("sampled_pred", "match_sampled", "SAMPLED"),
        ("argmax_pred",  "match_argmax",  "ARGMAX"),
    ]:
        valid      = [r for r in results if r[match_key] is not None]
        parse_fail = sum(1 for r in results if r[pred_key] is None)
        if not valid:
            print(f"[{title}] Geçerli sonuç yok.")
            continue

        gt_vals   = [r["gt_womenwork"] for r in valid]
        pred_vals = [r[pred_key]       for r in valid]
        accuracy  = sum(r[match_key] for r in valid) / len(valid)
        labels    = sorted(set(gt_vals) | set(pred_vals))

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

        col_w = 16
        print("\n" + "=" * 72)
        print(f"ÖZET — {title}")
        print("=" * 72)
        print(f"Toplam     : {len(results)}")
        print(f"Geçerli    : {len(valid)}")
        print(f"Parse hata : {parse_fail}")
        print(f"Accuracy   : {accuracy:.4f}  ({sum(r[match_key] for r in valid)}/{len(valid)})")
        print(f"\nGT dağılımı  : {dict(Counter(gt_vals))}")
        print(f"Pred dağılımı: {dict(Counter(pred_vals))}")

        print("\n── Confusion Matrix ─────────────────────────────────────────────")
        header = " " * 18 + "".join(f"{'Pred '+str(l):>{col_w}}" for l in labels)
        print(header)
        print(" " * 18 + "-" * (col_w * len(labels)))
        for g in labels:
            row = f"GT {label_names.get(g,str(g)):<15}"
            for p in labels:
                row += f"{cm.get((g,p),0):>{col_w}}"
            print(row)

        print("\n── Per-class Metrics ────────────────────────────────────────────")
        print(f"{'Sınıf':<18} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        print("-" * 60)
        n = len(valid)
        for lbl in labels:
            support = sum(1 for g in gt_vals if g == lbl)
            m = metrics[lbl]
            print(f"{label_names.get(lbl,str(lbl)):<18} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {support:>10}")
        macro_p  = sum(metrics[l]["precision"] for l in labels) / len(labels)
        macro_r  = sum(metrics[l]["recall"]    for l in labels) / len(labels)
        macro_f1 = sum(metrics[l]["f1"]        for l in labels) / len(labels)
        w_p  = sum(metrics[l]["precision"] * sum(1 for g in gt_vals if g == l) for l in labels) / n
        w_r  = sum(metrics[l]["recall"]    * sum(1 for g in gt_vals if g == l) for l in labels) / n
        w_f1 = sum(metrics[l]["f1"]        * sum(1 for g in gt_vals if g == l) for l in labels) / n
        print("-" * 60)
        print(f"{'Macro avg':<18} {macro_p:>10.4f} {macro_r:>10.4f} {macro_f1:>10.4f} {n:>10}")
        print(f"{'Weighted avg':<18} {w_p:>10.4f} {w_r:>10.4f} {w_f1:>10.4f} {n:>10}")
        print("=" * 72)

    # Ortalama olasılık dağılımı
    valid_probs = [r for r in results if r["p1"] is not None]
    if valid_probs:
        print(f"\n── Ortalama Olasılık Dağılımı (n={len(valid_probs)}) ──────────────────")
        for i in range(1, 6):
            avg = sum(r[f"p{i}"] for r in valid_probs) / len(valid_probs)
            print(f"  p{i}: {avg:.4f}")


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ts              = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = EXPORTS_DIR / "womenwork_verbsampling_checkpoint.json"

    print(f"Model: {MODEL} | Temperature: {TEMPERATURE}")
    print(f"Yöntem: Verbalized Sampling")
    print(f"Checkpoint: {checkpoint_path}\n")

    results = run(checkpoint_path=checkpoint_path)

    csv_path, json_path = save_results(results, ts)
    print(f"\nCSV  → {csv_path}")
    print(f"JSON → {json_path}")

    print_summary(results)

    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print("Checkpoint temizlendi.")
