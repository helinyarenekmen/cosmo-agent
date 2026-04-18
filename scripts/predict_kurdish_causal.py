"""
Batch script: Dijital İkizlerle Nedensel Çıkarım — Kürtçe Anadil Eğitimi
3 × 4 Tam Faktöriyel Deney

Faktör 1 — Güvenlik Bağlamı (3 düzey):
  K   : Kontrol — bağlam yok
  1a  : Çatışma — aktif sınır ötesi operasyon
  1b  : Barış   — Türkiye-Kürt grupları anlaşması

Faktör 2 — Talep Çerçevelemesi (4 düzey):
  K   : Kontrol — çerçeve yok
  2a  : Dini    — Kur'an ayeti
  2b  : İnsan hakları — BM Evrensel Beyannamesi
  2c  : Ulusal özgürlük — kendi kaderini tayin

Her persona 12 koşulun tamamına tabi tutulur (within-subject).
Her koşul için tek API çağrısıyla tüm bağımlı değişkenler (3a, 3b, 3c, 3d) ve
ek ölçümler (tehdit algısı, çerçeve meşruiyeti) alınır.

Çıktı: exports/kurdish_causal_<timestamp>.csv + .json
Kaldığı yerden devam (checkpoint) destekli, paralel istek.
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
MAX_TOKENS  = 600
RETRY_LIMIT = 3
RETRY_DELAY = 5
MAX_WORKERS = 10

BASE_DIR      = Path(__file__).parent.parent
PERSONAS_PATH = BASE_DIR / "personas_rewritten_v2.json"
EXPORTS_DIR   = BASE_DIR / "exports"
EXPORTS_DIR.mkdir(exist_ok=True)

OPENAI_API_KEY = ""  # API key'ini buraya yaz

# ─── Türkiye Bağlamı ──────────────────────────────────────────────────────────

TURKEY_CONTEXT = (
    "Türkiye'de etnik ve dilsel çeşitlilik, hassas siyasi bir konu olmayı sürdürmektedir. "
    "Ülkede Kürtçe konuşan nüfus önemli bir yer tutmaktadır; Kürtçe anadilde eğitim talebi "
    "siyasi yelpazenin farklı kesimlerinde farklı biçimlerde algılanmaktadır. "
    "Milliyetçi kesimler bu talebi devletin birliğine yönelik bir tehdit olarak değerlendirirken, "
    "sol, liberal ve bazı dindar muhafazakâr kesimler bunu temel bir hak olarak savunmaktadır. "
    "Güvenlik kaygıları ve etnik kimlik politikaları, kamuoyunun bu konudaki tutumunu "
    "doğrudan şekillendirmektedir."
)

# ─── Deney Koşulları ──────────────────────────────────────────────────────────

# 12 koşul: (security_context, framing) çiftleri
CONDITIONS = [
    ("K",  "K"),
    ("K",  "2a"),
    ("K",  "2b"),
    ("K",  "2c"),
    ("1a", "K"),
    ("1a", "2a"),
    ("1a", "2b"),
    ("1a", "2c"),
    ("1b", "K"),
    ("1b", "2a"),
    ("1b", "2b"),
    ("1b", "2c"),
]

CONDITION_IDS = {cond: f"C{i+1:02d}" for i, cond in enumerate(CONDITIONS)}

# Güvenlik bağlamı metinleri (Blok 1)
SECURITY_TEXTS = {
    "1a": (
        "Türkiye ordusu şu sıralar Suriye'nin kuzeyinde aktif bir sınır ötesi operasyon yürütüyor. "
        "Son haftalarda askeri kayıplar yaşandığı haberleri geliyor."
    ),
    "1b": (
        "Türkiye ile Suriye'deki Kürt grupları arasında yakın zamanda bir anlaşma sağlandı. "
        "Bölgedeki silahlı çatışmanın sona erdiği açıklandı."
    ),
}

# Çerçeveleme metinleri (Blok 2)
FRAMING_TEXTS = {
    "2a": (
        "Bu günlerde Türkiye'de Kürtlerin anadilde eğitim hakkı tartışılıyor. "
        "Bir grup bu hakkı savunmak amacıyla yürüyüş düzenledi. "
        "Yürüyüşte Kur'an'dan \"Sizi farklı dil ve renklerde yarattık\" mealindeki ayet "
        "pankartlarda taşındı."
    ),
    "2b": (
        "Bu günlerde Türkiye'de Kürtlerin anadilde eğitim hakkı tartışılıyor. "
        "Bir grup bu hakkı savunmak amacıyla yürüyüş düzenledi. "
        "Yürüyüşte Birleşmiş Milletler İnsan Hakları Evrensel Beyannamesi'ne atıfla "
        "dilin temel bir hak olduğu vurgulandı."
    ),
    "2c": (
        "Bu günlerde Türkiye'de Kürtlerin anadinde eğitim hakkı tartışılıyor. "
        "Bir grup bu hakkı savunmak amacıyla yürüyüş düzenledi. "
        "Yürüyüşte \"Dil yasakları sömürge politikasıdır, halkların kendi kaderini tayin hakkı "
        "pazarlık konusu olamaz\" yazılı pankartlar taşındı."
    ),
}

# Kontrol koşulu çerçeve metni (K)
FRAMING_CONTROL = (
    "Bu günlerde Türkiye'de Kürtlerin anadilde eğitim hakkı tartışılıyor. "
    "Bir grup bu hakkı savunmak amacıyla yürüyüş düzenledi."
)

# ─── JSON çıktı şeması ────────────────────────────────────────────────────────

def build_json_schema(security: str, framing: str) -> str:
    """Koşula göre beklenen JSON şemasını açıklar."""
    lines = ['Yanıtını SADECE aşağıdaki JSON formatında ver, başka hiçbir şey yazma:']
    lines.append('{')

    if security != "K":
        lines.append(
            '  "manip_check": {"p1": <0-1>, "p2": <0-1>, "p3": <0-1>},'
            '  // Manipülasyon kontrolü: p1=aktif çatışma, p2=anlaşma sağlandı, p3=hatırlamıyorum'
        )

    lines.append(
        '  "dv_3a": {"p1": <0-1>, "p2": <0-1>, "p3": <0-1>, "p4": <0-1>, "p5": <0-1>},'
        '  // 3a: Meşru hak mücadelesi (1=kesinlikle katılmıyorum … 5=kesinlikle katılıyorum)'
    )
    lines.append(
        '  "dv_3b": {"p1": <0-1>, "p2": <0-1>, "p3": <0-1>, "p4": <0-1>, "p5": <0-1>},'
        '  // 3b: Toplumsal huzuru bozucu (1=kesinlikle katılmıyorum … 5=kesinlikle katılıyorum)'
    )
    lines.append(
        '  "dv_3c": {"p1": <0-1>, "p2": <0-1>, "p3": <0-1>, "p4": <0-1>, "p5": <0-1>},'
        '  // 3c: Kürtçe anadil eğitimine yasal izin (1=kesinlikle katılmıyorum … 5=kesinlikle katılıyorum)'
    )
    lines.append(
        '  "dv_3d": {"p1": <0-1>, "p2": <0-1>, "p3": <0-1>, "p4": <0-1>, "p5": <0-1>, "p6": <0-1>},'
        '  // 3d: Davranış niyeti (p1=bizzat katılırdım, p2=sosyal medyada destekler, p3=sessizce destekler,'
        '  //   p4=sessizce karşı çıkar, p5=sosyal medyada karşı çıkar, p6=karşıt etkinliğe katılır)'
    )
    lines.append(
        '  "threat": {"p1": <0-1>, "p2": <0-1>, "p3": <0-1>, "p4": <0-1>}'
        '  // Tehdit algısı: p1=hiç tehdit altında değil … p4=çok ciddi tehdit altında'
    )

    if framing != "K":
        lines[-1] = lines[-1].rstrip() + ','
        lines.append(
            '  "frame_legit": {"p1": <0-1>, "p2": <0-1>, "p3": <0-1>, "p4": <0-1>}'
            '  // Çerçeve meşruiyeti: p1=hiç meşru değil … p4=çok meşru'
        )

    lines.append('}')
    lines.append('Tüm olasılıklar 0 ile 1 arasında olmalı; her dağılımın toplamı 1.0 olmalıdır.')
    return '\n'.join(lines)


# ─── Senaryo oluşturma ────────────────────────────────────────────────────────

def build_scenario(security: str, framing: str) -> str:
    """Bir koşul için tam senaryo metnini (Blok 1 + Blok 2 + sorular) oluşturur."""
    parts = []

    # --- Blok 1: Güvenlik bağlamı ---
    if security != "K":
        parts.append("=== Haber ===")
        parts.append(SECURITY_TEXTS[security])
        parts.append(
            "\nManipülasyon kontrolü: Az önce okuduğunuz habere göre Türkiye ile Suriye'deki "
            "Kürt grupları arasındaki ilişkiyi nasıl tanımlarsınız?\n"
            "  p1 = Aktif silahlı çatışma var\n"
            "  p2 = Anlaşma sağlandı, çatışma sona erdi\n"
            "  p3 = Hatırlamıyorum"
        )
        parts.append("")

    # --- Blok 2: Çerçeveleme ---
    parts.append("=== Haber ===")
    if framing == "K":
        parts.append(FRAMING_CONTROL)
    else:
        parts.append(FRAMING_TEXTS[framing])
    parts.append("")

    # --- Blok 3: Bağımlı değişkenler ---
    parts.append(
        "Tüm bunları göz önünde bulundurarak aşağıdaki görüşlere ne ölçüde katılıyorsunuz?\n"
        "Bu kişinin profiline göre her yanıt seçeneğinin ne kadar olası olduğunu tahmin et.\n"
    )
    parts.append(
        "3a. Bu grup meşru bir hak mücadelesi yürütüyor.\n"
        "  p1=Kesinlikle katılmıyorum | p2=Katılmıyorum | p3=Kararsızım | "
        "p4=Katılıyorum | p5=Kesinlikle katılıyorum\n"
    )
    parts.append(
        "3b. Bu grubun talepleri toplumsal huzuru bozucu nitelik taşıyor.\n"
        "  p1=Kesinlikle katılmıyorum | p2=Katılmıyorum | p3=Kararsızım | "
        "p4=Katılıyorum | p5=Kesinlikle katılıyorum\n"
    )
    parts.append(
        "3c. Devlet okullarında Kürtçe anadil eğitimine yasal olarak izin verilmelidir.\n"
        "  p1=Kesinlikle katılmıyorum | p2=Katılmıyorum | p3=Kararsızım | "
        "p4=Katılıyorum | p5=Kesinlikle katılıyorum\n"
    )
    parts.append(
        "3d. Böyle bir etkinlik düzenlendiğinde ne yapardınız?\n"
        "  p1=Yürüyüşe bizzat katılırdım\n"
        "  p2=Sosyal medyada paylaşır, desteklerdim\n"
        "  p3=Destekler ama kamuoyu önünde belli etmezdim\n"
        "  p4=Karşı çıkar ama kamuoyu önünde belli etmezdim\n"
        "  p5=Sosyal medyada karşıt görüşlü paylaşım yapardım\n"
        "  p6=Karşıt görüşlü bir etkinliğe bizzat katılırdım\n"
    )

    # --- Blok 4: Ek ölçümler ---
    parts.append(
        "Tehdit algısı: Şu anda Türkiye'nin güvenliği sizce ne kadar tehdit altında?\n"
        "  p1=Hiç tehdit altında değil | p2=Biraz tehdit altında | "
        "p3=Oldukça tehdit altında | p4=Çok ciddi tehdit altında\n"
    )
    if framing != "K":
        parts.append(
            "Çerçeve meşruiyeti: Az önce bahsi geçen protestocu grubun genel yaklaşımını "
            "ne kadar meşru buluyorsunuz?\n"
            "  p1=Hiç meşru değil | p2=Biraz meşru | p3=Oldukça meşru | p4=Çok meşru\n"
        )

    parts.append(build_json_schema(security, framing))
    return "\n".join(parts)


# ─── Prompt temizleme ─────────────────────────────────────────────────────────

_LEAKY_PATTERNS = re.compile(
    r'[^.;!?\n]*(?:'
    r'Kürtçe\s+eğitim'
    r'|anadil\s+hakkı'
    r'|Kürt\s+kimliğ'
    r'|Kürt\s+siyasi'
    r'|Kürt\s+özerk'
    r')[^.;!?\n]*[.;!?]?',
    re.IGNORECASE,
)
_DOUBLE_SPACE = re.compile(r'  +')


def clean_prompt(prompt: str) -> str:
    cleaned = _LEAKY_PATTERNS.sub('', prompt)
    cleaned = _DOUBLE_SPACE.sub(' ', cleaned)
    return '\n'.join(l.strip() for l in cleaned.splitlines() if l.strip())


# ─── API ──────────────────────────────────────────────────────────────────────

def get_client() -> OpenAI:
    key = OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise ValueError("API anahtarı bulunamadı.")
    return OpenAI(api_key=key)


def _sample_from_dist(vals: list[float]) -> int:
    r = random.random()
    cumsum = 0.0
    for i, v in enumerate(vals):
        cumsum += v
        if r < cumsum:
            return i + 1
    return len(vals)


def _normalize(vals: list[float]) -> list[float] | None:
    total = sum(vals)
    if total <= 0:
        return None
    return [v / total for v in vals]


def _parse_dist(obj: dict, keys: list[str]) -> tuple[dict | None, int | None]:
    """Verilen anahtarlar için obj'dan olasılık dağılımı çıkarır."""
    try:
        vals = [float(obj.get(k, 0)) for k in keys]
    except (TypeError, ValueError):
        return None, None
    normed = _normalize(vals)
    if normed is None:
        return None, None
    probs = {k: round(v, 6) for k, v in zip(keys, normed)}
    sampled = _sample_from_dist(normed)
    return probs, sampled


def parse_response(text: str, security: str, framing: str) -> dict:
    """Model yanıtını ayrıştırır; tüm DV dağılımlarını ve örneklenen değerleri döndürür."""
    result: dict = {}

    obj = None
    for src in [text.strip(),
                (re.search(r'\{.*\}', text, re.DOTALL) or type('', (), {'group': lambda s, x: ''})()).group(0)]:
        try:
            obj = json.loads(src)
            if isinstance(obj, dict):
                break
        except Exception:
            pass

    if obj is None:
        return result  # boş döner, parse hata olarak kaydedilir

    # Manipülasyon kontrolü (sadece güvenlik != K)
    if security != "K" and "manip_check" in obj:
        probs, sampled = _parse_dist(obj["manip_check"], ["p1", "p2", "p3"])
        if probs:
            result["manip_check_p1"] = probs["p1"]
            result["manip_check_p2"] = probs["p2"]
            result["manip_check_p3"] = probs["p3"]
            result["manip_check_sampled"] = sampled

    # 3a: Meşru hak mücadelesi (1-5)
    if "dv_3a" in obj:
        probs, sampled = _parse_dist(obj["dv_3a"], ["p1", "p2", "p3", "p4", "p5"])
        if probs:
            for k, v in probs.items():
                result[f"dv_3a_{k}"] = v
            result["dv_3a_sampled"] = sampled

    # 3b: Huzur bozucu (1-5)
    if "dv_3b" in obj:
        probs, sampled = _parse_dist(obj["dv_3b"], ["p1", "p2", "p3", "p4", "p5"])
        if probs:
            for k, v in probs.items():
                result[f"dv_3b_{k}"] = v
            result["dv_3b_sampled"] = sampled

    # 3c: Yasal izin (1-5)
    if "dv_3c" in obj:
        probs, sampled = _parse_dist(obj["dv_3c"], ["p1", "p2", "p3", "p4", "p5"])
        if probs:
            for k, v in probs.items():
                result[f"dv_3c_{k}"] = v
            result["dv_3c_sampled"] = sampled

    # 3d: Davranış niyeti (1-6)
    if "dv_3d" in obj:
        probs, sampled = _parse_dist(obj["dv_3d"], ["p1", "p2", "p3", "p4", "p5", "p6"])
        if probs:
            for k, v in probs.items():
                result[f"dv_3d_{k}"] = v
            result["dv_3d_sampled"] = sampled

    # Tehdit algısı (1-4)
    if "threat" in obj:
        probs, sampled = _parse_dist(obj["threat"], ["p1", "p2", "p3", "p4"])
        if probs:
            for k, v in probs.items():
                result[f"threat_{k}"] = v
            result["threat_sampled"] = sampled

    # Çerçeve meşruiyeti (1-4, sadece framing != K)
    if framing != "K" and "frame_legit" in obj:
        probs, sampled = _parse_dist(obj["frame_legit"], ["p1", "p2", "p3", "p4"])
        if probs:
            for k, v in probs.items():
                result[f"frame_legit_{k}"] = v
            result["frame_legit_sampled"] = sampled

    return result


def call_api(client: OpenAI, system_prompt: str, user_message: str) -> tuple[dict, str, int, int]:
    full_system = f"{TURKEY_CONTEXT.strip()}\n\n{system_prompt}"
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": full_system},
                    {"role": "user",   "content": user_message},
                ],
                temperature=TEMPERATURE,
                max_completion_tokens=MAX_TOKENS,
            )
            raw   = resp.choices[0].message.content or ""
            tok_p = getattr(resp.usage, "prompt_tokens",     0)
            tok_c = getattr(resp.usage, "completion_tokens", 0)
            return raw, tok_p, tok_c
        except Exception as exc:
            tqdm.write(f"  [Hata] deneme {attempt}/{RETRY_LIMIT}: {exc}")
            if attempt < RETRY_LIMIT:
                time.sleep(RETRY_DELAY * attempt)
    return "API_ERROR", 0, 0


# ─── Veri yükleme ─────────────────────────────────────────────────────────────

def load_personas() -> list[dict]:
    with open(PERSONAS_PATH, encoding="utf-8") as f:
        return json.load(f)


# ─── Kayıt ───────────────────────────────────────────────────────────────────

ALL_COLUMNS = [
    "persona_id", "qform", "age", "gender", "region",
    "condition_id", "security_context", "framing",
    # Manipülasyon kontrolü (sadece security != K)
    "manip_check_p1", "manip_check_p2", "manip_check_p3", "manip_check_sampled",
    # DV 3a
    "dv_3a_p1", "dv_3a_p2", "dv_3a_p3", "dv_3a_p4", "dv_3a_p5", "dv_3a_sampled",
    # DV 3b
    "dv_3b_p1", "dv_3b_p2", "dv_3b_p3", "dv_3b_p4", "dv_3b_p5", "dv_3b_sampled",
    # DV 3c
    "dv_3c_p1", "dv_3c_p2", "dv_3c_p3", "dv_3c_p4", "dv_3c_p5", "dv_3c_sampled",
    # DV 3d (6 kategori)
    "dv_3d_p1", "dv_3d_p2", "dv_3d_p3", "dv_3d_p4", "dv_3d_p5", "dv_3d_p6", "dv_3d_sampled",
    # Tehdit algısı
    "threat_p1", "threat_p2", "threat_p3", "threat_p4", "threat_sampled",
    # Çerçeve meşruiyeti (sadece framing != K)
    "frame_legit_p1", "frame_legit_p2", "frame_legit_p3", "frame_legit_p4", "frame_legit_sampled",
    # Meta
    "raw_response", "token_prompt", "token_completion",
]


def save_results(results: list[dict], ts: str) -> tuple[Path, Path]:
    csv_path  = EXPORTS_DIR / f"kurdish_causal_{ts}.csv"
    json_path = EXPORTS_DIR / f"kurdish_causal_{ts}.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    if results:
        with open(csv_path, "w", encoding="utf-8-sig") as f:
            f.write(",".join(ALL_COLUMNS) + "\n")
            for rec in results:
                row = []
                for col in ALL_COLUMNS:
                    val = rec.get(col, "")
                    s   = "" if val is None else str(val)
                    if "," in s or '"' in s or "\n" in s:
                        s = '"' + s.replace('"', '""') + '"'
                    row.append(s)
                f.write(",".join(row) + "\n")

    return csv_path, json_path


# ─── Ana akış ────────────────────────────────────────────────────────────────

def run(checkpoint_path: Path | None = None) -> list[dict]:
    personas = load_personas()
    print(f"Toplam persona: {len(personas)} | Toplam koşul: {len(CONDITIONS)}")
    print(f"Toplam API çağrısı: {len(personas) * len(CONDITIONS):,}")

    # Checkpoint: anahtar = "persona_id_condition_id"
    done: dict[str, dict] = {}
    if checkpoint_path and checkpoint_path.exists():
        with open(checkpoint_path, encoding="utf-8") as f:
            for rec in json.load(f):
                key = f"{rec['persona_id']}_{rec['condition_id']}"
                done[key] = rec
        print(f"Checkpoint yüklendi: {len(done)} kayıt atlanıyor.")

    # Tüm (persona, condition) görev çiftlerini oluştur
    tasks = []
    for persona in personas:
        for sec, frm in CONDITIONS:
            cid = CONDITION_IDS[(sec, frm)]
            key = f"{persona['persona_id']}_{cid}"
            if key not in done:
                tasks.append((persona, sec, frm, cid))

    client   = get_client()
    results  = list(done.values())
    total    = len(personas) * len(CONDITIONS)
    lock     = threading.Lock()
    completed_count = len(done)

    print(f"İşlenecek: {len(tasks)} görev | Workers: {MAX_WORKERS}\n")

    def process(task: tuple) -> dict:
        persona, sec, frm, cid = task
        pid          = persona["persona_id"]
        cleaned_sp   = clean_prompt(persona.get("system_prompt", ""))
        user_message = build_scenario(sec, frm)

        raw, tok_p, tok_c = call_api(client, cleaned_sp, user_message)
        parsed = parse_response(raw, sec, frm) if raw != "API_ERROR" else {}

        rec: dict = {
            "persona_id":        pid,
            "qform":             persona.get("qform"),
            "age":               persona.get("age"),
            "gender":            persona.get("gender"),
            "region":            persona.get("region"),
            "condition_id":      cid,
            "security_context":  sec,
            "framing":           frm,
            "raw_response":      raw,
            "token_prompt":      tok_p,
            "token_completion":  tok_c,
        }
        rec.update(parsed)
        return rec

    pbar = tqdm(total=len(tasks), desc="Tahmin", unit="görev")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process, t): t for t in tasks}
        for future in as_completed(futures):
            rec = future.result()
            # Temel kontrol: 3c (Kürtçe eğitim desteği) sampled değeri
            dv3c = rec.get("dv_3c_sampled", "?")
            pbar.set_postfix({
                "pid":  rec["persona_id"],
                "cond": rec["condition_id"],
                "3c":   dv3c,
            })
            pbar.update(1)

            with lock:
                results.append(rec)
                completed_count += 1
                if checkpoint_path and completed_count % 100 == 0:
                    with open(checkpoint_path, "w", encoding="utf-8") as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
                    tqdm.write(f"  → Checkpoint ({completed_count}/{total})")

    pbar.close()

    if checkpoint_path:
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    return results


# ─── Özet ────────────────────────────────────────────────────────────────────

def print_summary(results: list[dict]) -> None:
    """Koşul bazlı ortalama 3c (Kürtçe eğitim desteği) dağılımı gösterir."""
    from collections import defaultdict

    cond_scores: dict[str, list[float]] = defaultdict(list)
    for rec in results:
        val = rec.get("dv_3c_sampled")
        if val is not None:
            cid = f"{rec['security_context']}-{rec['framing']}"
            cond_scores[cid].append(float(val))

    print("\n" + "=" * 60)
    print("ÖZET — 3c (Kürtçe Anadil Eğitimi Desteği) | Ortalama Sampled")
    print("=" * 60)
    header = f"{'Koşul':<12} {'N':>6} {'Ort.':>8} {'Std.':>8}"
    print(header)
    print("-" * 60)
    for cid in sorted(cond_scores.keys()):
        vals = cond_scores[cid]
        n    = len(vals)
        mean = sum(vals) / n
        std  = (sum((v - mean) ** 2 for v in vals) / n) ** 0.5
        print(f"{cid:<12} {n:>6} {mean:>8.3f} {std:>8.3f}")
    print("=" * 60)

    # Parse hatası sayısı
    parse_fails = sum(
        1 for rec in results if rec.get("dv_3c_sampled") is None and rec.get("raw_response") != "API_ERROR"
    )
    api_errors = sum(1 for rec in results if rec.get("raw_response") == "API_ERROR")
    print(f"\nAPI Hatası : {api_errors}")
    print(f"Parse Hatası: {parse_fails}")
    print(f"Toplam Kayıt: {len(results)}")


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ts              = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = EXPORTS_DIR / "kurdish_causal_checkpoint.json"

    print(f"Model: {MODEL} | Temperature: {TEMPERATURE}")
    print(f"Yöntem: Verbalized Sampling — Çok Değişkenli")
    print(f"Deney: 3×4 Tam Faktöriyel ({len(CONDITIONS)} koşul)")
    print(f"Checkpoint: {checkpoint_path}\n")

    results = run(checkpoint_path=checkpoint_path)

    csv_path, json_path = save_results(results, ts)
    print(f"\nCSV  → {csv_path}")
    print(f"JSON → {json_path}")

    print_summary(results)

    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print("Checkpoint temizlendi.")
