# TGSS 2024 Agent Survey — Değişiklik Günlüğü

---

## 2026-04-10

### 01_persona_engine.py — v1 → v2

**Yedek:** `01_persona_engine_v1.py`

- **TEHDİT ALGILARIN bölümü kaldırıldı** — `thterror`, `thimmig`, `thhumanr`, `thpolar`, `thlgbtq` artık persona metnine yazılmıyor; ground_truth'ta temiz hedef olarak kalıyor (yıldızsız)
- **`likert_three_way` helper eklendi** — orta skor (3) için de cümle üretiyor
- `famroles` ve `menhead` bu helper'a taşındı (önceden kararsız → sessiz, şimdi kararsız → cümle yazılıyor)
- **Sosyal ağ mantığı güncellendi** — `meetfri` + `sharenum` birlikte değerlendiriliyor; ≤5+≤2 → "sosyal hayatın sınırlı", ≥8+≥4 → "geniş ve aktif çevre"; diğerleri için mekanik ifadeler yumuşatıldı
- **`dominant_party` güncellendi** — 2. parti 1. partiden ≤2 puan aşağıdaysa ve ≥5 ise her ikisi de belirtiliyor
- **Yeni bölüm eklendi: EKONOMİK DURUM, İYİ OLUŞ VE BAĞLAM** — `lifesat`, `happy`, `health`, `econpast`, `inchhesy`, `oldincyou`, `nbsafe`, `consctgr`, `judgefre`
- **ground_truth güncellendi:**
  - `lifesat*`, `happy*`, `health*`, `econpast*`, `inchhesy*` → leaky (persona'ya eklendi)
  - `oldincyou*`, `nbsafe*`, `consctgr*`, `judgefre*` → leaky (persona'ya eklendi)
  - `thterror`, `thimmig`, `thhumanr`, `thpolar`, `thlgbtq` → temiz hedef (yıldız kaldırıldı)
- **Güvenlik:** Çıktı dosyası varsa `_backup_YYYYMMDD_HHMMSS` ekiyle yedek alınıyor
- **Varsayılan giriş yolu** `TGSS/TGSS2024.sav` olarak güncellendi

---

### 02_rewrite_engine.py — v2 → v3

**Kaynak:** `agentrewrite.py` (önceki versiyon)  
**Yedek:** `02_rewrite_engine_v1.py`

- `TEMPERATURE` 0.2 → 0.7
- `MAX_OUTPUT_TOKENS` 500 → 700
- `MODEL` `gpt-5.4` → `gpt-5.4-mini`
- `prompt_version` `api_rewritten_v2` → `api_rewritten_v3`
- **Rewrite prompt'a iki yeni talimat bloğu eklendi:**
  - ÇIKARILACAK İÇERİK: terör/göç/LGBTİ+/insan hakları tehdit algıları ve göçmen komşu rahatsızlıkları
  - EKLENECEK İÇERİK: yaşam memnuniyeti, sağlık, ekonomik durum, yaşlılık güvencesi, mahalle güvenliği, komplo eğilimi, ifade özgürlüğü hissi
- **Güvenlik:** Çıktı dosyası varsa `_backup_YYYYMMDD_HHMMSS` ekiyle yedek alınıyor

---

## 2026-04-11

### Üretim Çalıştırmaları

| Dosya | Versiyon | Toplam | Başarılı | Hatalı |
|---|---|---|---|---|
| `personas.json` | — | 2615 | 2615 | 0 |
| `personas_rewritten.json` | api_rewritten_v3 | 2615 | 2589 | 26 |

- 26 hatalı persona JSON parse hatası (`Extra data`, `Invalid control character`) — yeniden gönderilecek
