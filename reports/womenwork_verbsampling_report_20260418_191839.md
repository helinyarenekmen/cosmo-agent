# womenwork Prediction Report — Verbalized Sampling

**Model:** gpt-5.4-mini | **Temperature:** 0.8 | **Method:** Verbalized Sampling | **Date:** 2026-04-18 19:18
**Source:** `womenwork_verbsampling_20260418_185933.csv`
**Question:** *"A man's job is to earn money; a woman's job is to look after the home and family."*
(1 = Strongly disagree → 5 = Strongly agree)
**Prompt cleaning:** sentences revealing gender-role attitudes removed before inference.

> **Verbalized Sampling:** the model outputs a probability distribution (p1–p5) over all classes.
> The final prediction is drawn by sampling from this distribution.

---

## 1. Overall Performance

| Metric | Value |
|---|---|
| Total personas | 2588 |
| Valid predictions | 2588 |
| Parse failures | 0 |
| **Accuracy** | **0.2716** |
| Macro F1 | 0.2471 |
| Weighted F1 | 0.2718 |

> 5-class random baseline ≈ 0.20

---

## 2. Ground Truth vs Sampled Distribution

![Distribution](../figures/womenwork_verbsampling_plots_20260418_191839/distribution.png)

| Class | Ground Truth | Sampled Prediction |
|---|---|---|
| Strongly disagree (1) | 282 (10.9%) | 232 (9.0%) |
| Disagree (2) | 602 (23.3%) | 402 (15.5%) |
| Neither (3) | 361 (13.9%) | 432 (16.7%) |
| Agree (4) | 892 (34.5%) | 820 (31.7%) |
| Strongly agree (5) | 451 (17.4%) | 702 (27.1%) |

---

## 3. Predicted Probability Analysis

### 3a. Mean Predicted Probabilities by GT Class

![Mean Probs by GT](../figures/womenwork_verbsampling_plots_20260418_191839/mean_probs_by_gt.png)

| | p(SD=1) | p(D=2) | p(N=3) | p(A=4) | p(SA=5) |
|---|---|---|---|---|---|
| Overall mean | 0.0979 | 0.1524 | 0.1683 | 0.3208 | 0.2607 |

### 3b. Prediction Entropy

![Entropy Distribution](../figures/womenwork_verbsampling_plots_20260418_191839/entropy.png)

| Metric | Value |
|---|---|
| Mean entropy | 1.9973 bits |
| Median entropy | 2.0087 bits |
| Max possible (5 classes) | 2.322 bits |
| High-confidence predictions (entropy < 0.5) | 0 (0.0%) |

---

## 4. Confusion Matrix

![Confusion Matrix](../figures/womenwork_verbsampling_plots_20260418_191839/confusion_matrix.png)

| | **Pred SD(1)** | **Pred D(2)** | **Pred N(3)** | **Pred A(4)** | **Pred SA(5)** |
|---|---|---|---|---|---|
| **GT SD(1)** | 50 | 66 | 36 | 73 | 57 |
| **GT D(2)** | 60 | 111 | 118 | 170 | 143 |
| **GT N(3)** | 37 | 54 | 74 | 108 | 88 |
| **GT A(4)** | 53 | 123 | 140 | 315 | 261 |
| **GT SA(5)** | 32 | 48 | 64 | 154 | 153 |

---

## 5. Normalised Confusion Matrix

![Normalised Confusion Matrix](../figures/womenwork_verbsampling_plots_20260418_191839/confusion_matrix_normalised.png)

> Row-normalised: shows what the model predicts *given* the true class.

---

## 6. Per-class Metrics

| Class | Support | Precision | Recall | F1 |
|---|---|---|---|---|
| Strongly disagree (1) | 282 | 0.2155 | 0.1773 | 0.1946 |
| Disagree (2) | 602 | 0.2761 | 0.1844 | 0.2211 |
| Neither (3) | 361 | 0.1713 | 0.2050 | 0.1866 |
| Agree (4) | 892 | 0.3841 | 0.3531 | 0.3680 |
| Strongly agree (5) | 451 | 0.2179 | 0.3392 | 0.2654 |
| **Macro avg** | 2588 | 0.2530 | 0.2518 | 0.2471 |
| **Weighted avg** | 2588 | 0.2820 | 0.2716 | 0.2718 |

---

## 7. Accuracy by Region

![Accuracy by Region](../figures/womenwork_verbsampling_plots_20260418_191839/accuracy_by_region.png)

| Region | N | Accuracy |
|---|---|---|
| Kuzeydoğu Anadolu Bölgesi | 69 | 0.3478 |
| Orta Anadolu Bölgesi | 146 | 0.3425 |
| Doğu Karadeniz Bölgesi | 83 | 0.3373 |
| Ortadoğu Anadolu Bölgesi | 133 | 0.3008 |
| Güneydoğu Anadolu Bölgesi | 299 | 0.2876 |
| Batı Karadeniz Bölgesi | 164 | 0.2866 |
| Ege Bölgesi | 326 | 0.2791 |
| Batı Marmara Bölgesi | 114 | 0.2632 |
| İstanbul Bölgesi | 418 | 0.2608 |
| Doğu Marmara Bölgesi | 243 | 0.2593 |
| Akdeniz Bölgesi | 329 | 0.2584 |
| Batı Anadolu Bölgesi | 264 | 0.1894 |

---

## 8. Accuracy by Gender

![Accuracy by Gender](../figures/womenwork_verbsampling_plots_20260418_191839/accuracy_by_gender.png)

| Gender | N | Accuracy |
|---|---|---|
| Erkek | 1322 | 0.2806 |
| Kadın | 1266 | 0.2622 |

---

## 9. Accuracy by Age Group

![Accuracy by Age Group](../figures/womenwork_verbsampling_plots_20260418_191839/accuracy_by_age.png)

---

## 10. Notes

- Parse failures: **0** personas (`0.0%`).
- Mean entropy 1.997 bits out of max 2.322 — model is quite uncertain on average.
