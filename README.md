# 🎓 EasyLens: Deep Learning Model Reviewer & Defense Documentation

> **Assistive Vision System for the Visually Impaired through MobileNetV2 Deep Learning & Google ML Kit**  
> *Holy Angel University – School of Computing (4th Year Undergraduate Thesis)*

---

## 📌 Repository Contents

| File | Description |
| :--- | :--- |
| 📖 [**`EASYLENS_MODEL_REVIEW_DEFENSE_GUIDE.md`**](./EASYLENS_MODEL_REVIEW_DEFENSE_GUIDE.md) | **Master Thesis Defense Guide & Review Documentation:** Detailed analysis covering all 4 training phases, hyperparameter matrices, dataset engineering, ELI5 ("Explain Like I'm 5") mental models, and panel defense Q&A. |
| 📓 [**`EasyLens (3).ipynb`**](./EasyLens%20(3).ipynb) | **Full Jupyter Training & Evaluation Notebook:** Google Colab notebook containing complete code for data preprocessing, MobileNetV2 transfer learning, evaluation metrics, confusion matrices, and latency benchmarking. |

---

## ⚡ Performance Summary

```
+---------------------------------------------------------------------------------------------------+
|                                     EASYLENS MODEL METRICS                                        |
+--------------------------+---------------------------+--------------------------------------------+
| Top-1 Test Accuracy      | 85.55%                    | Evaluated on 2,125 unseen test images      |
| Balanced Accuracy        | 85.02%                    | Unbiased across class frequencies          |
| Top-2 Categorical Acc.   | 92.10%                    | Correct class is in top 2 choices          |
| Top-3 Categorical Acc.   | 94.54%                    | Correct class is in top 3 choices          |
| Weighted Precision       | 86.63%                    | High confidence, low false-positive rate   |
| Macro ROC-AUC            | 0.9902                    | Outstanding class separability (0 to 1)    |
| Average Inference Latency| 2.48 ms / image           | >10x faster than 30 FPS real-time threshold|
| Frame Rate Throughput    | 402.45 FPS                | Edge-optimized for smartphone hardware     |
+--------------------------+---------------------------+--------------------------------------------+
```

---

## 🔄 4-Phase Fine-Tuning Strategy

1. **Phase 1 (Warm-up & Base Freezing):** Frozen MobileNetV2 backbone, trained custom Dense head ($512 \rightarrow 256 \rightarrow 30$). Initial baseline reached ~83.47%.
2. **Phase 2 (Mid-Level Unfreezing & Class Weight Mitigation):** Unfroze top 30 layers with `Adam(lr=1e-5)` and balanced class weights to force the model to learn rare minority objects.
3. **Data Cleaning Intermission (24-Class Purge):** Removed unlearnable "ghost" classes ($\le 3$ images) and merged duplicate `person` casings to create a clean, statistically sound 24-class dataset.
4. **Phase 3 (Deep Fine-Tuning):** Full unfreezing of all 155 layers with `Adam(lr=5e-6)`, achieving **85.47% accuracy**, **92.10% Top-2**, and **94.54% Top-3** accuracy.
5. **Phase 4 (Micro-Optimization):** Delicate fine-tuning with `Adam(lr=1e-7)` down to `1e-8`, achieving final plateau convergence at **85.55% accuracy** and **86.63% precision**.

---

## 🎯 24 Curated High-Priority Navigation & Hazard Classes
* **Vehicular/Transit:** `Bus`, `Truck`, `car`, `train`, `motorcycle`, `bicycle`, `scooter`
* **Pedestrian Infrastructure:** `crosswalk`, `stairs`, `door`, `elevator`, `stop_sign`, `traffic_cone`, `fire_hydrant`
* **Signaling/Safety:** `red_light`, `green_light`, `yellow_light`, `Person`, `gun`
* **Environmental/Obstacles:** `pothole`, `tree`, `branch`, `Bushes`, `rat`

---

## 📖 For Detailed Defense Preparation
Refer to [**`EASYLENS_MODEL_REVIEW_DEFENSE_GUIDE.md`**](./EASYLENS_MODEL_REVIEW_DEFENSE_GUIDE.md) for full explanations, ELI5 analogies, architectural diagrams, confusion matrix analysis, and anticipated defense panel Q&As.
