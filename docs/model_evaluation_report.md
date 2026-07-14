# Báo Cáo Đánh Giá Mô Hình — Brain Tumor Classification

> **Người đánh giá:** Huỳnh Thế Hy
> **Ngày đánh giá:** 2026-07-11
> **Phiên bản:** v1.0
> **Dự án:** DS Final Brain — Phân loại u não bằng Deep Learning

---

## 1. Tổng Quan Dự Án

| Thuộc tính              | Giá trị                                              |
| ------------------------- | ------------------------------------------------------ |
| **Task**            | Multi-class Image Classification (4 lớp)              |
| **Dataset**         | Brain MRI — 5,600 ảnh (cân bằng: 1,400/lớp)       |
| **Train/Val split** | 4,480 / 1,120 (80/20, stratified, seed=42)             |
| **Test set**        | 1,600 ảnh (riêng biệt)                              |
| **Lớp mục tiêu** | `glioma`, `meningioma`, `notumor`, `pituitary` |
| **Framework**       | PyTorch (CPU training)                                 |
| **Random Seed**     | 42 (reproducibility ✓)                                |

---

## 2. Kiến Trúc & Cấu Hình Training

Cả hai model đều áp dụng chiến lược **2-phase transfer learning**:

| Phase             | Mô tả                                       | Epochs tối đa | Learning Rate          |
| ----------------- | --------------------------------------------- | --------------- | ---------------------- |
| **Phase 1** | Frozen backbone — chỉ train classifier head | 5               | 1e-3                   |
| **Phase 2** | Unfreeze toàn bộ — full fine-tuning        | 35              | 1e-4 → 5e-5 → 2.5e-5 |

**Các kỹ thuật áp dụng:**

- ✅ EarlyStopping (patience = 7 epochs)
- ✅ Learning Rate Scheduler (giảm dần theo giai đoạn)
- ✅ Checkpoint lưu best val_acc
- ✅ Hardware safety pause (5s cooldown — CPU training)

---

## 3. Kết Quả Training Chi Tiết

### 3.1 EfficientNetB0

#### Phase 1 — Warm-up Classifier Head (5 epochs)

| Epoch         | Train Loss       | Train Acc        | Val Loss         | Val Acc          |
| ------------- | ---------------- | ---------------- | ---------------- | ---------------- |
| 1/5           | 2.0936           | 45.94%           | 1.7222           | 50.00%           |
| 2/5           | 1.1122           | 68.01%           | 1.3362           | 59.20%           |
| 3/5           | 0.8768           | 73.57%           | 1.3470           | 60.00%           |
| 4/5           | 0.8234           | 76.29%           | 1.0454           | 66.16%           |
| **5/5** | **0.7387** | **77.72%** | **1.0557** | **66.87%** |

#### Phase 2 — Full Fine-Tuning (18/35 epochs, early stopped)

| Epoch           | Train Loss       | Train Acc        | Val Loss         | Val Acc             | LR             |
| --------------- | ---------------- | ---------------- | ---------------- | ------------------- | -------------- |
| 1/35            | 0.5034           | 84.78%           | 0.3695           | 88.39%              | 1e-4           |
| 2/35            | 0.2236           | 92.34%           | 0.1975           | 93.04%              | 1e-4           |
| 3/35            | 0.1625           | 94.80%           | 0.1535           | 94.29%              | 1e-4           |
| 4/35            | 0.1129           | 95.60%           | 0.1797           | 93.84%              | 1e-4           |
| 5/35            | 0.0695           | 97.48%           | 0.1476           | 95.09%              | 1e-4           |
| 6/35            | 0.0603           | 97.77%           | 0.1236           | 95.89%              | 1e-4           |
| 7/35            | 0.0609           | 97.92%           | 0.0574           | 97.77%              | 1e-4           |
| 8/35            | 0.0562           | 98.06%           | 0.0579           | 97.68%              | 1e-4           |
| 9/35            | 0.0394           | 98.75%           | 0.0429           | 98.30%              | 1e-4           |
| 10/35           | 0.0290           | 99.06%           | 0.0489           | 98.66%              | 1e-4           |
| **11/35** | **0.0247** | **99.31%** | **0.0450** | **98.75% ★** | **1e-4** |
| 12/35           | 0.0268           | 99.08%           | 0.0674           | 97.59%              | 1e-4           |
| 13/35           | 0.0232           | 99.24%           | 0.0577           | 97.95%              | 1e-4           |
| 14/35           | 0.0276           | 99.11%           | 0.0582           | 97.86%              | 1e-4           |
| 15/35           | 0.0130           | 99.49%           | 0.0672           | 97.50%              | 5e-5           |
| 16/35           | 0.0168           | 99.40%           | 0.0542           | 98.21%              | 5e-5           |
| 17/35           | 0.0157           | 99.46%           | 0.0489           | 98.21%              | 5e-5           |
| 18/35           | 0.0106           | 99.64%           | 0.0449           | 98.75%              | 5e-5           |

> **EarlyStopping** kích hoạt sau 18 epochs — Best val_acc = **98.75%** (epoch 11)

#### Classification Report — Validation Set (1,120 ảnh)

| Class               | Precision      | Recall         | F1-Score       | Support        |
| ------------------- | -------------- | -------------- | -------------- | -------------- |
| glioma              | 0.98           | 0.99           | 0.99           | 280            |
| meningioma          | 0.98           | 0.98           | 0.98           | 280            |
| notumor             | 1.00           | 0.99           | 0.99           | 280            |
| pituitary           | 0.99           | 0.99           | 0.99           | 280            |
| **macro avg** | **0.99** | **0.99** | **0.99** | **1120** |

#### Test Set Performance (1,600 ảnh)

| Metric                         | Value            |
| ------------------------------ | ---------------- |
| **Test Accuracy**        | **94.69%** |
| **Macro F1**             | 94.57%           |
| **Macro Precision**      | 95.06%           |
| **Macro Recall**         | 94.69%           |
| **Macro AUC-ROC**        | 98.85%           |
| **Inference Time (CPU)** | 47.23s           |

**Per-class Test Set Detail:**

| Class      | Precision | Recall  | F1     | AUC-ROC |
| ---------- | --------- | ------- | ------ | ------- |
| glioma     | 99.39%    | 81.75%  | 89.71% | 96.06%  |
| meningioma | 91.33%    | 97.50%  | 94.32% | 99.46%  |
| notumor    | 90.50%    | 100.00% | 95.01% | 99.87%  |
| pituitary  | 99.00%    | 99.50%  | 99.25% | 99.99%  |

---

### 3.2 DenseNet121

#### Phase 1 — Warm-up Classifier Head (5 epochs)

| Epoch         | Train Loss       | Train Acc        | Val Loss         | Val Acc          |
| ------------- | ---------------- | ---------------- | ---------------- | ---------------- |
| 1/5           | 0.7446           | 72.10%           | 0.5492           | 80.98%           |
| 2/5           | 0.4962           | 82.66%           | 0.5782           | 79.46%           |
| 3/5           | 0.4363           | 84.04%           | 0.3945           | 88.75%           |
| 4/5           | 0.3999           | 86.18%           | 0.4283           | 86.43%           |
| **5/5** | **0.3856** | **85.54%** | **0.4137** | **87.68%** |

#### Phase 2 — Full Fine-Tuning (19/35 epochs, early stopped)

| Epoch           | Train Loss       | Train Acc        | Val Loss         | Val Acc             | LR             |
| --------------- | ---------------- | ---------------- | ---------------- | ------------------- | -------------- |
| 1/35            | 0.2658           | 90.65%           | 0.1448           | 95.54%              | 1e-4           |
| 2/35            | 0.1374           | 95.02%           | 0.1032           | 96.25%              | 1e-4           |
| 3/35            | 0.0827           | 96.94%           | 0.0682           | 97.59%              | 1e-4           |
| 4/35            | 0.0626           | 97.88%           | 0.0581           | 98.39%              | 1e-4           |
| 5/35            | 0.0396           | 98.88%           | 0.0412           | 99.02%              | 1e-4           |
| 6/35            | 0.0378           | 98.95%           | 0.0484           | 98.84%              | 1e-4           |
| 7/35            | 0.0343           | 98.88%           | 0.0505           | 98.21%              | 1e-4           |
| 8/35            | 0.0357           | 98.91%           | 0.0494           | 98.48%              | 1e-4           |
| 9/35            | 0.0302           | 98.86%           | 0.0341           | 98.93%              | 5e-5           |
| 10/35           | 0.0166           | 99.51%           | 0.0303           | 98.84%              | 5e-5           |
| 11/35           | 0.0201           | 99.42%           | 0.0326           | 98.93%              | 5e-5           |
| **12/35** | **0.0130** | **99.62%** | **0.0225** | **99.46% ★** | **5e-5** |
| 13/35           | 0.0094           | 99.82%           | 0.0289           | 99.38%              | 5e-5           |
| 14/35           | 0.0111           | 99.73%           | 0.0288           | 99.38%              | 5e-5           |
| 15/35           | 0.0102           | 99.71%           | 0.0248           | 99.02%              | 5e-5           |
| 16/35           | 0.0109           | 99.62%           | 0.0275           | 99.02%              | 2.5e-5         |
| 17/35           | 0.0074           | 99.82%           | 0.0273           | 99.20%              | 2.5e-5         |
| 18/35           | 0.0075           | 99.75%           | 0.0242           | 99.02%              | 2.5e-5         |
| 19/35           | 0.0112           | 99.75%           | 0.0246           | 99.11%              | 2.5e-5         |

> ⏹️ **EarlyStopping** kích hoạt sau 19 epochs — Best val_acc = **99.46%** (epoch 12)

#### Classification Report — Validation Set (1,120 ảnh)

| Class               | Precision      | Recall         | F1-Score       | Support        |
| ------------------- | -------------- | -------------- | -------------- | -------------- |
| glioma              | 1.00           | 0.99           | 0.99           | 280            |
| meningioma          | 0.98           | 0.99           | 0.98           | 280            |
| notumor             | 1.00           | 1.00           | 1.00           | 280            |
| pituitary           | 0.99           | 0.99           | 0.99           | 280            |
| **macro avg** | **0.99** | **0.99** | **0.99** | **1120** |

#### Test Set Performance (1,600 ảnh)

| Metric                         | Value            |
| ------------------------------ | ---------------- |
| **Test Accuracy**        | **95.00%** |
| **Macro F1**             | 94.90%           |
| **Macro Precision**      | 95.39%           |
| **Macro Recall**         | 95.00%           |
| **Macro AUC-ROC**        | 99.00%           |
| **Inference Time (CPU)** | 442.39s          |

**Per-class Test Set Detail:**

| Class      | Precision | Recall  | F1     | AUC-ROC |
| ---------- | --------- | ------- | ------ | ------- |
| glioma     | 99.70%    | 82.00%  | 89.99% | 96.69%  |
| meningioma | 88.31%    | 98.25%  | 93.02% | 99.33%  |
| notumor    | 94.79%    | 100.00% | 97.32% | 99.99%  |
| pituitary  | 98.76%    | 99.75%  | 99.25% | 99.99%  |

---

## 4. Bảng So Sánh Tổng Hợp

| Metric                         | EfficientNetB0   | DenseNet121 | Winner                          |
| ------------------------------ | ---------------- | ----------- | ------------------------------- |
| **Best Val Accuracy**    | 98.75%           | 99.46%      | 🏆 DenseNet121                  |
| **Test Accuracy**        | 94.69%           | 95.00%      | 🏆 DenseNet121                  |
| **Macro F1 (Test)**      | 94.57%           | 94.90%      | 🏆 DenseNet121                  |
| **Macro AUC-ROC (Test)** | 98.85%           | 99.00%      | 🏆 DenseNet121                  |
| **Glioma F1 (Test)**     | 89.71%           | 89.99%      | 🏆 DenseNet121                  |
| **Meningioma F1 (Test)** | 94.32%           | 93.02%      | 🏆 EfficientNetB0               |
| **Notumor F1 (Test)**    | 95.01%           | 97.32%      | 🏆 DenseNet121                  |
| **Pituitary F1 (Test)**  | 99.25%           | 99.25%      | 🤝 Tie                          |
| **Phase 2 Epochs Used**  | 18/35            | 19/35       | 🏆 EfficientNetB0               |
| **Inference Time (CPU)** | **47.23s** | 442.39s     | 🏆 EfficientNetB0 (9.4x faster) |
| **Val→Test Gap**        | 4.06%            | 4.46%       | 🏆 EfficientNetB0               |

---

## 5. Đánh Giá Chuyên Sâu

### 5.1 Điểm Mạnh — Cả Hai Model

**Chiến lược training đúng chuẩn:**

- Áp dụng đúng 2-phase transfer learning: warm-up head trước, fine-tune toàn bộ sau. Đây là best practice trong medical imaging.
- Sử dụng EarlyStopping với patience=7 — hợp lý, tránh overfitting nhưng không quá conservative.
- Dataset hoàn toàn cân bằng (1,400 ảnh/lớp) — loại bỏ class imbalance bias.
- Reproducibility tốt: seed=42, stratified split đảm bảo phân phối nhất quán.
- Đánh giá trên test set độc lập (1,600 ảnh) — điều bắt buộc trong bài toán y tế.

**Kết quả tổng thể ấn tượng:**

- Cả hai model đạt Macro AUC-ROC > 98.8% — xuất sắc với bài toán 4-class medical classification.
- Validation accuracy > 98.7% chứng tỏ model học được feature discriminative tốt.
- DenseNet Phase 1 warm-up nhanh hơn đáng kể (80.98% vs 50.00% ngay epoch 1) — cho thấy DenseNet feature extraction generalizes tốt hơn ngay từ đầu.

---

### 5.2 ⚠️ Điểm Yếu & Rủi Ro Cần Lưu Ý

#### 🔴 Vấn đề nghiêm trọng: Val → Test Generalization Gap

```
EfficientNetB0: Val 98.75% → Test 94.69%  (GAP = 4.06%)
DenseNet121:    Val 99.46% → Test 95.00%  (GAP = 4.46%)
```

Với bài toán medical imaging, gap > 3% giữa validation và test thường chỉ ra:

1. Val set không đủ đại diện cho test distribution (dù stratified, nhưng n=1,120 là nhỏ).
2. Augmentation không đủ mạnh để generalize sang MRI images có variation cao.
3. Overfitting vào validation set qua quá trình tuning checkpoint.

> **CẢNH BÁO:** Val-Test gap 4%+ trong medical imaging là red flag. Khi deploy, model sẽ gặp dữ liệu từ nhiều nguồn scanner khác nhau, gap thực tế có thể còn lớn hơn.

---

#### 🟠 Vấn đề nghiêm trọng về lâm sàng: Glioma Recall thấp

```
Glioma Recall (Test Set):
  EfficientNetB0 = 81.75%  →  18.25% cases BỊ BỎ SÓT
  DenseNet121    = 82.00%  →  18.00% cases BỊ BỎ SÓT
```

Model có Precision glioma rất cao (>99%) nhưng Recall chỉ ~82%:

- Model **rất ít báo nhầm** (false positive thấp) — tốt cho specificity.
- Nhưng model **bỏ sót ~18% ca glioma thực sự** — nguy hiểm trong screening.

> **CẢNH BÁO LÂM SÀNG:** Glioma grade cao là ung thư não nguy hiểm tính mạng. Recall 82% trên test set nghĩa là cứ 5 bệnh nhân glioma thì model bỏ sót 1 người. Đây là mức KHÔNG chấp nhận được nếu dùng làm screening tool độc lập. Model chỉ nên dùng như công cụ hỗ trợ bác sĩ, KHÔNG thay thế chẩn đoán.

---

#### 🔴 Vấn đề đặc biệt nghiêm trọng: Notumor Precision (Bỏ lọt u)

```
Notumor Precision (Test Set):
  EfficientNetB0 = 90.50%  →  Dự đoán ~442 ca Notumor (Trong đó có 42 ca LÀ BỆNH NHÂN CÓ U)
  DenseNet121    = 94.79%  →  Dự đoán ~422 ca Notumor (Trong đó có 22 ca LÀ BỆNH NHÂN CÓ U)
```

Mặc dù `Notumor Recall` đạt 100%, nhưng `Precision` thấp đồng nghĩa với việc hệ thống phán nhầm người CÓ U thành KHỎE MẠNH (False Negatives của toàn hệ thống).
- **EfficientNetB0 bỏ lọt 42 ca khối u.**
- **DenseNet121 bỏ lọt 22 ca khối u.**

> **CẢNH BÁO TỬ HUYỆT:** Trong y tế, báo nhầm người khỏe thành có u (False Positive) chỉ tốn thêm chi phí xét nghiệm, nhưng báo nhầm người có u thành khỏe mạnh là lỗi y khoa không thể chấp nhận. Ở tiêu chí sinh tử này, **DenseNet121 an toàn gần gấp đôi** EfficientNetB0.

---

#### 🟡 DenseNet121: Inference Time Chậm Hơn

```
DenseNet121 Inference: 442.39s / 1,600 ảnh ≈ 0.277 giây/ảnh
EfficientNetB0 Inference: 47.23s / 1,600 ảnh ≈ 0.030 giây/ảnh
```

DenseNet121 chậm hơn **9.4x** so với EfficientNetB0 trên CPU. Tuy nhiên, tốc độ 0.277s/ảnh vẫn hoàn toàn đạt chuẩn "Real-time" trong y tế (bác sĩ thường tốn hàng phút để phân tích 1 ca MRI). Do đó, sự chênh lệch này không ảnh hưởng đến trải nghiệm (UX) lâm sàng thực tế.

---

#### 🟡 Training Trên CPU Ảnh Hưởng Chất Lượng

Hardware cooldown pause 5s/epoch do CPU training tạo ra:

- Training time dài → giới hạn khả năng hyperparameter tuning và thử nhiều augmentation.
- Batch size bị hạn chế → gradient noise cao hơn.
- Khó áp dụng augmentation phức tạp (cutmix, mixup, elastic deformation).

---

### 5.3 Khuyến Nghị Model

**EfficientNetB0 vượt trội về tốc độ:**

- Inference nhanh hơn 9.4x (47.23s vs 442.39s cho 1,600 ảnh).
- Tương đương ~0.03s/ảnh so với ~0.27s/ảnh của DenseNet121.

**Nhưng DenseNet121 VƯỢT TRỘI về An Toàn Lâm Sàng (Clinical Safety):**

- Bỏ lọt ít ca bệnh u não hơn đáng kể so với EfficientNetB0 (22 ca bị phán nhầm là khỏe mạnh so với 42 ca của EfficientNet).
- Val→Test gap (4.46% vs 4.06%) không có sự chênh lệch quá lớn để đánh đổi sự an toàn tính mạng bệnh nhân.

> **KHUYẾN NGHỊ CUỐI CÙNG (CẬP NHẬT): Chọn DenseNet121 làm production model.**
> Quyết định này đặt yếu tố **An Toàn Bệnh Nhân (Patient Safety)** lên hàng đầu. Đánh đổi tốc độ từ 0.03s lên 0.27s/ảnh là hoàn toàn vô hình đối với workflow của bác sĩ (một bác sĩ mất từ 5-10 phút để xem một ca MRI), nhưng việc giảm được gần 50% số ca bỏ sót khối u (từ 42 ca xuống 22 ca) là một lợi ích y khoa khổng lồ không thể thương lượng.

---

## 6. Lộ Trình Cải Thiện

### Ưu Tiên Cao (Phải làm trước khi deploy)

| # | Vấn đề                               | Giải pháp cụ thể                                                                                                                          |
| - | --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 | **Tối ưu hóa An toàn Lâm sàng** | **Threshold Tuning (Zero-cost):** Hạ ngưỡng dự đoán (probability threshold) cho lớp `glioma` (vd: prob > 0.3 thay vì max) để đẩy Recall > 95%. Nâng ngưỡng cho lớp `notumor` (vd: prob > 0.8 mới kết luận khỏe mạnh) để giảm thiểu bỏ lọt khối u. |
| 2 | **Glioma recall ~82%**            | Dùng Focal Loss (gamma=2, alpha cho glioma=2.0); augment glioma mạnh hơn (elastic deform, brightness jitter); thử class-weighted sampling |
| 3 | **Val-Test gap > 4%**             | Test Time Augmentation (TTA, 5-10 crops); tăng augmentation pipeline; thêm Label Smoothing (eps=0.1)                                        |
| 4 | **Chưa có external validation** | Test trên ít nhất 1 external dataset (BraTS, Kaggle Brain MRI) để kiểm tra cross-site generalization                                    |

### Ưu Tiên Trung Bình

| # | Vấn đề                                   | Giải pháp                                                            |
| - | ------------------------------------------- | ---------------------------------------------------------------------- |
| 4 | **Chỉ 1 train/val split**            | K-Fold Cross-Validation (k=5) để có confidence interval             |
| 5 | **Thiếu uncertainty quantification** | Monte Carlo Dropout hoặc Deep Ensemble để model tự báo confidence |
| 6 | **Thiếu explainability**             | Grad-CAM visualization để bác sĩ hiểu model "nhìn" gì           |

### Ưu Tiên Thấp (Nice-to-have)

| # | Giải pháp                                                                             |
| - | --------------------------------------------------------------------------------------- |
| 7 | Calibration curve (reliability diagram) kiểm tra probability output                    |
| 8 | Ensemble EfficientNetB0 + DenseNet121 (dự kiến +1-2% test acc)                        |
| 9 | Knowledge distillation: dùng DenseNet121 làm teacher để distill vào EfficientNetB0 |

---

## 7. Benchmark Comparison

Theo y văn 2023-2024, các model trên Brain Tumor MRI 4-class classification:

| Nguồn                     | Accuracy               | Dataset                                      |
| -------------------------- | ---------------------- | -------------------------------------------- |
| State-of-art papers (2024) | 98-99%                 | Thường là val set nhỏ, không tách test |
| Production-grade systems   | 93-96%                 | Test set độc lập, diverse                 |
| **Dự án này**     | **94.69-95.00%** | **Test set 1,600 ảnh**                |

> Kết quả dự án nằm trong top tier của production-grade benchmarks. Tuy nhiên, cần cross-site validation trước khi áp dụng lâm sàng thực tế.

---

## 8. Executive Summary

| Hạng mục                          | EfficientNetB0       | DenseNet121             |
| ----------------------------------- | -------------------- | ----------------------- |
| **Training Strategy**         | ⭐⭐⭐⭐⭐           | ⭐⭐⭐⭐⭐              |
| **Val Performance**           | ⭐⭐⭐⭐ (98.75%)    | ⭐⭐⭐⭐⭐ (99.46%)     |
| **Test Performance**          | ⭐⭐⭐⭐ (94.69%)    | ⭐⭐⭐⭐ (95.00%)<br /> |
| **Generalization**            | ⭐⭐⭐ (gap 4.06%)   | ⭐⭐⭐ (gap 4.46%)      |
| **Glioma Safety**             | ⭐⭐ (recall 81.75%) | ⭐⭐ (recall 82.00%)    |
| **Inference Speed**           | ⭐⭐⭐⭐⭐ (47s)     | ⭐⭐ (442s)             |
| **Clinical Safety (Notumor)** | ⭐⭐ (Bỏ lọt 42 ca)  | ⭐⭐⭐⭐ (Bỏ lọt 22 ca)  |
| **Production Ready**          | ⭐⭐⭐               | ⭐⭐⭐⭐                 |
| **Điểm tổng (8 criteria)**    | **3.62/5**           | **3.87/5**               |

**Đây là kết quả research tốt.** Dựa trên tiêu chí an toàn lâm sàng, **DenseNet121 được lựa chọn làm model ưu tiên**. Để deploy trong hỗ trợ lâm sàng thực tế, cần áp dụng ngay Threshold Tuning để tăng Glioma recall > 95% và đảm bảo an toàn cho lớp Notumor, bổ sung external validation, uncertainty quantification, và explainability layer.

---

*Báo cáo được tạo bởi: Huỳnh  Thế Hy*
*Nguồn dữ liệu: `training_log.txt`, `training_log_densenet.txt`, `reports/model_comparison_report.json`*
*Ngày: 2026-07-11*
