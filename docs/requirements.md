# REQUIREMENTS SPECIFICATION

# Project Title

**BrainTumorAI - Hệ thống phát hiện và giải thích khối u não trên ảnh MRI sử dụng EfficientNetB0 và Explainable AI**

---

# 1. Giới thiệu

## 1.1 Mô tả bài toán

Khối u não là một trong những bệnh lý nguy hiểm có thể ảnh hưởng nghiêm trọng đến sức khỏe và tính mạng người bệnh. Việc phát hiện sớm và chính xác vị trí khối u từ ảnh cộng hưởng từ (MRI) đóng vai trò quan trọng trong quá trình chẩn đoán và điều trị.

Hiện nay, trí tuệ nhân tạo (Artificial Intelligence - AI) và học sâu (Deep Learning) đã đạt được nhiều thành tựu trong lĩnh vực phân tích ảnh y tế. Dự án này hướng đến việc xây dựng một hệ thống hỗ trợ phát hiện và phân loại khối u não từ ảnh MRI, đồng thời cung cấp khả năng giải thích quyết định của mô hình nhằm tăng độ tin cậy cho người sử dụng.

---

## 1.2 Mục tiêu dự án

Xây dựng hệ thống AI có khả năng:

* Phát hiện sự xuất hiện của khối u não trên ảnh MRI.
* Phân loại loại khối u não.
* Khoanh vùng vị trí khối u.
* Trực quan hóa vùng ảnh mà mô hình tập trung khi đưa ra dự đoán.
* Cung cấp giao diện thân thiện cho người dùng tải ảnh và nhận kết quả.
* Giải thích lý do AI đưa ra quyết định.
* Heatmap giải thích quyết định AI.

---

# 2. Phạm vi dự án

## Trong phạm vi

* Xử lý ảnh MRI não.
* Huấn luyện mô hình Deep Learning.
* Phân loại ảnh MRI thành 4 lớp:
  * Glioma
  * Meningioma
  * Pituitary
  * No Tumor
* Giải thích kết quả bằng Explainable AI.
* Xây dựng giao diện Web AI Demo.

## Ngoài phạm vi

* Thay thế bác sĩ chẩn đoán.
* Chẩn đoán lâm sàng thực tế.
* Phân tích dữ liệu bệnh án điện tử.
* Hỗ trợ ảnh CT Scan hoặc X-Ray.

---

# 3. Dataset

## Dataset sử dụng

Brain Tumor MRI Dataset

### Các lớp dữ liệu

| Class      | Ý nghĩa          |
| ---------- | ------------------ |
| Glioma     | U thần kinh đệm |
| Meningioma | U màng não       |
| Pituitary  | U tuyến yên      |
| No Tumor   | Không có khối u |

### Định dạng dữ liệu

* JPG
* PNG

### Kích thước

* Khoảng 7000+ ảnh MRI

---

# 4. Kiến trúc hệ thống

MRI Image
     │
     ▼
Data Preprocessing
     │
     ▼
EfficientNetB0
     │
     ├────────► Tumor Classification
     │
     └────────► Grad-CAM
                    │
                    ▼
            Explainable Result
                    │
                    ▼
              Web Dashboard

---

# 5. Yêu cầu chức năng

## FR-01: Tải ảnh MRI

Người dùng có thể:

* Upload ảnh MRI từ máy tính.
* Xem ảnh trước khi phân tích.

---

## FR-02: Phân loại khối u

Hệ thống phải:

* Nhận ảnh MRI đầu vào.
* Dự đoán loại khối u.

Kết quả gồm:

* Tên loại khối u.
* Xác suất dự đoán.

---

## FR-03: Giải thích dự đoán

Hệ thống phải:

* Sinh Heatmap bằng Grad-CAM.
* Hiển thị vùng ảnh được AI tập trung.

---

## FR-04: Hiển thị kết quả

Hệ thống phải hiển thị:

* Ảnh gốc.
* Kết quả dự đoán.
* Độ tin cậy.
* Heatmap giải thích.

---

## FR-05: Quản lý mô hình

Hệ thống cho phép:

* Lưu mô hình đã huấn luyện.
* Tải lại mô hình để dự đoán.

---

# 6. Yêu cầu phi chức năng

## NFR-01: Hiệu năng

* Thời gian dự đoán dưới 5 giây/ảnh.

---

## NFR-02: Độ chính xác

Mục tiêu:

* Accuracy ≥ 90%.

---

## NFR-03: Khả năng sử dụng

* Giao diện đơn giản.
* Dễ thao tác.

---

## NFR-04: Khả năng mở rộng

Hệ thống có thể mở rộng:

* Segmentation.
* Detection.
* Hỗ trợ thêm dữ liệu MRI mới.

---

# 7. Quy trình khoa học dữ liệu

## Giai đoạn 1: Data Understanding

Thực hiện:

* Kiểm tra dữ liệu.
* Thống kê số lượng ảnh.
* Phân tích phân bố lớp.

Kết quả:

* Báo cáo EDA.

---

## Giai đoạn 2: Data Preparation

Bao gồm:

* Resize ảnh 224x224.
* Normalize.
* Data Augmentation.

Các kỹ thuật:

* Rotation
* Flip
* Zoom
* Brightness Adjustment

---

## Giai đoạn 3: Modeling

### Baseline Models

* CNN
* ResNet50

### Main Model

EfficientNetB0

---

## Giai đoạn 4: Evaluation

Các chỉ số:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

---

## Giai đoạn 5: Explainable AI

Áp dụng:

* Grad-CAM

Mục tiêu:

* Giải thích quyết định của mô hình.

---

# 8. Công nghệ sử dụng

| Thành phần         | Công nghệ         |
| -------------------- | ------------------- |
| Programming Language | Python              |
| Data Processing      | Pandas, NumPy       |
| Visualization        | Matplotlib, Seaborn |
| Deep Learning        | PyTorch             |
| Transfer Learning    | EfficientNetB0      |
| Explainable AI       | Grad-CAM            |
| Backend API          | FastAPI             |
| Frontend             | Streamlit           |
| Database             | SQLite              |
| Deployment           | Docker              |

---

# 9. Cấu trúc thư mục dự án

```text
BrainTumorAI/

├── data/
│
├── notebooks/
│
├── src/
│   ├── preprocessing/
│   ├── training/
│   ├── inference/
│   └── explainability/
│
├── models/
│
├── app/
│   ├── frontend/
│   └── backend/
│
├── reports/
│
├── requirements.txt
│
└── README.md
```

# 10. Kết quả mong đợi

* Xây dựng thành công hệ thống phát hiện khối u não từ ảnh MRI.
* Đạt Accuracy trên 90%.
* Hiển thị được vùng tổn thương bằng Grad-CAM.
* Xây dựng Web Application hoàn chỉnh.
* Triển khai Demo phục vụ báo cáo đồ án.

# 11. Sản phẩm bàn giao

## Source Code

* Mã nguồn hoàn chỉnh.
* Tài liệu hướng dẫn cài đặt.

## AI Model

* File trọng số mô hình.
* File cấu hình.

## Báo cáo

* Báo cáo đồ án.
* Slide thuyết trình.

## Demo

* Web Application chạy thực tế.
* Video Demo hệ thống.
