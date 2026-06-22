# BrainTumorAI v2.0

**Hệ thống phát hiện và giải thích khối u não trên ảnh MRI sử dụng PyTorch (EfficientNetB0), Explainable AI (Grad-CAM), FastAPI và Next.js**

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-orange.svg)
![Next.js](https://img.shields.io/badge/Next.js-16-black.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688.svg)

---

## Giới thiệu dự án

Dự án **BrainTumorAI** là một hệ thống phân loại ảnh MRI não thành **4 lớp**, đi kèm với bản đồ giải thích (Heatmap) giúp bác sĩ/người dùng hiểu được quyết định của AI.

| Lớp                 | Mô tả                                                        |
| -------------------- | -------------------------------------------------------------- |
| **Glioma**     | U thần kinh đệm — phát sinh từ tế bào thần kinh đệm |
| **Meningioma** | U màng não — thường lành tính, phát triển chậm       |
| **Pituitary**  | U tuyến yên — có thể gây rối loạn hormone              |
| **No Tumor**   | Não bình thường — lớp đối chứng âm tính             |

**Điểm nhấn công nghệ:**

- **Mô hình**: PyTorch + `timm` (EfficientNetB0 làm mặc định) hoặc DenseNet121 (đối chiếu chuyên sâu y tế).
- **Explainable AI (XAI)**: Tích hợp **Grad-CAM** làm nổi bật các khu vực quan trọng quyết định dự đoán.
- **Backend API**: Xây dựng bằng FastAPI, xử lý song song, trả về xác suất và ảnh Grad-CAM Base64.
- **Frontend**: Giao diện web hiện đại xây dựng trên nền tảng **Next.js (App Router)** và **Tailwind CSS**.

---

## Cấu trúc dự án

```text
BrainTumorAI/
├── app/
│   ├── frontend/         # Next.js Web App (React 19, Tailwind CSS v4)
│   └── backend/          # FastAPI REST API (Inference & Grad-CAM)
├── data/
│   ├── Training/         # Dữ liệu huấn luyện (4 thư mục lớp)
│   └── Testing/          # Dữ liệu kiểm thử
├── models/               # Nơi lưu các file checkpoint (.pth)
├── src/
│   ├── config.py         # Cấu hình tập trung toàn bộ dự án
│   ├── utils.py          # Helper functions (seed, paths)
│   ├── preprocessing/    # Xử lý dữ liệu, augmentation pipeline
│   ├── training/         # Cấu trúc huấn luyện 2-phase, EarlyStopping
│   ├── inference/        # Lớp TumorPredictor
│   └── explainability/   # Tích hợp Grad-CAM tự động phát hiện target layer
├── tests/                # Bộ kiểm thử tự động (Pytest)
├── .env.example          # Mẫu biến môi trường
├── docker-compose.yml    # Docker config
├── pytest.ini            # Cấu hình Pytest
├── requirements.txt      # Python dependencies
└── README.md             # Tài liệu này
```

---

## Bắt đầu nhanh

### 1. Cài đặt Backend (FastAPI & PyTorch)

Yêu cầu: Python 3.9+# Tạo và kích hoạt môi trường ảo

### 2. Cài đặt Frontend (Next.js)

Yêu cầu: Node.js 18+

```bash
cd app/frontend
npm install
```

### 3. Khám phá & Tiền xử lý dữ liệu (EDA)

```python
1# Chạy trong môi trường Python hoặc Jupyter Notebook
from src.preprocessing.eda import full_eda_report
full_eda_report(data_dir="data/Training")
# Báo cáo biểu đồ sẽ lưu tại: reports/figures/eda/
```

### 4. Huấn luyện mô hình (Training)

Mô hình mặc định được tối ưu qua phương pháp **2-phase fine-tuning**:

1. Đóng băng backbone, chỉ train classifier (5 epochs, LR cao).
2. Mở băng toàn bộ, fine-tune toàn hệ thống (25 epochs, LR thấp).

```bash
# Quay lại thư mục gốc dự án
# Train bằng EfficientNetB0 (Khuyến nghị)
python src/training/train.py --model-name efficientnet --epochs 30

# Train bằng DenseNet121 (Làm mô hình đối chiếu chuyên sâu)
python src/training/train.py --model-name densenet --epochs 20
```

> Trọng số tốt nhất sẽ tự động được lưu vào thư mục `models/`.

### 5. Khởi động Hệ thống (Phục vụ Dự đoán)

Cần mở 2 Terminal riêng biệt:

**Terminal 1 — Backend (FastAPI)**

```bash
# Đứng tại thư mục gốc của dự án, kích hoạt venv (nếu chưa)
uvicorn app.backend.main:app --reload --host 127.0.0.1 --port 8000
```

**Terminal 2 — Frontend (Next.js)**

```bash
cd app/frontend
npm run dev
# Mở trình duyệt tại: http://localhost:3000
```

---

## Kiểm thử (Testing)

Dự án có bộ test suite với hàng chục test cases bao phủ toàn bộ pipeline:

```bash
pytest tests/ -v
```

*Gồm: test_api.py, test_dataset.py, test_eda.py, test_models.py, test_transforms.py, test_utils.py*

---

## Pipeline Machine Learning

```text
data/Training/
    │
    ↓ [EDA] class_distribution → check_integrity
    ↓ [Preprocessing] Stratified split 80/20 · class_weight
    ↓ [Augmentation] Flip · Rotation · Affine · ColorJitter · RandomErasing
    ↓ [Phase 1] Freeze backbone → train classifier (lr=1e-3)
    ↓ [Phase 2] Unfreeze all → fine-tune (lr=1e-4)
    ↓ [EarlyStopping] patience=5 · restore best weights
    ↓ [Checkpoint] models/efficientnet_best.pth
    ↓ [Evaluate] Confusion Matrix · ROC-AUC
    ↓ [Serve] FastAPI /predict → Grad-CAM overlay → Next.js Web App
```

---

## API Endpoints (Backend)

| Method   | Endpoint     | Mô tả                                        |
| -------- | ------------ | ---------------------------------------------- |
| `GET`  | `/`        | Thông tin API cơ bản                        |
| `GET`  | `/health`  | Kiểm tra trạng thái mô hình và kết nối |
| `GET`  | `/classes` | Danh sách 4 nhãn phân loại                 |
| `POST` | `/predict` | Phân loại ảnh + Trả về Grad-CAM           |
| `GET`  | `/docs`    | Giao diện thử nghiệm Swagger UI             |

**Request `/predict`:** Upload file (PNG / JPEG / JPG), giới hạn mặc định 10MB.

**Response `/predict`:**

```json
{
  "class_name": "GLIOMA",
  "confidence": 0.923456,
  "probabilities": {
    "glioma": 0.923456,
    "meningioma": 0.042100,
    "notumor": 0.020200,
    "pituitary": 0.014244
  },
  "heatmap_base64": "/9j/4AAQSkZJRgABAQ..."
}
```

---

## Đánh giá hiệu năng (Ước tính)

| Model                                          | Accuracy | Macro F1 |
| ---------------------------------------------- | -------- | -------- |
| **EfficientNetB0** (2-phase fine-tuning) | ~95–97% | ~0.95    |
| **DenseNet121** (Đối chiếu)                    | ~94–96% | ~0.94    |

---

## Tuyên bố miễn trừ trách nhiệm

> ⚠️ **Lưu ý y tế:** Hệ thống này là **công cụ học thuật và hỗ trợ sàng lọc**, **KHÔNG** thay thế cho các chẩn đoán chính thức của bác sĩ chuyên khoa. Mọi quyết định lâm sàng phải dựa trên thăm khám thực tế và ý kiến chuyên môn.

---

**Tác giả**: Huỳnh Thế Hy
**Email**: huynhthehy2005@gmail.com
**Cập nhật lần cuối**: 2026-06-22 — Phiên bản 2.0.1
