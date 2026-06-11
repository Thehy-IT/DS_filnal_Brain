# BrainTumorAI

**Hệ thống phát hiện và giải thích khối u não trên ảnh MRI sử dụng Vision Transformer và Explainable AI**

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-orange.svg)

## Giới thiệu dự án

Khối u não là một trong những bệnh lý nguy hiểm có thể ảnh hưởng nghiêm trọng đến sức khỏe và tính mạng người bệnh. Dự án **BrainTumorAI** hướng đến việc xây dựng một hệ thống hỗ trợ phát hiện và phân loại khối u não từ ảnh cộng hưởng từ (MRI) thành 4 lớp:

- **Glioma** (U thần kinh đệm)
- **Meningioma** (U màng não)
- **Pituitary** (U tuyến yên)
- **No Tumor** (Không có khối u)

Hệ thống sử dụng kiến trúc **Vision Transformer (ViT-B16)** để phân loại, đồng thời cung cấp khả năng giải thích quyết định của AI thông qua kỹ thuật **Explainable AI (Grad-CAM / Attention Rollout)** nhằm tăng độ tin cậy cho người sử dụng.

## Cấu trúc dự án

```bash
BrainTumorAI/
├── app/                  # Frontend & Backend Code
│   ├── frontend/         # Streamlit UI
│   └── backend/          # FastAPI REST API
├── data/                 # Raw and Processed Data
├── docs/                 # Documentation & Requirements
├── models/               # Saved Model Weights (.pth)
├── notebooks/            # Jupyter Notebooks for EDA & Prototyping
├── reports/              # Generated reports & graphs
├── src/                  # Source Code
│   ├── preprocessing/    # Data preparation pipelines
│   ├── training/         # Model definition & training loops
│   ├── inference/        # Model loading & inference
│   └── explainability/   # Grad-CAM / XAI implementation
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## ⚙️ Hướng dẫn cài đặt

### 1. Khởi tạo môi trường ảo (Virtual Environment)

Khuyến nghị sử dụng Python 3.9 hoặc mới hơn.

```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường (Windows)
venv\Scripts\activate

# Kích hoạt môi trường (Linux/Mac)
source venv/bin/activate
```

### 2. Cài đặt các thư viện cần thiết

```bash
pip install -r requirements.txt
```

## Hướng dẫn sử dụng

### 3. Chuẩn bị dữ liệu

Đảm bảo dữ liệu MRI được tổ chức theo cấu trúc:

```
data/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

### 4. Huấn luyện mô hình

#### Huấn luyện với Vision Transformer (ViT-B16 - Khuyến nghị):

```bash
python src/training/train.py \
  --data-dir data/Training \
  --model-name vit \
  --batch-size 32 \
  --epochs 50 \
  --lr 1e-4 \
  --save-dir models
```

#### Huấn luyện với CNN Baseline:

```bash
python src/training/train.py \
  --data-dir data/Training \
  --model-name cnn \
  --batch-size 32 \
  --epochs 30 \
  --lr 1e-3 \
  --save-dir models
```

**Tùy chọn phổ biến:**

- `--data-dir`: Đường dẫn đến thư mục dữ liệu huấn luyện
- `--model-name`: Tên mô hình (`vit` hoặc `cnn`)
- `--batch-size`: Kích thước batch (mặc định: 32)
- `--epochs`: Số epochs (mặc định: 50)
- `--lr`: Learning rate (mặc định: 1e-4)
- `--save-dir`: Thư mục lưu mô hình (mặc định: `models/`)

Mô hình tốt nhất sẽ được lưu tại `models/vit_best.pth` hoặc `models/cnn_best.pth`.

### 5. Chạy Backend Server (FastAPI)

```bash
uvicorn app.backend.main:app --reload --host 127.0.0.1 --port 8000
```

API Documentation sẽ có sẵn tại `http://127.0.0.1:8000/docs` (Swagger UI).

**Endpoint chính:**

- `POST /predict`: Nhận ảnh MRI và trả về dự đoán + heatmap Grad-CAM
  - Input: `file` (PNG, JPG, JPEG)
  - Output: JSON với `class_name`, `confidence`, `heatmap_base64`

### 6. Chạy Frontend Dashboard (Streamlit)

```bash
streamlit run app/frontend/app.py
```

Dashboard sẽ mở tại `http://localhost:8501`. Bạn có thể:

- Tải ảnh MRI lên
- Xem dự đoán mô hình
- Quan sát heatmap Grad-CAM để hiểu quyết định của AI

### 7. Chạy toàn bộ hệ thống với Docker Compose

```bash
docker-compose up -d
```

Truy cập:

- Frontend: `http://localhost:8501`
- Backend API: `http://localhost:8000`
- API Docs: `http://localhost:8000/docs`

### 8. Phân tích dữ liệu với Jupyter Notebooks

```bash
jupyter notebook
```

Mở `notebooks/01_EDA_Deep_Analysis.ipynb` để:

- Khám phá phân bố dữ liệu
- Phân tích đặc trưng hình ảnh
- Trực quan hóa các ví dụ từng lớp

---

## Kết quả và Hiệu suất

### Hiệu suất mô hình (Ước tính):

- **Vision Transformer (ViT-B16)**: ~95-97% accuracy
- **Baseline CNN**: ~85-90% accuracy

### Explainability:

- **Grad-CAM**: Trực quan hóa vùng mô hình chú ý khi phân loại
- **Attention Rollout**: Tổng hợp attention từ các layer để hiểu quyết định

---

## Tài liệu bổ sung

- [Requirements Specification](docs/requirements.md) - Mô tả chi tiết yêu cầu dự án
- [Jupyter Notebooks](notebooks/) - EDA, prototyping, và phân tích sâu
- [Reports](reports/) - Báo cáo kết quả và trực quan hóa

---

## Đóng góp

Nếu bạn muốn đóng góp vào dự án, vui lòng:

1. Fork repository này
2. Tạo branch mới (`git checkout -b feature/improvement`)
3. Commit thay đổi của bạn (`git commit -m 'Add improvement'`)
4. Push lên branch (`git push origin feature/improvement`)
5. Tạo Pull Request

## Nhóm phát triển

| Vai trò                             | Người phát triển |
| ------------------------------------ | -------------------- |
| **Project Lead & ML Engineer** | Tên không có sẵn |
| **Data Engineer**              | Tên không có sẵn |
| **Backend Developer**          | Tên không có sẵn |
| **Frontend Developer**         | Tên không có sẵn |

*Để cập nhật thông tin nhóm, vui lòng chỉnh sửa README.md này.*

---

## Liên hệ

- **Email**: huynhthehy2005@gmail.com
- **Issues & Questions**: Mở issue trên GitHub
- **Documentation**: Xem [docs/](docs/) folder

---

**Last Updated**: 2026-06-11
