# BrainTumorAI

**Hệ thống phát hiện và giải thích khối u não trên ảnh MRI sử dụng Vision Transformer và Explainable AI**

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-orange.svg)

## 📌 Giới thiệu dự án

Khối u não là một trong những bệnh lý nguy hiểm có thể ảnh hưởng nghiêm trọng đến sức khỏe và tính mạng người bệnh. Dự án **BrainTumorAI** hướng đến việc xây dựng một hệ thống hỗ trợ phát hiện và phân loại khối u não từ ảnh cộng hưởng từ (MRI) thành 4 lớp:
- **Glioma** (U thần kinh đệm)
- **Meningioma** (U màng não)
- **Pituitary** (U tuyến yên)
- **No Tumor** (Không có khối u)

Hệ thống sử dụng kiến trúc **Vision Transformer (ViT-B16)** để phân loại, đồng thời cung cấp khả năng giải thích quyết định của AI thông qua kỹ thuật **Explainable AI (Grad-CAM / Attention Rollout)** nhằm tăng độ tin cậy cho người sử dụng.

## 🏗️ Cấu trúc dự án

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

## 🚀 Hướng dẫn sử dụng (Sẽ cập nhật sau khi build)

- **Training**: 
  - `python src/training/train.py`
- **Chạy Backend Server**:
  - `uvicorn app.backend.main:app --reload`
- **Chạy Frontend Dashboard**:
  - `streamlit run app/frontend/app.py`

## 👥 Nhóm phát triển
*Đang cập nhật...*
