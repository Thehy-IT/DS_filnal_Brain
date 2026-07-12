# Hướng dẫn chạy dự án BrainTumorAI (End-to-End Toàn Diện)

Tài liệu này cung cấp chi tiết toàn bộ vòng đời (lifecycle) của dự án Data Science: từ bước phân tích dữ liệu, huấn luyện, đánh giá, kiểm thử đến khởi chạy giao diện người dùng.

---

## Bước 1: Chuẩn bị Môi trường

**1.1. Cài đặt môi trường AI (Backend & Data Science)**
Mở Terminal ở thư mục gốc của dự án:
```bash
python -m venv venv

# Kích hoạt môi trường ảo (Trên Windows):
venv\Scripts\activate
# Trên Mac/Linux:
source venv/bin/activate

# Cài đặt toàn bộ thư viện AI
pip install -r requirements.txt
```

**1.2. Cài đặt môi trường Giao diện (Frontend)**
Mở thêm một Terminal mới, di chuyển vào thư mục frontend:
```bash
cd app/frontend
npm install
```

---

## Bước 2: Khám phá & Phân tích Dữ liệu (EDA)

Trước khi huấn luyện, bạn cần hiểu rõ đặc tả của dữ liệu. (Đảm bảo ảnh MRI đã nằm trong `data/Training` và `data/Testing`). Có 2 cách thực hiện:

**Cách 1: Xem qua Jupyter Notebook (Dành cho việc nghiên cứu)**
Mở Terminal (đã kích hoạt venv) và khởi chạy Jupyter:
```bash
jupyter notebook
```
- Trình duyệt sẽ tự động mở. Tại giao diện web, truy cập vào thư mục `notebooks/` và click mở file `01_EDA_Deep_Analysis.ipynb`.
- Nhấn nút "Run All" để xem các biểu đồ trực quan phân tích phân phối nhãn, pixel, và kích thước ảnh.

**Cách 2: Chạy Script xuất báo cáo tự động**
Mở Python console (đã kích hoạt venv) và chạy:
```python
from src.preprocessing.eda import full_eda_report
full_eda_report(data_dir="data/Training")
```
*Kết quả toàn bộ biểu đồ phân tích sẽ tự động được lưu vào `reports/figures/eda/`.*

---

## Bước 3: Huấn luyện Mô hình (Training)

Sau khi hiểu dữ liệu, tiến hành huấn luyện mô hình. Lệnh dưới đây dùng kiến trúc **EfficientNetB0** (tối ưu nhất cho dự án):
```bash
python src/training/train.py --model-name efficientnet --epochs 30
```
*Lưu ý:*
- Quá trình này sẽ sử dụng kỹ thuật **2-phase fine-tuning** và **Early Stopping** để chống Overfitting.
- Trọng số tốt nhất (best weights) sẽ được tự động lưu vào thư mục `models/` (ví dụ: `models/efficientnet_best.pth`).

---

## Bước 4: Đánh giá & Kiểm thử (Evaluate & Testing)

**4.1. Đánh giá hiệu năng mô hình (Evaluation)**
Để xem đánh giá chuyên sâu (Confusion Matrix, ROC-AUC Curve, F1-score) và so sánh các kiến trúc (EfficientNet vs DenseNet), bạn sử dụng Jupyter Notebook:

Mở Terminal (đã kích hoạt venv) và chạy:
```bash
jupyter notebook
```
- Trên trình duyệt, điều hướng vào thư mục `notebooks/` và mở file `02_Model_Compare_Evaluate_Visualize.ipynb`.
- Nhấn "Run All" để tiến hành quá trình so sánh và vẽ biểu đồ đánh giá tự động dựa trên `reports/training_history.json`.

**4.2. Chạy Kiểm thử tự động (Unit Tests)**
Kiểm tra xem toàn bộ logic dự án (từ Data Loader, Model Shape đến API) có bị lỗi nào không trước khi chạy Web:
```bash
pytest tests/ -v
```
*(Nếu tất cả test cases hiện màu xanh PASSED, hệ thống của bạn đã đạt chuẩn).*

**4.3. Nâng cao: Xuất mô hình sang định dạng ONNX (Tùy chọn)**
Để tối ưu hóa tốc độ chạy (inference) trên môi trường Production thực tế mà không cần cài đặt nguyên bộ thư viện PyTorch khổng lồ, bạn có thể chuyển đổi file trọng số `.pth` sang định dạng `.onnx` bằng lệnh:
```bash
python src/inference/export_onnx.py --model-name efficientnet --pth-path models/efficientnet_best.pth --onnx-path models/efficientnet_best.onnx
```

---

## Bước 5: Khởi chạy Giao diện (Production App)

Để trải nghiệm sản phẩm cuối cùng (dự đoán kèm biểu đồ giải thích AI Grad-CAM), bạn mở song song 2 Terminal:

**Terminal 1 (AI Backend):**
Đứng ở thư mục gốc, bật `venv` và chạy:
```bash
uvicorn app.backend.main:app --reload --host 127.0.0.1 --port 8000
```
*(Giao diện xem API Swagger: http://127.0.0.1:8000/docs)*

**Terminal 2 (Web Frontend):**
```bash
cd app/frontend
npm run dev
```
*(Truy cập Web người dùng tại: http://localhost:3000)*

---

## 🚀 Lựa chọn khác: Chạy siêu tốc bằng Docker (Cho triển khai nhanh)

Nếu bạn là kỹ sư muốn test nhanh sản phẩm mà không muốn tự chạy từng bước 2, 3, 4 ở trên (với điều kiện máy đã cài Docker Desktop), chỉ cần mở 1 Terminal ở gốc dự án:
```bash
docker-compose up -d --build
```
Hệ thống sẽ tự động cấu hình Backend + Frontend và bạn chỉ việc vào Web: `http://localhost:3000`.
