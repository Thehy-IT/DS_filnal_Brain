# Hệ Thống Phân Tích U Não MRI Lai 2D/3D & Khoanh Vùng Giải Thích (MONAI & PyTorch)

Hệ thống y tế sử dụng Trí tuệ Nhân tạo (AI) và Học sâu (Deep Learning) để tự động phát hiện, phân loại (classification) đa lớp khối u não, trích xuất bản đồ giải thích y khoa (Grad-CAM) kết hợp ước lượng Độ bất định lâm sàng (Clinical Uncertainty Map) và phân vùng tự động (Segmentation) bằng U-Net trực tiếp trên ảnh chụp MRI 2D (JPG/PNG/DICOM) hoặc thể tích ảnh 3D (NIfTI/DICOM).

Hệ thống được phát triển dựa trên hệ sinh thái **MONAI** và **PyTorch**, hỗ trợ tích hợp cổng PACS lưu trữ ảnh bệnh viện và đóng gói Docker chạy đa nền tảng.

---

## 🚀 Các Tính Năng Nổi Bật & Kiến Trúc Lai

### 1. Kiến trúc Lai 2D/3D & Hệ Thống Song Song 2 Model (Dual-Model Pipeline)
Hệ thống kết hợp sức mạnh của hai mô hình học sâu chuyên biệt để mang lại hiệu quả chẩn đoán cao nhất:
* **Mô hình Phân Loại (Classifier - DenseNet121):** Được huấn luyện trên tập dữ liệu 2D JPG/PNG (`data/Training`, `data/Testing`) để nhận diện 4 nhóm bệnh lý:
  * `glioma` (u tế bào thần kinh đệm)
  * `meningioma` (u màng não)
  * `pituitary` (u tuyến yên)
  * `notumor` (không có u)
  * Mô hình đạt độ chính xác cao và xuất ra xác suất tin cậy của chẩn đoán.
* **Mô hình Phân Vùng (Segmenter - MONAI U-Net):** Mô hình U-Net 2D được huấn luyện bằng cách sử dụng **học giám sát yếu (weakly supervised learning)** trên tập nhãn giả (pseudo masks) sinh ra từ Grad-CAM của mô hình phân loại. Phục vụ khoanh vùng chính xác biên giới hạn khối u.
* **Xử lý Thể tích 3D (3D Slicing Engine):** Khi bác sĩ tải lên tệp thể tích ảnh y khoa 3D NIfTI (`.nii`/`.nii.gz`) hoặc thư mục DICOM chuẩn PACS, hệ thống tự động điều chỉnh khoảng cách điểm ảnh (Spacing) về `1.0 mm³` và căn chỉnh hướng giải phẫu chuẩn `RAS`. Trích xuất các chuỗi lát cắt giải phẫu theo 3 mặt phẳng chính: **Axial (Mặt cắt ngang)**, **Coronal (Mặt cắt dọc đứng trước-sau)**, và **Sagittal (Mặt cắt dọc đứng bên)** để tiến hành suy luận lát cắt (slice-by-slice).

### 2. Sửa lỗi Domain Shift & Lọc Sọ (Otsu Skull Stripping)
* **Khắc phục Domain Shift cho U-Net:** U-Net được huấn luyện tối ưu hóa với giá trị pixel chuẩn hóa Min-Max về đoạn `[0, 1]`. Hệ thống triển khai hàm tiền xử lý riêng biệt `preprocess_slice_for_unet` song song với chuẩn hóa Z-score của Classifier, giúp triệt tiêu hoàn toàn hiện tượng nhiễu viền sọ và loại bỏ lỗi khoanh vùng ảo ở 4 góc tối ngoài sọ.
* **Loại bỏ xương sọ (Skull Stripping):** Áp dụng thuật toán Otsu thresholding kết hợp lấp lỗ đa chiều (binary fill holes) và Distance Transform của scipy để tạo mặt nạ sọ não (Brain Mask), nhân triệt tiêu toàn bộ tín hiệu gây nhiễu ngoài sọ.

### 3. Bản Đồ Độ Bất Định Lâm Sàng (Clinical Uncertainty Map)
* Áp dụng kỹ thuật **Monte Carlo Dropout** chạy suy luận ngẫu nhiên nhiều lần (mặc định 5 passes) trên lát cắt để tính toán độ lệch chuẩn (Std) và sai số entropy vùng biên khối u.
* Giúp bác sĩ lâm sàng phát hiện và cảnh báo những vùng biên khối u mà AI chẩn đoán có độ bất định cao, hỗ trợ quá trình hội chẩn an toàn hơn.

### 4. Giao Diện Chẩn Đoán Streamlit Localized
* **Trực quan hóa song song 3 panel (Giao diện Premium):**
  1. **Ảnh MRI Gốc + Nhãn Mask đỏ:** Khoanh vùng vị trí u (cho phép tùy chỉnh độ mờ nhãn - Mask Opacity).
  2. **Bản đồ nhiệt Grad-CAM (XAI):** Minh họa vùng tập trung đặc trưng sâu của mạng thần kinh.
  3. **Bản đồ độ bất định (Clinical Uncertainty):** Chỉ ra các khu vực AI nghi ngờ/không chắc chắn.
* **Hiệu chỉnh hình thái học (Morphological Processing):** Tích hợp công cụ co biên (erosion) hoặc giãn biên (dilation) trực tiếp trên giao diện Streamlit với bán kính điều chỉnh từ 1-5 pixels giúp bác sĩ tinh chỉnh nhãn u theo ý muốn.
* **Báo cáo lâm sàng định lượng (Clinical Report):** Tự động dịch nghĩa tiếng Việt các lớp u, tính toán diện tích khối u ($cm^2$), tọa độ centroid của u trên lát cắt, hiển thị cảnh báo y tế dựa trên mức độ tin cậy của AI.

---

## 📁 Cấu Trúc Thư Mục Dự Án

```text
e:/Brain_tnqtr/
├── requirements.txt                # Danh sách thư viện y tế và dependencies cài đặt
├── README.md                       # Tài liệu hướng dẫn sử dụng hệ thống (File này)
├── dashboard/                      # Giao diện chẩn đoán Streamlit Dashboard
│   ├── app.py                      # File khởi chạy chính Streamlit Dashboard
│   ├── utils.py                    # Nạp ảnh 2D, trích xuất DICOM, co/giãn biên mask hình thái học
│   └── components/                 # Các widgets giao diện thành phần
│       ├── viewer.py               # Bộ hiển thị 3 panel song song (Gốc + Mask, Grad-CAM, Uncertainty)
│       └── report.py               # Sinh và xuất báo cáo lâm sàng định lượng bằng tiếng Việt
├── data/                           # Dữ liệu ảnh MRI và trọng số mô hình
│   ├── Training/                   # Ảnh MRI 2D gốc dùng huấn luyện phân loại (glioma, meningioma, pituitary, notumor)
│   ├── Testing/                    # Ảnh MRI 2D gốc dùng kiểm thử phân loại
│   ├── processed/                  # Dữ liệu ảnh và nhãn giả đã xử lý để huấn luyện U-Net
│   │   └── segmentation/
│   │       ├── train/              # Thư mục chứa images/ và masks/ cho tập train U-Net
│   │       └── val/                # Thư mục chứa images/ và masks/ cho tập validation U-Net
│   ├── reference/                  # Checkpoint mô hình tốt nhất (.pth)
│   │   ├── best_metric_model.pth   # Trọng số mô hình phân loại DenseNet121
│   │   └── best_unet_model.pth     # Trọng số mô hình phân vùng MONAI U-Net
│   └── synthetic_volume.nii.gz     # Tệp thể tích 3D giả lập phục vụ kiểm thử chế độ NIfTI
├── deploy/                         # Đóng gói Docker & Cấu hình PACS
│   ├── docker/                     # Dockerfiles cho API server và Streamlit Dashboard
│   │   ├── Dockerfile.api
│   │   └── Dockerfile.dashboard
│   ├── docker-compose.yml          # Điều phối Docker Containers (API, Dashboard, Orthanc PACS)
│   └── pacs/                       # Mô phỏng PACS DICOM Server (Orthanc)
│       ├── orthanc.json            # Cấu hình máy chủ Orthanc PACS
│       └── dicom_listener.py       # Bộ lắng nghe sự kiện nhận DICOM C-STORE
├── docs/                           # Tài liệu đặc tả hệ thống
├── scripts/                        # Các kịch bản huấn luyện và xử lý dữ liệu
│   ├── generate_pseudo_masks.py    # Sinh nhãn giả (pseudo masks) từ mô hình Classifier + Grad-CAM
│   ├── train_model.py              # Huấn luyện mô hình DenseNet121 2D trên ảnh JPG
│   ├── train_unet.py               # Huấn luyện mô hình MONAI U-Net 2D trên nhãn giả
│   ├── convert_to_nnunet.py        # Kịch bản chuyển đổi định dạng chuẩn nnU-Net (nếu cần mở rộng)
│   ├── download_brats.py           # Kịch bản tải tập dữ liệu BraTS
│   └── evaluate_clinical_metrics.py# Đánh giá chỉ số chất lượng mô hình
├── src/                            # Mã nguồn lõi (AI Core Pipeline)
│   ├── core/                       # Cấu hình tham số hệ thống và ánh xạ nhãn
│   ├── data/                       # Dataset, custom transforms, bộ cắt lát slicer.py
│   ├── models/                     # Kiến trúc mạng 2D/3D (Classifier, UNet, Swin UNETR)
│   ├── training/                   # Tổ hợp cấu hình trainer, losses, metrics
│   └── inference/                  # Engine suy luận y khoa predictor.py & FastAPI Service api.py
└── tests/                          # Bộ kiểm thử tự động (pytest)
```

---

## 🛠️ Hướng Dẫn Vận Hành & Huấn Luyện Hệ Thống

### 1. Vận Hành Môi Trường Local (Dành cho nhà phát triển)

#### Bước 1: Cài đặt môi trường ảo & Thư viện
Khởi tạo môi trường ảo Python và tiến hành cài đặt các thư viện cần thiết:
```bash
python -m venv venv
venv\Scripts\activate      # Trên Windows (cmd)
# Hoặc: venv\Scripts\Activate.ps1 (Trên PowerShell)
pip install -r requirements.txt
```

#### Bước 2: Chạy kiểm thử tự động
Xác thực tất cả các tính năng tiền xử lý, mô hình học máy, các module suy luận và cắt lát 3D hoạt động chính xác trước khi khởi động ứng dụng:
```bash
pytest
```

#### Bước 3: Huấn luyện Mô hình từ đầu (Tùy chọn)
Nếu bạn muốn huấn luyện lại hệ thống:
1. **Huấn luyện Classifier:**
   ```bash
   python scripts/train_model.py
   ```
   Trọng số mô hình phân loại tốt nhất sẽ được lưu tại `data/reference/best_metric_model.pth`.
2. **Sinh Nhãn Giả (Pseudo Masks):**
   Sử dụng mô hình phân loại đã được huấn luyện để tự động sinh ra các nhãn phân vùng từ Grad-CAM kết hợp lọc sọ và căn chỉnh khoảng cách:
   ```bash
   python scripts/generate_pseudo_masks.py
   ```
   Ảnh và nhãn giả được sinh ra và chia vào thư mục `data/processed/segmentation/train/` và `data/processed/segmentation/val/`.
3. **Huấn luyện U-Net:**
   Chạy kịch bản huấn luyện U-Net 2D dựa trên nhãn giả vừa sinh ra:
   ```bash
   python scripts/train_unet.py
   ```
   Mô hình tốt nhất với Dice Score tối ưu sẽ được lưu tại `data/reference/best_unet_model.pth`.

#### Bước 4: Khởi chạy Streamlit Dashboard
Chạy trực tiếp dashboard trên máy local:
```bash
streamlit run dashboard/app.py
```
* **Chế độ kiểm thử 3D NIfTI nhanh:** Truy cập địa chỉ `http://localhost:8501/?test_nifti=true` để tự động tải tệp 3D NIfTI giả lập được sinh sẵn trong dự án.
* **Chế độ ảnh thông thường:** Truy cập `http://localhost:8501`, thực hiện tải lên một hoặc nhiều tệp ảnh `.jpg`, `.png`, hoặc `.dcm` trong thư mục `data/Testing/` để bắt đầu phân tích.

---

### 2. Triển Khai Bằng Docker (Dành cho môi trường kiểm thử/Lâm sàng)

Hệ thống được đóng gói hoàn chỉnh bằng Docker Compose, tự động khởi tạo và kết nối 3 dịch vụ đồng thời (**FastAPI inference API**, **Streamlit Clinical Dashboard**, và **Orthanc PACS Server**).

#### Bước 1: Di chuyển vào thư mục deploy
```bash
cd deploy
```

#### Bước 2: Khởi chạy các container bằng Docker Compose
Lệnh này sẽ tự động build image cho API và Dashboard từ mã nguồn mới nhất:
```bash
docker compose up --build
```

#### Bước 3: Truy cập và sử dụng dịch vụ
Sau khi các container khởi động thành công, bạn có thể truy cập các địa chỉ sau:
* **Giao diện Dashboard bác sĩ:** Truy cập `http://localhost:8501/?test_nifti=true` hoặc `http://localhost:8501`.
* **Tài liệu API tương tác (Swagger UI):** Truy cập `http://localhost:8000/docs` để thử nghiệm trực tiếp các API phân tích ảnh y tế.
* **Máy chủ quản trị Orthanc PACS:** Truy cập `http://localhost:8042`.
  * *Tài khoản mặc định:* `orthanc` / `orthanc`.
  * *DICOM Listener:* Cổng DICOM C-STORE lắng nghe tại cổng `4242` để nhận ảnh từ các thiết bị chụp quét hoặc trạm làm việc khác.

#### Bước 4: Dừng cụm dịch vụ
Nhấn tổ hợp phím `Ctrl + C` và dọn dẹp các tài nguyên container bằng lệnh:
```bash
docker compose down
```

---

## 🩺 Hướng dẫn sử dụng cho Bác sĩ Lâm sàng

1. **Tải lên dữ liệu:** Kéo và thả một hoặc nhiều tệp tin ảnh MRI (NIfTI `.nii`/`.nii.gz`, DICOM `.dcm`, hoặc JPG/PNG thông thường) vào khu vực tải lên ở thanh điều hướng bên trái.
2. **Chọn ảnh cần phân tích:** Nếu tải lên nhiều ảnh cùng lúc, sử dụng hộp chọn **"Chọn tập tin phân tích"** để chuyển đổi giữa các ca bệnh.
3. **Xem lát cắt đối với ảnh 3D:** Sử dụng thanh trượt **"Lát Cắt (Slice)"** và menu chọn **"Mặt Cắt Giải Phẫu"** (Axial, Coronal, Sagittal) để khảo sát thể tích khối u theo mọi hướng.
4. **Tùy chỉnh chẩn đoán:**
   * Thay đổi **"Độ Mờ Nhãn U (Mask Opacity)"** để nhìn rõ vị trí giải phẫu bên dưới lớp nhãn màu đỏ.
   * Điều chỉnh **"Ngưỡng Phân Vùng U-Net"** (hoặc độ nhạy Grad-CAM) để thay đổi vùng chọn khối u của AI.
   * Sử dụng chức năng **"Hiệu Chỉnh Hình Thái Học"** (`erosion` để thu nhỏ bớt viền nhiễu, hoặc `dilation` để bù đắp các vùng u bị khuyết thiếu) kết hợp với **"Bán Kính Hiệu Chỉnh"**.
5. **Xem Báo cáo:** Đọc báo cáo tự động ở góc phải để nắm thông tin về loại u chẩn đoán, độ tin cậy AI, diện tích bề mặt khối u ước lượng ($cm^2$), tọa độ centroid của khối u, và các khuyến cáo lâm sàng tương ứng.
