# TÀI LIỆU ĐẶC TẢ YÊU CẦU PHẦN MỀM (SOFTWARE REQUIREMENTS SPECIFICATION)
## Dự án: Hệ thống phát hiện, khoanh vùng và giải thích khối u não trên ảnh MRI
---

## 1. Giới Thiệu Chung (Introduction)

### 1.1. Mục tiêu dự án
Xây dựng một hệ thống phần mềm ứng dụng học sâu (Deep Learning) để tự động phát hiện, phân vùng và giải thích cấu trúc khối u não trên ảnh chụp cộng hưởng từ (MRI) 3D. Hệ thống hướng đến việc hỗ trợ các bác sĩ chẩn đoán hình ảnh rút ngắn thời gian phân tích, nâng cao độ chính xác và tăng cường tính minh bạch của các quyết định hỗ trợ bởi AI trong môi trường y tế lâm sàng.

### 1.2. Đối tượng sử dụng chính
* **Bác sĩ chẩn đoán hình ảnh (Radiologist):** Người trực tiếp xem ảnh, kiểm tra kết quả phân vùng tự động, chỉnh sửa mask và phê duyệt báo cáo lâm sàng.
* **Kỹ sư AI / Lập trình viên y khoa (Medical AI Engineer):** Người vận hành, đánh giá độ lệch dữ liệu (Data Drift) và cập nhật mô hình định kỳ.

---

## 2. Đặc Tả Yêu Cầu về Dữ Liệu (Data-Centric Requirements)

### 2.1. Nguồn dữ liệu
* **Tập dữ liệu huấn luyện công khai:** Sử dụng tập dữ liệu BraTS (Brain Tumor Segmentation Challenge) từ Medical Segmentation Decathlon (MSD Task01_BrainTumour) chứa 484 ca chụp MRI 3D đa chuỗi.
* **Tập dữ liệu lâm sàng cục bộ:** Hỗ trợ các ảnh quét định dạng DICOM và NIfTI (.nii, .nii.gz) trực tiếp từ máy chụp MRI bệnh viện.

### 2.2. Định dạng chuỗi xung đầu vào (Modalities)
Mô hình yêu cầu 4 chuỗi xung MRI cơ bản để đạt hiệu suất tối ưu:
1. **FLAIR (Fluid-Attenuated Inversion Recovery):** Xác định vùng phù nề quanh u (edema).
2. **T1-weighted (T1):** Ảnh cấu trúc giải phẫu cơ bản.
3. **T1-contrast enhanced (T1c):** Làm nổi bật lõi u hoạt động (active tumor core).
4. **T2-weighted (T2):** Xác định ranh giới tổn thương tổng thể.

### 2.3. Pipeline tiền xử lý dữ liệu (Preprocessing Pipeline)
Mỗi mẫu dữ liệu đi qua pipeline xử lý tự động trước khi nạp vào mô hình:
* **Đồng bộ không gian (Spatial Registration):** Đưa ảnh về cùng độ phân giải voxel ($1.5 \times 1.5 \times 2.0\text{ mm}$) và hướng không gian RAS (Right-Anterior-Superior).
* **Loại bỏ xương sọ (Skull Stripping):** Tách biệt nhu mô não khỏi các mô ngoài sọ (xương sọ, da đầu, mắt) bằng thuật toán ngưỡng hóa Otsu động kết hợp lấp đầy lỗ trống (morphological hole filling).
* **Chuẩn hóa cường độ tín hiệu (Intensity Normalization):** Khử nhiễu thiết bị chụp bằng phương pháp chuẩn hóa Z-score trên vùng nhu mô não đã lọc.

### 2.4. Xử lý mất cân bằng nhãn (Imbalance Handling)
Do thể tích khối u nhỏ hơn rất nhiều so với thể tích toàn bộ não (thường chiếm dưới 1% tổng số voxel), hệ thống áp dụng:
* Kỹ thuật lấy mẫu tập trung vùng biên tổn thương (RandCropByPosNegLabel).
* Hàm mất mát lai **DiceFocalLoss** để cân bằng trọng số giữa các voxel nền (background) và voxel khối u.

---

## 3. Đặc Tả Yêu Cầu Chức Năng (Functional Requirements)

### 3.1. Phát hiện khối u (Tumor Detection)
* **Mô tả:** Nhận diện sự hiện diện của khối u não trên ảnh MRI.
* **Đầu vào:** Chuỗi xung MRI 3D.
* **Xử lý:** Mô hình phân loại 3D DenseNet121 trích xuất đặc trưng không gian 3D.
* **Đầu ra:** Kết quả phân loại nhị phân (Có u / Không u) kèm theo độ tự tin (Confidence score) dạng phần trăm (%).

### 3.2. Khoanh vùng khối u (Tumor Segmentation)
* **Mô tả:** Phân định chính xác ranh giới của các phân vùng khối u.
* **Xử lý:** Sử dụng kiến trúc U-Net 3D (Baseline) và Swin UNETR (Transformer) chạy suy luận cửa sổ trượt (Sliding Window Inference).
* **Đầu ra:** Bản đồ phân vùng đa lớp (Segmentation Mask) gồm các cấu trúc:
  - Lõi hoại tử (Necrotic core - Nhãn 1)
  - Phù nề quanh u (Edema - Nhãn 2)
  - Khối u hoạt động bắt thuốc (Enhancing tumor - Nhãn 3)

### 3.3. Giải thích mô hình (Explainability / XAI)
Nhằm xây dựng lòng tin lâm sàng cho bác sĩ, hệ thống cung cấp:
* **3D Grad-CAM:** Hiển thị bản đồ nhiệt (Heatmap) thể hiện khu vực mô hình tập trung đưa ra quyết định phân vùng.
* **Attention Maps:** Trích xuất các ma trận tự chú ý (Self-Attention) từ khối Transformer của Swin UNETR.
* **SHAP values:** Định lượng mức độ ảnh hưởng của các lát cắt ảnh đối với kết quả dự đoán.

### 4.4. Tương tác lâm sàng (Human-in-the-Loop)
Cho phép bác sĩ trực tiếp can thiệp và hiệu chỉnh kết quả của AI trên giao diện web:
* **Tinh chỉnh ngưỡng xác suất (Probability Cutoff):** Thay đổi độ nhạy của thuật toán phân vùng bằng thanh trượt (slider từ 0.1 đến 0.9).
* **Hiệu chỉnh hình thái học (Morphological Refinement):** Áp dụng các bộ lọc giãn biên (dilation) hoặc co biên (erosion) mask để làm mịn hoặc hiệu chỉnh ranh giới u theo kinh nghiệm y khoa.

### 4.5. Xuất báo cáo lâm sàng (Clinical Reporting)
* Tự động tính toán thể tích tuyệt đối của khối u (đơn vị $cm^3$) dựa trên số lượng voxel hoạt động và voxel spacing của file MRI gốc.
* Xác định tọa độ trọng tâm vật lý (Centroid) của khối u trong không gian 3D thực tế (đơn vị mm).
* Sinh báo cáo dưới định dạng văn bản có cấu trúc (Markdown/TXT) kèm theo chữ ký disclaimer y tế bắt buộc.

---

## 4. Đặc Tả Yêu Cầu Kỹ Thuật (Technical Requirements)

### 4.1. Công nghệ lõi và hệ sinh thái
* **Ngôn ngữ:** Python 3.10
* **Framework học sâu:** PyTorch kết hợp MONAI (Medical Open Network for AI) phục vụ các phép biến đổi ảnh 3D và kiến trúc mạng y sinh.
* **Thư viện xử lý ảnh:** `nibabel` (đọc tệp NIfTI), `pydicom` (đọc tệp DICOM), `scipy`, `scikit-image` (xử lý hình thái học).

### 4.2. Các chỉ số đánh giá chất lượng (Metrics)
* **Dice Similarity Coefficient (DSC):** Đạt tối thiểu 0.85 trên tập kiểm chứng BraTS đối với toàn bộ khối u (Whole Tumor).
* **Hausdorff Distance 95 (HD95):** Đo sai số ranh giới u, mục tiêu dưới 5.0 mm.
* **Sensitivity & Specificity:** Đo độ nhạy và độ đặc hiệu của bộ phát hiện u, mục tiêu đạt trên 90%.

### 4.3. Hiệu năng hệ sinh thái (Performance)
* **Inference Time:** Thời gian chạy suy luận cho 1 ca chụp MRI 3D kích thước $240 \times 240 \times 155$ không quá 10 giây trên GPU Nvidia RTX 3060 trở lên và không quá 60 giây trên môi trường CPU.

---

## 5. Yêu Cầu Hệ Thống & Giao Diện UI/UX (System & UI/UX Requirements)

### 5.1. Bảng điều khiển Streamlit (Dashboard Web App)
* **Trình trực quan hóa:** Hiển thị lát cắt ảnh MRI xám gốc kế bên ảnh overlay mask màu đỏ. Hỗ trợ cuộn lát cắt linh hoạt.
* **Điều chỉnh độ mờ (Opacity slider):** Cho phép kéo thanh trượt điều khiển độ đậm/nhạt của mask màu đỏ từ 0.0 (trong suốt) đến 1.0 (hiển thị rõ nét).
* **Đa hướng chiếu (Planes):** Hỗ trợ xem lát cắt trên cả 3 mặt giải phẫu học: Axial, Sagittal, Coronal.

### 5.2. Khả năng tích hợp PACS/DICOM
* **Giao thức kết nối:** Hỗ trợ các chuẩn truyền thông mạng DICOM (DIMSE services).
* **Dịch vụ lắng nghe (DICOM Listener):** Lắng nghe yêu cầu đẩy ảnh từ máy trạm lâm sàng thông qua dịch vụ `C-STORE` trên cổng TCP 104 với AET là `AI_LISTENER`.
* **PACS Simulator:** Sử dụng phần mềm Orthanc PACS làm kho lưu trữ ảnh trung tâm, giao tiếp qua REST API (cổng 8042).

### 5.3. Yêu cầu đóng gói (Deployment)
* Hệ thống được đóng gói hoàn toàn bằng Docker sử dụng multi-container (Docker Compose):
  - Dịch vụ API Backend (FastAPI, uvicorn)
  - Giao diện Dashboard (Streamlit)
  - Dịch vụ PACS (Orthanc DICOM Server)

---

## 6. Yêu Cầu Bảo Mật & Đạo Đức (Security & Ethics)

### 6.1. Bảo mật dữ liệu y khoa (HIPAA / Nghị định 13)
* **Khử định danh (Anonymization):** API suy luận và DICOM Listener phải hỗ trợ xóa bỏ hoặc mã hóa các trường thông tin cá nhân nhạy cảm trong file DICOM (như Tên bệnh nhân, Ngày sinh, ID bệnh viện) trước khi lưu trữ vào thư mục dữ liệu thô.
* **Môi trường nội bộ (On-premises deployment):** Toàn bộ hệ thống chạy độc lập trong mạng nội bộ của bệnh viện, không gửi dữ liệu y tế ra máy chủ đám mây ngoài trừ khi có sự đồng ý của bệnh nhân.

### 6.2. Cảnh báo miễn trừ trách nhiệm (Clinical Disclaimer)
Mọi trang giao diện và báo cáo xuất ra từ hệ thống phải hiển thị rõ ràng thông báo cảnh báo:
> *"Hệ thống hỗ trợ chẩn đoán AI chỉ đóng vai trò tham khảo và hỗ trợ kỹ thuật. Quyết định chẩn đoán lâm sàng và phương án điều trị cuối cùng thuộc về bác sĩ chuyên khoa có chứng chỉ hành nghề."*
