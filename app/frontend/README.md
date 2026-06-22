# NeuroScan — Web demo (Next.js)

Bản web demo cho đồ án **Phân loại khối u não trên ảnh MRI** (4 lớp: glioma,
meningioma, notumor, pituitary · EfficientNetB0). Dựng bằng Next.js (App Router)
+ Tailwind CSS, port từ trang `web/index.html`.

> Hiện tại phần **dự đoán là DEMO (mock)** — giao diện + luồng upload đã hoàn
> chỉnh, kết quả là dữ liệu mẫu. Sẽ cắm mô hình thật sau khi train xong `.keras`.

## Chạy local

```bash
npm install      # nếu chưa cài
npm run dev      # http://localhost:3000
```

Build production: `npm run build && npm start`.

## Cấu trúc

```
src/
├── app/
│   ├── layout.tsx      # font Inter, metadata, lang vi
│   ├── page.tsx        # ghép các section
│   └── globals.css     # theme màu (Tailwind v4 @theme)
├── components/
│   ├── Header.tsx  Hero.tsx  About.tsx  CausesTreatment.tsx
│   ├── Types.tsx       # 4 nhóm phân loại
│   ├── TestSection.tsx # upload + dự đoán (client component)
│   └── Footer.tsx
└── lib/classes.ts      # dữ liệu 4 lớp (tên VI, màu, mô tả)
```

## Cắm mô hình thật (bước sau)

1. Train notebook `BrainTumor_MRI_Classification.ipynb` → lưu `.keras`.
2. Convert sang TensorFlow.js (xem `../web/export_tfjs.py`):
   ```bash
   python ../web/export_tfjs.py --model ../brain_tumor_efficientnetb0.keras --out model
   ```
   Đặt thư mục `model/` vào `public/model/` (URL: `/model/model.json`).
3. Cài tfjs: `npm install @tensorflow/tfjs`.
4. Trong `src/components/TestSection.tsx`, thay hàm `mockPredict()` bằng inference
   thật: `tf.loadLayersModel("/model/model.json")` → `tf.browser.fromPixels(img)`
   → `resizeBilinear([224,224])` → `toFloat()` → `expandDims(0)` →
   `model.predict(t)`. Giữ thang **[0,255]** (EfficientNet tự normalize bên trong),
   không rescale thủ công.
