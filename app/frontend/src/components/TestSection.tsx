"use client";

import { useRef, useState } from "react";
import { CLASSES, type ClassKey } from "@/lib/classes";

interface Pred {
  key: ClassKey;
  p: number;
}

export default function TestSection() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [preds, setPreds] = useState<Pred[] | null>(null);
  const [heatmap, setHeatmap] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [drag, setDrag] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  function handleFile(f: File) {
    if (!f.type.startsWith("image/")) return;
    setFile(f);
    const reader = new FileReader();
    reader.onload = (ev) => {
      setPreview(ev.target?.result as string);
      setPreds(null);
      setHeatmap(null);
      setError(null);
    };
    reader.readAsDataURL(f);
  }

  async function analyze() {
    if (!file) return;
    setAnalyzing(true);
    setPreds(null);
    setHeatmap(null);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      // Gọi đến FastAPI backend (đảm bảo backend đang chạy trên port 8000)
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000/predict";
      const res = await fetch(apiUrl, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(`API Error: ${res.status} - ${text}`);
      }

      const data = await res.json();
      
      const parsedPreds: Pred[] = [];
      for (const [k, v] of Object.entries(data.probabilities)) {
        parsedPreds.push({ key: k.toLowerCase() as ClassKey, p: v as number });
      }
      parsedPreds.sort((a, b) => b.p - a.p);
      
      setPreds(parsedPreds);
      if (data.heatmap_base64) {
        setHeatmap(`data:image/jpeg;base64,${data.heatmap_base64}`);
      }
    } catch (err: any) {
      setError(err.message || "Không thể kết nối đến backend. Đảm bảo uvicorn đang chạy trên cổng 8000.");
    } finally {
      setAnalyzing(false);
    }
  }

  const top = preds?.[0];
  const topClass = top ? CLASSES[top.key] : null;

  return (
    <section id="test" className="border-t border-line bg-white">
      <div className="mx-auto max-w-[1180px] px-6 py-14">
        <div className="mb-2 text-[0.8rem] font-bold uppercase tracking-[0.08em] text-primary">
          Phân tích AI
        </div>
        <h2 className="mb-4 text-[1.9rem] font-extrabold">
          Chẩn đoán ảnh MRI với BrainTumorAI
        </h2>
        <p className="mb-6 max-w-[780px] text-[1.05rem] text-muted">
          Tải lên một ảnh MRI sọ não (JPG/PNG). Hệ thống sẽ gửi ảnh lên FastAPI backend 
          để xử lý qua mô hình EfficientNetB0 và sinh ra bản đồ giải thích (Grad-CAM).
        </p>

        <div className="grid items-start gap-7 md:grid-cols-2">
          {/* cột trái: upload */}
          <div>
            <div
              onClick={() => fileInput.current?.click()}
              onDragOver={(e) => {
                e.preventDefault();
                setDrag(true);
              }}
              onDragLeave={() => setDrag(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDrag(false);
                if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
              }}
              className={`cursor-pointer rounded-[14px] border-2 border-dashed px-6 py-10 text-center transition ${
                drag
                  ? "border-accent bg-[#e8f2fc]"
                  : "border-[#b9cfe0] bg-[#f8fbfd] hover:border-primary hover:bg-[#eef6fb]"
              }`}
            >
              <div className="mb-1 text-[2.4rem]">🖼️</div>
              <p>
                <b>Kéo &amp; thả</b> ảnh MRI vào đây
                <br />
                hoặc bấm để chọn tệp
              </p>
              <p className="mt-1 text-[0.85rem] text-muted">
                Hỗ trợ JPG, JPEG, PNG
              </p>
            </div>
            <input
              ref={fileInput}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => {
                if (e.target.files?.length) handleFile(e.target.files[0]);
              }}
            />

            {preview && (
              <>
                <img
                  src={preview}
                  alt="Ảnh MRI xem trước"
                  className="mt-4 w-full rounded-xl border border-line"
                />
                <button
                  onClick={analyze}
                  disabled={analyzing}
                  className="mt-4 w-full rounded-lg bg-primary px-5 py-2.5 font-semibold text-white transition-colors hover:bg-primary-dark disabled:opacity-60"
                >
                  {analyzing ? "Đang phân tích…" : "Phân tích"}
                </button>
              </>
            )}
          </div>

          {/* cột phải: kết quả */}
          <div>
            <div className="card-shadow min-h-[120px] rounded-[14px] border border-line bg-card p-6">
              {!preds && !analyzing && !error && (
                <p className="text-muted">
                  Kết quả phân tích sẽ hiển thị tại đây sau khi bạn tải ảnh và bấm{" "}
                  <b>Phân tích</b>.
                </p>
              )}
              {analyzing && <p className="text-muted">⏳ Đang xử lý trên Server (FastAPI)…</p>}
              
              {error && (
                <div className="rounded-[9px] bg-[#fee2e2] px-4 py-3 text-[0.9rem] text-[#991b1b]">
                  <b>Lỗi kết nối:</b> <br/> {error}
                </div>
              )}

              {preds && top && topClass && (
                <>
                  <div
                    className="mb-4 rounded-[14px] p-6 text-center text-white"
                    style={{ background: topClass.color }}
                  >
                    <div className="text-[0.9rem] opacity-90">
                      Kết quả dự đoán
                    </div>
                    <div className="text-2xl font-extrabold">{topClass.vi}</div>
                    <div className="text-[1.05rem] opacity-95">
                      Độ tin cậy: {(top.p * 100).toFixed(1)}%
                    </div>
                  </div>

                  <b>Xác suất theo từng nhóm</b>
                  {preds.map((o) => (
                    <div key={o.key} className="my-2.5">
                      <div className="mb-1 flex justify-between text-[0.9rem]">
                         <span>{CLASSES[o.key]?.vi || o.key}</span>
                        <span>{(o.p * 100).toFixed(1)}%</span>
                      </div>
                      <div className="h-[9px] overflow-hidden rounded-md bg-[#eef2f6]">
                        <div
                          className="h-full rounded-md transition-[width] duration-500"
                          style={{
                            width: `${(o.p * 100).toFixed(1)}%`,
                            background: CLASSES[o.key]?.color || '#cbd5e1',
                          }}
                        />
                      </div>
                    </div>
                  ))}
                  
                  {heatmap && (
                    <div className="mt-6">
                      <b className="mb-2 block">Bản đồ giải thích Grad-CAM</b>
                      <img 
                        src={heatmap} 
                        alt="Grad-CAM heatmap" 
                        className="w-full rounded-xl border border-line shadow-sm"
                      />
                      <p className="mt-2 text-[0.8rem] text-muted">
                        Vùng màu nóng biểu thị khu vực mà mô hình AI tập trung phân tích để đưa ra quyết định dự đoán này.
                      </p>
                    </div>
                  )}

                  {topClass.isTumor ? (
                    <div className="mt-4 rounded-[9px] bg-[#fdf0e1] px-4 py-2.5 text-[0.9rem] text-[#9a6213]">
                      ⚠️ Mô hình nghi ngờ có khối u. Hãy tham khảo ý kiến bác sĩ chuyên khoa.
                    </div>
                  ) : (
                    <div className="mt-4 rounded-[9px] bg-[#e7f6ee] px-4 py-2.5 text-[0.9rem] text-[#1e7d52]">
                      ✅ Mô hình không phát hiện khối u trong ảnh này.
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        </div>

        <div className="mt-8 rounded-xl border border-[#f3e2b3] bg-[#fff8e6] px-5 py-4 text-[0.92rem] text-[#7a5a13]">
          ⚠️ <b>Tuyên bố miễn trừ:</b> Công cụ mang tính học thuật và hỗ trợ sàng
          lọc, <b>không thay thế chẩn đoán của bác sĩ chuyên khoa</b>. Mọi quyết
          định lâm sàng cần dựa trên thăm khám và ý kiến chuyên môn.
        </div>
      </div>
    </section>
  );
}
