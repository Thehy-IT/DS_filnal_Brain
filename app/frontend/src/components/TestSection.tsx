"use client";

import { useRef, useState, useEffect } from "react";
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
  
  // Custom Comparison UI States
  const [viewMode, setViewMode] = useState<"sideBySide" | "slider">("sideBySide");
  const [sliderPos, setSliderPos] = useState<number>(50);
  const [modalImage, setModalImage] = useState<{ src: string; title: string } | null>(null);

  const fileInput = useRef<HTMLInputElement>(null);

  function handleFile(f: File) {
    if (!f.type.startsWith("image/")) {
      setError("Tệp đã chọn không phải định dạng hình ảnh hợp lệ (JPG/PNG).");
      return;
    }
    if (f.size > 10 * 1024 * 1024) {
      setError("Dung lượng tệp quá lớn. Vui lòng chọn tệp dưới 10MB.");
      return;
    }

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

      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "/api/predict";
      const res = await fetch(apiUrl, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(`API Error: ${res.status} - ${text}`);
      }

      const data = await res.json();
      if (!data || typeof data.probabilities !== "object") {
        throw new Error("Phản hồi từ máy chủ không đúng cấu trúc.");
      }
      
      const parsedPreds: Pred[] = [];
      for (const [k, v] of Object.entries(data.probabilities)) {
        parsedPreds.push({ key: k.toLowerCase() as ClassKey, p: v as number });
      }
      parsedPreds.sort((a, b) => b.p - a.p);
      
      setPreds(parsedPreds);
      if (data.heatmap_base64) {
        setHeatmap(`data:image/jpeg;base64,${data.heatmap_base64}`);
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg || "Không thể kết nối đến backend. Đảm bảo uvicorn đang chạy trên cổng 8000.");
    } finally {
      setAnalyzing(false);
    }
  }

  // Handle keyboard shortcut ESC and body scroll lock for modal
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") setModalImage(null);
    };

    if (modalImage) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "";
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      document.body.style.overflow = "";
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [modalImage]);

  const top = preds?.[0];
  const topClass = top ? CLASSES[top.key] : null;

  return (
    <section id="test" className="border-t border-line bg-white">
      <div className="mx-auto max-w-[1240px] px-4 sm:px-6 py-14">
        <div className="mb-2 text-[0.8rem] font-bold uppercase tracking-[0.08em] text-primary">
          Phân tích AI &amp; Giải thích trực quan
        </div>
        <h2 className="mb-3 text-2xl sm:text-[1.9rem] font-extrabold text-ink">
          Chẩn đoán ảnh MRI với BrainTumorAI
        </h2>
        <p className="mb-8 max-w-[820px] text-[1.02rem] text-muted leading-relaxed">
          Tải lên ảnh MRI sọ não (JPG/PNG). Hệ thống AI sẽ phân tích phân loại khối u 
          và tạo bản đồ nhiệt Grad-CAM trực quan giúp đối chiếu vị trí mô bệnh học nghi ngờ.
        </p>

        <div className="grid items-start gap-8 lg:grid-cols-12">
          {/* Cột trái: Upload & Preview ảnh gốc */}
          <div className="lg:col-span-5 flex flex-col gap-5">
            <div
              role="button"
              tabIndex={0}
              onClick={() => fileInput.current?.click()}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  fileInput.current?.click();
                }
              }}
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
              className={`cursor-pointer rounded-xl border-2 border-dashed p-6 text-center transition duration-200 outline-none focus-visible:ring-2 focus-visible:ring-primary ${
                drag
                  ? "border-accent bg-[#e8f2fc]"
                  : "border-[#b9cfe0] bg-[#f8fbfd] hover:border-primary hover:bg-[#eef6fb]"
              }`}
            >
              <div className="mx-auto mb-2 flex h-11 w-11 items-center justify-center rounded-full bg-primary/10 text-primary">
                <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
              </div>
              <p className="text-[0.95rem] font-medium text-ink">
                <b>Kéo &amp; thả ảnh MRI vào đây</b> hoặc <span className="font-semibold text-primary underline">chọn tệp từ máy</span>
              </p>
              <p className="mt-1 text-[0.78rem] text-muted">
                Hỗ trợ định dạng JPG, JPEG, PNG (Tối đa 10MB)
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

            {/* Hiển thị ảnh gốc xem trước với kích thước to rõ */}
            {preview && (
              <div className="rounded-xl border border-line bg-card p-4 card-shadow">
                <div className="mb-3 flex items-center justify-between">
                  <span className="text-[0.88rem] font-bold text-ink flex items-center gap-1.5">
                    <span className="h-2 w-2 rounded-full bg-primary"></span>
                    Ảnh MRI gốc tải lên
                  </span>
                  <button
                    onClick={() => setModalImage({ src: preview, title: "Ảnh MRI Gốc (Chất lượng cao)" })}
                    className="text-[0.78rem] font-medium text-primary hover:underline flex items-center gap-1"
                    type="button"
                  >
                    <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7" />
                    </svg>
                    Phóng to
                  </button>
                </div>

                <div 
                  role="button"
                  tabIndex={0}
                  onClick={() => setModalImage({ src: preview, title: "Ảnh MRI Gốc (Chất lượng cao)" })}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      setModalImage({ src: preview, title: "Ảnh MRI Gốc (Chất lượng cao)" });
                    }
                  }}
                  className="group relative overflow-hidden rounded-lg bg-slate-900 border border-slate-800 flex items-center justify-center min-h-[280px] max-h-[380px] cursor-pointer outline-none focus-visible:ring-2 focus-visible:ring-primary"
                >
                  <img
                    src={preview}
                    alt="Ảnh MRI xem trước"
                    className="w-full h-full max-h-[380px] object-contain transition-transform duration-300 group-hover:scale-105"
                  />
                  <div className="absolute inset-0 bg-black/30 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center text-white text-xs font-medium gap-1.5 backdrop-blur-[2px]">
                    <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                    </svg>
                    Bấm để xem chi tiết
                  </div>
                </div>

                <div className="mt-4 flex gap-3">
                  <button
                    onClick={() => fileInput.current?.click()}
                    type="button"
                    className="rounded-lg border border-line px-3.5 py-2.5 text-[0.85rem] font-medium text-muted transition hover:bg-neutral-100 hover:text-foreground"
                  >
                    Đổi ảnh khác
                  </button>
                  <button
                    onClick={analyze}
                    disabled={analyzing}
                    className="flex-1 rounded-lg bg-primary px-4 py-2.5 text-[0.92rem] font-semibold text-white shadow-sm transition hover:bg-primary-dark disabled:opacity-60 flex items-center justify-center gap-2"
                  >
                    {analyzing ? (
                      <>
                        <svg className="animate-spin h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        Đang phân tích AI…
                      </>
                    ) : (
                      "Bắt đầu phân tích"
                    )}
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Cột phải: Kết quả chẩn đoán & So sánh Grad-CAM */}
          <div className="lg:col-span-7">
            <div className="card-shadow min-h-[300px] rounded-xl border border-line bg-card p-6">
              {!preds && !analyzing && !error && (
                <div className="flex flex-col items-center justify-center py-16 text-center text-muted">
                  <svg className="h-12 w-12 text-slate-300 mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  <p className="max-w-[360px] text-[0.95rem]">
                    Kết quả phân tích chẩn đoán và ảnh nhiệt so sánh Grad-CAM sẽ hiển thị tại đây sau khi bạn tải ảnh và bấm <b>Bắt đầu phân tích</b>.
                  </p>
                </div>
              )}

              {analyzing && (
                <div className="flex flex-col items-center justify-center py-16 text-center text-muted">
                  <div className="mb-4 h-10 w-10 animate-spin rounded-full border-4 border-primary border-t-transparent"></div>
                  <p className="text-[1.05rem] font-medium text-ink">Đang xử lý phân tích trên hệ thống Server...</p>
                  <p className="text-[0.85rem] text-muted mt-1">Trích xuất đặc trưng hình ảnh &amp; tính toán gradient Grad-CAM</p>
                </div>
              )}

              {error && (
                <div className="rounded-lg bg-red-50 border border-red-200 p-4 text-[0.9rem] text-red-800">
                  <b className="font-semibold">Thông báo:</b>
                  <p className="mt-1 text-[0.85rem]">{error}</p>
                </div>
              )}

              {preds && top && topClass && (
                <>
                  {/* Card Thống kê Dự đoán Top 1 */}
                  <div
                    className="mb-6 rounded-xl p-5 text-white shadow-md flex items-center justify-between"
                    style={{ background: topClass.color }}
                  >
                    <div>
                      <div className="text-[0.85rem] uppercase tracking-wider opacity-90 font-medium">
                        Kết quả chẩn đoán chính
                      </div>
                      <div className="text-2xl font-extrabold mt-0.5">{topClass.vi}</div>
                      <div className="text-[0.85rem] opacity-90 mt-1">
                        Mô tả: <span>{CLASSES[top.key]?.desc}</span>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-3xl font-black">{(top.p * 100).toFixed(1)}%</div>
                      <div className="text-[0.78rem] opacity-90 uppercase font-semibold">Độ tin cậy</div>
                    </div>
                  </div>

                  {/* Thanh tỷ lệ xác suất các nhóm */}
                  <div className="mb-6">
                    <h3 className="text-[0.9rem] font-bold text-ink mb-3 uppercase tracking-wider text-muted">
                      Xác suất theo từng phân loại
                    </h3>
                    <div className="space-y-3">
                      {preds.map((o) => (
                        <div key={o.key} className="text-[0.88rem]">
                          <div className="mb-1 flex justify-between font-medium">
                            <span className="text-ink">{CLASSES[o.key]?.vi || o.key}</span>
                            <span className="font-semibold">{(o.p * 100).toFixed(1)}%</span>
                          </div>
                          <div className="h-2.5 overflow-hidden rounded-full bg-slate-100">
                            <div
                              className="h-full rounded-full transition-all duration-500"
                              style={{
                                width: `${(o.p * 100).toFixed(1)}%`,
                                background: CLASSES[o.key]?.color || "#cbd5e1",
                              }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {topClass.isTumor ? (
                    <div className="mb-6 rounded-lg bg-amber-50 border border-amber-200 px-4 py-3 text-[0.88rem] text-amber-900 flex items-center gap-2">
                      <svg className="h-5 w-5 text-amber-600 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                      </svg>
                      <div>
                        <b>Cảnh báo AI:</b> Phát hiện dấu hiệu khối u trong ảnh scan. Cần thực hiện sinh thiết và thăm khám y khoa chính thức.
                      </div>
                    </div>
                  ) : (
                    <div className="mb-6 rounded-lg bg-emerald-50 border border-emerald-200 px-4 py-3 text-[0.88rem] text-emerald-900 flex items-center gap-2">
                      <svg className="h-5 w-5 text-emerald-600 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                      <div>
                        <b>Kết quả bình thường:</b> Không tìm thấy dấu hiệu khối u não nguy hiểm trong ảnh MRI này.
                      </div>
                    </div>
                  )}

                  {/* Phần so sánh ảnh MRI gốc & Grad-CAM */}
                  {heatmap && preview && (
                    <div className="border-t border-line pt-6">
                      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
                        <div>
                          <h3 className="text-base font-bold text-ink">
                            Bản đồ nhiệt giải thích Grad-CAM
                          </h3>
                          <p className="text-[0.8rem] text-muted">
                            So sánh vị trí ảnh gốc và vùng kích hoạt nơ-ron mà AI tập trung phân tích.
                          </p>
                        </div>

                        {/* Chuyển chế độ xem */}
                        <div className="flex rounded-lg bg-slate-100 p-1 border border-slate-200">
                          <button
                            type="button"
                            onClick={() => setViewMode("sideBySide")}
                            className={`rounded-md px-3 py-1.5 text-xs font-semibold transition ${
                              viewMode === "sideBySide"
                                ? "bg-white text-primary shadow-sm"
                                : "text-slate-600 hover:text-ink"
                            }`}
                          >
                            Song song (Side-by-Side)
                          </button>
                          <button
                            type="button"
                            onClick={() => setViewMode("slider")}
                            className={`rounded-md px-3 py-1.5 text-xs font-semibold transition ${
                              viewMode === "slider"
                                ? "bg-white text-primary shadow-sm"
                                : "text-slate-600 hover:text-ink"
                            }`}
                          >
                            Phủ màu (Slider)
                          </button>
                        </div>
                      </div>

                      {/* View Mode 1: Side by Side (Song song) */}
                      {viewMode === "sideBySide" && (
                        <div className="grid gap-4 grid-cols-1 sm:grid-cols-2">
                          {/* Khung Ảnh Gốc */}
                          <div className="flex flex-col rounded-xl border border-slate-200 bg-slate-50 p-3">
                            <div className="mb-2 flex items-center justify-between">
                              <span className="text-[0.8rem] font-bold text-ink uppercase tracking-wide">
                                1. Ảnh MRI Gốc
                              </span>
                              <button
                                onClick={() => setModalImage({ src: preview, title: "Ảnh MRI Gốc" })}
                                className="text-[0.75rem] text-primary hover:underline"
                              >
                                Phóng to
                              </button>
                            </div>
                            <div 
                              role="button"
                              tabIndex={0}
                              onClick={() => setModalImage({ src: preview, title: "Ảnh MRI Gốc" })}
                              onKeyDown={(e) => {
                                if (e.key === "Enter" || e.key === " ") {
                                  e.preventDefault();
                                  setModalImage({ src: preview, title: "Ảnh MRI Gốc" });
                                }
                              }}
                              className="group relative overflow-hidden rounded-lg bg-slate-900 flex items-center justify-center h-[260px] cursor-pointer border border-slate-800 outline-none focus-visible:ring-2 focus-visible:ring-primary"
                            >
                              <img
                                src={preview}
                                alt="Ảnh MRI gốc"
                                className="h-full w-full object-contain transition-transform duration-300 group-hover:scale-105"
                              />
                            </div>
                          </div>

                          {/* Khung Ảnh Grad-CAM Heatmap */}
                          <div className="flex flex-col rounded-xl border border-slate-200 bg-slate-50 p-3">
                            <div className="mb-2 flex items-center justify-between">
                              <span className="text-[0.8rem] font-bold text-primary uppercase tracking-wide">
                                2. Vùng nhiệt Grad-CAM
                              </span>
                              <button
                                onClick={() => setModalImage({ src: heatmap, title: "Bản đồ nhiệt Grad-CAM" })}
                                className="text-[0.75rem] text-primary hover:underline"
                              >
                                Phóng to
                              </button>
                            </div>
                            <div 
                              role="button"
                              tabIndex={0}
                              onClick={() => setModalImage({ src: heatmap, title: "Bản đồ nhiệt Grad-CAM" })}
                              onKeyDown={(e) => {
                                if (e.key === "Enter" || e.key === " ") {
                                  e.preventDefault();
                                  setModalImage({ src: heatmap, title: "Bản đồ nhiệt Grad-CAM" });
                                }
                              }}
                              className="group relative overflow-hidden rounded-lg bg-slate-900 flex items-center justify-center h-[260px] cursor-pointer border border-slate-800 outline-none focus-visible:ring-2 focus-visible:ring-primary"
                            >
                              <img
                                src={heatmap}
                                alt="Bản đồ nhiệt Grad-CAM"
                                className="h-full w-full object-contain transition-transform duration-300 group-hover:scale-105"
                              />
                            </div>
                          </div>
                        </div>
                      )}

                      {/* View Mode 2: Interactive Slider Overlay */}
                      {viewMode === "slider" && (
                        <div className="rounded-xl border border-slate-200 bg-slate-900 p-3 text-white">
                          <div className="mb-2 flex items-center justify-between px-1">
                            <span className="text-[0.8rem] font-semibold text-slate-300">
                              Kéo thanh trượt để đè lớp Grad-CAM lên Ảnh gốc
                            </span>
                            <span className="text-xs font-mono text-primary bg-primary/20 px-2 py-0.5 rounded">
                              {sliderPos}% Grad-CAM
                            </span>
                          </div>

                          <div className="relative h-[320px] sm:h-[360px] w-full overflow-hidden rounded-lg border border-slate-800 select-none">
                            {/* Base Image (Ảnh Gốc) */}
                            <img
                              src={preview}
                              alt="Ảnh MRI gốc"
                              className="absolute inset-0 h-full w-full object-contain bg-slate-950 pointer-events-none"
                            />

                            {/* Overlaid Image (Grad-CAM Heatmap) clipped cleanly using clipPath */}
                            <img
                              src={heatmap}
                              alt="Bản đồ nhiệt Grad-CAM"
                              className="absolute inset-0 h-full w-full object-contain bg-slate-950 pointer-events-none"
                              style={{ clipPath: `inset(0 ${100 - sliderPos}% 0 0)` }}
                            />

                            {/* Slider Line Divider */}
                            <div
                              className="absolute top-0 bottom-0 w-0.5 bg-white shadow-[0_0_10px_rgba(255,255,255,0.8)] pointer-events-none z-10"
                              style={{ left: `${sliderPos}%` }}
                            >
                              <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 h-8 w-8 rounded-full bg-white text-slate-900 shadow-lg flex items-center justify-center text-xs font-bold border-2 border-primary">
                                ↔
                              </div>
                            </div>

                            {/* Native Range Input over the visual container */}
                            <input
                              type="range"
                              min="0"
                              max="100"
                              value={sliderPos}
                              onChange={(e) => setSliderPos(Number(e.target.value))}
                              className="absolute inset-0 opacity-0 cursor-ew-resize z-20 w-full h-full focus-visible:opacity-20 transition-opacity"
                              aria-label="So sánh trượt Grad-CAM và Ảnh gốc"
                            />
                          </div>
                        </div>
                      )}

                      <p className="mt-3 text-[0.82rem] text-muted leading-relaxed">
                        💡 <b>Chú thích:</b> Vùng màu đỏ/cam/vàng nóng tượng trưng cho các điểm ảnh mà mạng nơ-ron nhân tạo tập trung chú ý nhất để đưa ra phán đoán trên.
                      </p>
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        </div>

        {/* Tuyên bố miễn trừ trách nhiệm */}
        <div className="mt-8 rounded-xl border border-[#f3e2b3] bg-[#fff8e6] px-5 py-4 text-[0.92rem] text-[#7a5a13]">
          ⚠️ <b>Tuyên bố miễn trừ:</b> Công cụ mang tính học thuật và hỗ trợ sàng lọc, <b>không thay thế chẩn đoán của bác sĩ chuyên khoa</b>. Mọi quyết định lâm sàng cần dựa trên thăm khám và ý kiến chuyên môn.
        </div>
      </div>

      {/* Modal Lightbox Phóng To Ảnh */}
      {modalImage && (
        <div 
          role="dialog"
          aria-modal="true"
          aria-label={modalImage.title}
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4 backdrop-blur-sm animate-fadeIn"
          onClick={() => setModalImage(null)}
        >
          <div 
            className="relative max-h-[90vh] max-w-[90vw] overflow-hidden rounded-2xl bg-slate-900 p-4 shadow-2xl border border-slate-700 flex flex-col"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="mb-3 flex items-center justify-between px-2 text-white">
              <h4 className="text-base font-semibold">{modalImage.title}</h4>
              <button
                type="button"
                onClick={() => setModalImage(null)}
                className="rounded-full bg-slate-800 p-1.5 text-slate-400 hover:bg-slate-700 hover:text-white"
                aria-label="Đóng dialog"
              >
                <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <div className="flex-1 overflow-auto flex items-center justify-center min-h-[300px]">
              <img
                src={modalImage.src}
                alt={modalImage.title}
                className="max-h-[75vh] w-auto max-w-full object-contain rounded-lg"
              />
            </div>
            <div className="mt-3 text-center text-xs text-slate-400">
              Nhấn ESC hoặc bấm ra ngoài để đóng
            </div>
          </div>
        </div>
      )}
    </section>
  );
}


