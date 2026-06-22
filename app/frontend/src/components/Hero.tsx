const STATS = [
  { label: "Số nhóm chẩn đoán", value: "4" },
  { label: "Độ phân giải đầu vào", value: "224×224" },
  { label: "Mô hình", value: "EfficientNetB0" },
  { label: "Xử lý", value: "Ngay trên trình duyệt" },
];

export default function Hero() {
  return (
    <div className="bg-gradient-to-br from-primary to-accent px-6 py-16 text-white">
      <div className="mx-auto grid max-w-[1180px] items-center gap-8 md:grid-cols-[1.2fr_0.8fr]">
        <div>
          <h1 className="mb-4 text-3xl font-extrabold leading-tight md:text-[2.6rem]">
            Hỗ trợ phát hiện khối u não
            <br />
            qua ảnh cộng hưởng từ (MRI)
          </h1>
          <p className="mb-6 max-w-[560px] text-lg text-[#e4eef7]">
            Ứng dụng trí tuệ nhân tạo phân tích ảnh MRI và nhận diện 4 tình
            trạng: u thần kinh đệm, u màng não, u tuyến yên và không có khối u —
            hỗ trợ sàng lọc và tham khảo.
          </p>
          <div className="flex flex-wrap gap-4">
            <a href="#test">
              <button className="rounded-lg bg-white px-5 py-2.5 font-semibold text-primary transition hover:bg-white/90">
                Thử nghiệm ngay
              </button>
            </a>
            <a href="#about">
              <button className="rounded-lg border-[1.5px] border-white/60 bg-transparent px-5 py-2.5 font-semibold text-white transition hover:bg-white/10">
                Tìm hiểu thêm
              </button>
            </a>
          </div>
        </div>
        <div className="rounded-[18px] border border-white/25 bg-white/10 p-6 backdrop-blur">
          {STATS.map((s, i) => (
            <div
              key={s.label}
              className={`flex justify-between py-2.5 ${
                i < STATS.length - 1 ? "border-b border-white/20" : ""
              }`}
            >
              <span>{s.label}</span>
              <b className="text-xl">{s.value}</b>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
