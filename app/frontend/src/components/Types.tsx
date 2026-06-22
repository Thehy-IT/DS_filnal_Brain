import { CLASS_ORDER, CLASSES } from "@/lib/classes";

export default function Types() {
  return (
    <section id="types" className="mx-auto max-w-[1180px] px-6 py-14">
      <div className="mb-2 text-[0.8rem] font-bold uppercase tracking-[0.08em] text-primary">
        Phạm vi nhận diện
      </div>
      <h2 className="mb-4 text-[1.9rem] font-extrabold">
        4 nhóm mô hình phân loại
      </h2>
      <p className="mb-8 max-w-[780px] text-[1.05rem] text-muted">
        Mô hình được huấn luyện để phân biệt bốn tình trạng phổ biến trên ảnh
        MRI.
      </p>

      <div className="grid gap-5 md:grid-cols-4">
        {CLASS_ORDER.map((key) => {
          const c = CLASSES[key];
          return (
            <div
              key={key}
              className="card-shadow rounded-[14px] border border-line border-t-4 bg-card p-6"
              style={{ borderTopColor: c.color }}
            >
              <span
                className="mb-2.5 inline-block rounded-[20px] px-3 py-1 text-[0.82rem] font-semibold text-white"
                style={{ background: c.color }}
              >
                {c.vi}
              </span>
              <p className="text-[0.92rem] text-muted">{c.desc}</p>
            </div>
          );
        })}
      </div>
    </section>
  );
}
