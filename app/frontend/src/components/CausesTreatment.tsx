export default function CausesTreatment() {
  return (
    <section id="causes" className="mx-auto max-w-[1180px] px-6 pb-14 pt-4">
      <div className="grid gap-5 md:grid-cols-2">
        <div className="card-shadow rounded-[14px] border border-line bg-card p-6">
          <div className="mb-2 text-[0.8rem] font-bold uppercase tracking-[0.08em] text-primary">
            Cơ chế
          </div>
          <h3 className="mb-3 text-[1.3rem] font-semibold">
            Nguyên nhân &amp; yếu tố nguy cơ
          </h3>
          <ul className="ml-[1.1rem] list-disc text-muted">
            <li className="my-1">Đột biến gen và yếu tố di truyền</li>
            <li className="my-1">Phơi nhiễm bức xạ ion hóa</li>
            <li className="my-1">Tuổi tác, tiền sử gia đình</li>
            <li className="my-1">Một số rối loạn hệ miễn dịch</li>
          </ul>
          <p className="mt-2.5 text-[0.92rem] text-muted">
            Phần lớn trường hợp chưa xác định được nguyên nhân chính xác.
          </p>
        </div>

        <div
          id="treatment"
          className="card-shadow rounded-[14px] border border-line bg-card p-6"
        >
          <div className="mb-2 text-[0.8rem] font-bold uppercase tracking-[0.08em] text-primary">
            Phác đồ
          </div>
          <h3 className="mb-3 text-[1.3rem] font-semibold">Hướng điều trị</h3>
          <ul className="ml-[1.1rem] list-disc text-muted">
            <li className="my-1">Phẫu thuật loại bỏ khối u</li>
            <li className="my-1">Xạ trị (radiotherapy)</li>
            <li className="my-1">Hóa trị (chemotherapy)</li>
            <li className="my-1">Theo dõi định kỳ với u lành tính nhỏ</li>
          </ul>
          <p className="mt-2.5 text-[0.92rem] text-muted">
            Phác đồ phụ thuộc loại u, vị trí, kích thước và thể trạng bệnh nhân.
          </p>
        </div>
      </div>
    </section>
  );
}
