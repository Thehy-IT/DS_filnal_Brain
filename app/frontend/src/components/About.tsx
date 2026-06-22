function Card({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return (
    <div className={`card-shadow rounded-[14px] border border-line bg-card p-6 ${className}`}>
      {children}
    </div>
  );
}

export default function About() {
  return (
    <section id="about" className="mx-auto max-w-[1180px] px-6 py-14">
      <div className="mb-2 text-[0.8rem] font-bold uppercase tracking-[0.08em] text-primary">
        Tổng quan
      </div>
      <h2 className="mb-4 text-[1.9rem] font-extrabold">Khối u não là gì?</h2>

      <Card className="mb-6">
        <p>
          U não là một trong những bệnh lý thần kinh phức tạp và tiềm ẩn nhiều
          nguy cơ ảnh hưởng đến sức khỏe cũng như tính mạng của người bệnh. Việc
          hiểu rõ <b>khối u não là gì</b>, <b>nguyên nhân hình thành</b> cũng như{" "}
          <b>cách điều trị ra sao</b> đóng vai trò then chốt trong việc chẩn đoán
          chính xác và xây dựng phác đồ điều trị hiệu quả.
        </p>
        <p className="mt-3">
          Khối u não là sự phát triển bất thường của các tế bào trong não hoặc mô
          lân cận. Khối u có thể <b>lành tính</b> (phát triển chậm, ranh giới rõ)
          hoặc <b>ác tính</b> (phát triển nhanh, xâm lấn). Dù lành hay ác, khối u
          đều có thể gây tăng áp lực nội sọ, chèn ép mô não và ảnh hưởng chức năng
          thần kinh.
        </p>
      </Card>

      <div className="grid gap-5 md:grid-cols-3">
        <Card>
          <h3 className="mb-2.5 flex items-center gap-2 text-[1.15rem] font-semibold">
            <span className="grid h-[34px] w-[34px] flex-none place-items-center rounded-[9px] bg-accent text-white">
              🔍
            </span>
            Chẩn đoán
          </h3>
          <p className="text-muted">
            MRI là phương pháp hình ảnh chủ lực giúp xác định vị trí, kích thước và
            đặc điểm khối u với độ tương phản mô mềm cao.
          </p>
        </Card>

        <Card>
          <h3 className="mb-2.5 flex items-center gap-2 text-[1.15rem] font-semibold">
            <span className="grid h-[34px] w-[34px] flex-none place-items-center rounded-[9px] bg-primary text-white">
              ⚠️
            </span>
            Triệu chứng
          </h3>
          <ul className="ml-[1.1rem] list-disc text-muted">
            <li className="my-1">Đau đầu kéo dài, nặng dần</li>
            <li className="my-1">Buồn nôn, co giật</li>
            <li className="my-1">Suy giảm thị lực/thính lực</li>
            <li className="my-1">Rối loạn vận động, trí nhớ</li>
          </ul>
        </Card>

        <Card>
          <h3 className="mb-2.5 flex items-center gap-2 text-[1.15rem] font-semibold">
            <span className="grid h-[34px] w-[34px] flex-none place-items-center rounded-[9px] bg-tgreen text-white">
              ⏱️
            </span>
            Tầm quan trọng
          </h3>
          <p className="text-muted">
            Phát hiện sớm và chính xác làm tăng đáng kể hiệu quả điều trị và tiên
            lượng sống của người bệnh.
          </p>
        </Card>
      </div>
    </section>
  );
}
