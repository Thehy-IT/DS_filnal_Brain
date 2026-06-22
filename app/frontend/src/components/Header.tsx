const NAV = [
  { href: "#about", label: "Khái niệm" },
  { href: "#causes", label: "Nguyên nhân" },
  { href: "#types", label: "Phân loại" },
  { href: "#treatment", label: "Điều trị" },
  { href: "#test", label: "Thử nghiệm" },
];

export default function Header() {
  return (
    <header className="sticky top-0 z-50 border-b border-line bg-white/90 backdrop-blur">
      <div className="mx-auto flex max-w-[1180px] items-center justify-between px-6 py-3.5">
        <div className="flex items-center gap-2.5 text-xl font-extrabold text-primary">
          <span className="grid h-9 w-9 place-items-center rounded-[9px] bg-gradient-to-br from-primary to-accent text-xl text-white">
            🧠
          </span>
          NeuroScan
        </div>
        <nav className="hidden gap-6 text-[0.95rem] font-medium md:flex">
          {NAV.map((n) => (
            <a
              key={n.href}
              href={n.href}
              className="text-muted transition-colors hover:text-primary"
            >
              {n.label}
            </a>
          ))}
        </nav>
        <a href="#test">
          <button className="rounded-lg bg-primary px-5 py-2.5 text-[0.95rem] font-semibold text-white transition-colors hover:bg-primary-dark">
            Phân tích ảnh MRI
          </button>
        </a>
      </div>
    </header>
  );
}
