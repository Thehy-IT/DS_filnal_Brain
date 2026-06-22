import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin", "vietnamese"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  title: "NeuroScan — Hỗ trợ phát hiện khối u não qua ảnh MRI",
  description:
    "Ứng dụng trí tuệ nhân tạo phân tích ảnh MRI và nhận diện 4 tình trạng: u thần kinh đệm, u màng não, u tuyến yên và không có khối u — hỗ trợ sàng lọc và tham khảo.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="vi" className={inter.variable}>
      <body className="font-sans antialiased">{children}</body>
    </html>
  );
}
