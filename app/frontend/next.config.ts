import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // API Proxy: /api/* → FastAPI backend
  // Giúp tránh CORS hoàn toàn — trình duyệt chỉ gọi cùng origin (port 3000)
  // Next.js server-side sẽ forward request đến backend
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${process.env.BACKEND_URL || "http://127.0.0.1:8000"}/:path*`,
      },
    ];
  },
};

export default nextConfig;
