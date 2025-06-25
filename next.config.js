/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  eslint: {
    ignoreDuringBuilds: true,
  },
  images: { unoptimized: true },
  optimizeFonts: false,
  env: {
    NEXT_PUBLIC_DISABLE_GOOGLE_FONTS: 'true',
  },
};

module.exports = nextConfig;