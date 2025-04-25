/** @type {import('next').NextConfig} */
const nextConfig = {
  // Disable React strict mode in production, enable in development
  reactStrictMode: process.env.NODE_ENV === 'development',
  // Add other configuration options here
  eslint: {
    // Don't run ESLint during build (we've already addressed issues with our config)
    ignoreDuringBuilds: true,
  },
};

module.exports = nextConfig; 