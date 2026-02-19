import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Webpack config for react-pdf (PDF.js worker)
  webpack: (config) => {
    config.resolve.alias.canvas = false;
    return config;
  },
};

export default nextConfig;
