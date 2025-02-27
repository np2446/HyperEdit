/** @type {import('next').NextConfig} */
const nextConfig = {
  // Enable environment variables
  env: {
    BACKEND_URL: process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'
  },
  
  // Rewrites to handle API requests to the backend
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'}/:path*`
      }
    ]
  }
}

export default nextConfig 