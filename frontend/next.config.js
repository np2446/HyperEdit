/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      {
        source: '/test_outputs/:path*',
        destination: 'http://localhost:8000/test_outputs/:path*',
      },
    ]
  },
}

module.exports = nextConfig 