import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,       // binds to 0.0.0.0 — accessible from outside the machine
    port: 5173,
    proxy: {
      // Proxy all backend routes to FastAPI on port 8000
      // Requests from the browser go to Vite, Vite forwards to backend server-side
      // This means API_URL = "" works from any IP/hostname
      '/health':     { target: 'http://localhost:8000', changeOrigin: true },
      '/chat':       { target: 'http://localhost:8000', changeOrigin: true },
      '/ingest':     { target: 'http://localhost:8000', changeOrigin: true },
      '/namespaces': { target: 'http://localhost:8000', changeOrigin: true },
      '/docs':       { target: 'http://localhost:8000', changeOrigin: true },
      '/metrics':    { target: 'http://localhost:8000', changeOrigin: true },
    }
  }
})
