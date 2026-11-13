/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        surface: {
          0: '#05070C',
          1: '#0A0D12',
          2: '#0F131C',
          3: '#161D2B',
          4: '#1E2636',
        },
        accent: {
          primary: '#38BDF8',
        },
        risk: {
          low: '#10B981',
          medium: '#FBBF24',
          high: '#F97316',
          critical: '#EF4444',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Consolas', 'monospace'],
      },
    },
  },
  plugins: [],
}
