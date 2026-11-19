/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', '-apple-system', 'Segoe UI', 'Roboto', 'sans-serif'],
      },
      colors: {
        brand: { 50: '#f0fdfa', 100: '#ccfbf1', 600: '#0d9488', 700: '#0f766e', 800: '#115e59' },
      },
    },
  },
  plugins: [],
}
