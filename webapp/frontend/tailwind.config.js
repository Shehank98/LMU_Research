/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        teal:  { 400: '#00b4d8', 500: '#0096c7', 600: '#0077b6' },
        navy:  { 900: '#0d1b2a', 800: '#1a2e44', 700: '#1e3a5f' },
      },
    },
  },
  plugins: [],
}
