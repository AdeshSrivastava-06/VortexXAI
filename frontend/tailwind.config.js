/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#0F172A',
        surface: '#1E293B',
        accentSafe: '#10B981', // Emerald green
        accentSafeHover: '#059669',
        accentSafeCyan: '#06b6d4', // Cyan
        accentWarning: '#F59E0B', // Amber
        accentCritical: '#EF4444', // Crimson red
      }
    },
  },
  plugins: [],
}
