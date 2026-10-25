import { Navigate, Route, Routes } from 'react-router-dom'
import Header from './components/Header'
import Diagnose from './pages/Diagnose'
import Results from './pages/Results'

export default function App() {
  return (
    <div className="flex min-h-screen flex-col">
      <Header />
      <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-8 sm:px-6">
        <Routes>
          <Route path="/" element={<Diagnose />} />
          <Route path="/results" element={<Results />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
      <footer className="border-t border-slate-200 bg-white">
        <div className="mx-auto max-w-6xl px-4 py-5 text-xs text-slate-500 sm:px-6">
          <span>Pattiyage Shehan Kavishka · E187041 · London Metropolitan University</span>
        </div>
      </footer>
    </div>
  )
}
