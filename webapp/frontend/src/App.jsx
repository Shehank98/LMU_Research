import { Routes, Route, Navigate } from 'react-router-dom'
import NavBar from './components/NavBar'
import LiveDiagnosis from './pages/LiveDiagnosis'
import Results from './pages/Results'

export default function App() {
  return (
    <div className="min-h-screen bg-slate-900">
      <NavBar />
      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-8">
        <Routes>
          <Route path="/"          element={<LiveDiagnosis />} />
          <Route path="/diagnosis" element={<LiveDiagnosis />} />
          <Route path="/results"   element={<Results />} />
          <Route path="*"          element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  )
}
