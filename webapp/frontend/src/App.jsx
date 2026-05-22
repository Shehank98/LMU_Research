import { Routes, Route } from 'react-router-dom'
import NavBar from './components/NavBar'
import Home from './pages/Home'
import Performance from './pages/Performance'
import LiveDiagnosis from './pages/LiveDiagnosis'
import XAIAnalysis from './pages/XAIAnalysis'
import ModelComparison from './pages/ModelComparison'
import Results from './pages/Results'

export default function App() {
  return (
    <div className="min-h-screen bg-slate-900">
      <NavBar />
      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-8">
        <Routes>
          <Route path="/"           element={<Home />} />
          <Route path="/performance" element={<Performance />} />
          <Route path="/diagnosis"  element={<LiveDiagnosis />} />
          <Route path="/xai"        element={<XAIAnalysis />} />
          <Route path="/comparison" element={<ModelComparison />} />
          <Route path="/results"    element={<Results />} />
        </Routes>
      </main>
    </div>
  )
}
