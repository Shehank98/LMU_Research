import { useEffect, useState } from 'react'
import { NavLink } from 'react-router-dom'
import { getModelStatus } from '../api'

const NAV = [
  { to: '/', label: 'Diagnose', end: true },
  { to: '/results', label: 'Results' },
  { to: '/about', label: 'About' },
]

function ModelStatus() {
  const [status, setStatus] = useState(null)
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    let timer
    const poll = async () => {
      try {
        const s = await getModelStatus()
        setStatus(s)
        setFailed(false)
        if (s.loading) timer = setTimeout(poll, 4000)
      } catch {
        setFailed(true)
        timer = setTimeout(poll, 8000)
      }
    }
    poll()
    return () => clearTimeout(timer)
  }, [])

  let dot = 'bg-slate-300', text = 'Checking models'
  if (failed) { dot = 'bg-rose-500'; text = 'Server unreachable' }
  else if (status?.loading) { dot = 'bg-amber-400 animate-pulse'; text = 'Loading models' }
  else if (status) {
    const { loaded_count: n, total_count: t } = status
    dot = n === t ? 'bg-emerald-500' : n > 0 ? 'bg-amber-400' : 'bg-rose-500'
    text = `${n} of ${t} model files ready`
  }
  return (
    <span className="hidden items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1 text-xs text-slate-600 sm:inline-flex"
          title={status?.missing?.length ? `Missing: ${status.missing.join(', ')}` : status?.repo}>
      <span className={`h-2 w-2 rounded-full ${dot}`} />
      {text}
    </span>
  )
}

export default function Header() {
  return (
    <header className="sticky top-0 z-10 border-b border-slate-200 bg-white/90 backdrop-blur">
      <div className="mx-auto flex h-16 max-w-6xl items-center justify-between gap-4 px-4 sm:px-6">
        <div className="flex items-center gap-4 sm:gap-8">
          <NavLink to="/" className="flex items-center gap-2.5">
            <span className="grid h-8 w-8 place-items-center rounded-lg bg-brand-700">
              <span className="h-3.5 w-3.5 rounded-full border-[2.5px] border-white" />
            </span>
            <span className="leading-tight">
              <span className="block text-sm font-semibold text-slate-900">NeuroScan</span>
              <span className="hidden text-xs text-slate-500 sm:block">Brain tumour MRI classification</span>
            </span>
          </NavLink>
          <nav className="flex items-center gap-1">
            {NAV.map(({ to, label, end }) => (
              <NavLink key={to} to={to} end={end}
                className={({ isActive }) =>
                  `rounded-md px-2.5 py-2 text-sm font-medium sm:px-3 transition-colors ${
                    isActive ? 'bg-slate-100 text-slate-900' : 'text-slate-500 hover:text-slate-900'}`}>
                {label}
              </NavLink>
            ))}
          </nav>
        </div>
        <ModelStatus />
      </div>
    </header>
  )
}
