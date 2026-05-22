const ACCENT = {
  teal:   'border-teal-400',
  amber:  'border-amber-400',
  green:  'border-green-400',
  red:    'border-red-400',
  purple: 'border-purple-400',
}

export default function KPICard({ label, value, subtitle, accent = 'teal' }) {
  return (
    <div className={`bg-slate-800 rounded-xl p-5 border-t-4 ${ACCENT[accent] || ACCENT.teal} shadow-lg`}>
      <div className="text-3xl font-extrabold text-white leading-tight">{value}</div>
      <div className="text-xs text-slate-400 uppercase tracking-widest mt-1">{label}</div>
      {subtitle && <div className="text-xs text-slate-500 mt-0.5">{subtitle}</div>}
    </div>
  )
}
