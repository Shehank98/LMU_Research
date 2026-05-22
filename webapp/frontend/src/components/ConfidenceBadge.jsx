const CFG = {
  high:     { bg: 'bg-green-500/15',  text: 'text-green-400',  border: 'border-green-500/30',  label: 'HIGH CONFIDENCE' },
  moderate: { bg: 'bg-amber-500/15',  text: 'text-amber-400',  border: 'border-amber-500/30',  label: 'MODERATE' },
  low:      { bg: 'bg-red-500/15',    text: 'text-red-400',    border: 'border-red-500/30',    label: 'LOW CONFIDENCE' },
}

export default function ConfidenceBadge({ level, pct }) {
  const c = CFG[level] || CFG.low
  return (
    <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold border ${c.bg} ${c.text} ${c.border}`}>
      {c.label} {typeof pct === 'number' ? pct.toFixed(1) + '%' : ''}
    </span>
  )
}
