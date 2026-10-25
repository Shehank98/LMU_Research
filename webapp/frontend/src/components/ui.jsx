export const CLASS_COLORS = {
  Glioma: '#e11d48',
  Meningioma: '#d97706',
  'No Tumor': '#059669',
  Pituitary: '#0284c7',
}

export const pct = (x, digits = 1) =>
  x === null || x === undefined ? '–' : `${(x * 100).toFixed(digits)}%`

export function PageHeader({ title, description, children }) {
  return (
    <div className="mb-6 flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
        {description && <p className="mt-1 max-w-2xl text-sm text-slate-500">{description}</p>}
      </div>
      {children}
    </div>
  )
}

export function Card({ title, subtitle, action, children, className = '' }) {
  return (
    <section className={`card ${className}`}>
      {(title || action) && (
        <div className="flex items-start justify-between gap-4 border-b border-slate-100 px-5 py-4 sm:px-6">
          <div>
            <h2 className="section-title">{title}</h2>
            {subtitle && <p className="mt-0.5 text-sm text-slate-500">{subtitle}</p>}
          </div>
          {action}
        </div>
      )}
      <div className="card-pad">{children}</div>
    </section>
  )
}

const TONES = {
  green: 'bg-emerald-50 text-emerald-700 ring-emerald-200',
  amber: 'bg-amber-50 text-amber-700 ring-amber-200',
  red: 'bg-rose-50 text-rose-700 ring-rose-200',
  slate: 'bg-slate-100 text-slate-600 ring-slate-200',
  brand: 'bg-brand-50 text-brand-700 ring-brand-100',
}

export function Badge({ tone = 'slate', children }) {
  return (
    <span className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 ring-inset ${TONES[tone]}`}>
      {children}
    </span>
  )
}

export const levelTone = (level) => ({ high: 'green', moderate: 'amber', low: 'red' }[level] || 'slate')

export function Bar({ value, color = '#0f766e', height = 'h-2' }) {
  const w = Math.max(0, Math.min(1, value || 0)) * 100
  return (
    <div className={`w-full overflow-hidden rounded-full bg-slate-100 ${height}`}>
      <div className={`${height} rounded-full transition-all duration-500`} style={{ width: `${w}%`, backgroundColor: color }} />
    </div>
  )
}

export function Stat({ label, value, hint }) {
  return (
    <div className="card px-5 py-4">
      <p className="text-xs font-medium uppercase tracking-wide text-slate-500">{label}</p>
      <p className="mt-1 text-2xl font-semibold text-slate-900">{value}</p>
      {hint && <p className="mt-0.5 truncate text-xs text-slate-500" title={hint}>{hint}</p>}
    </div>
  )
}

export function Spinner({ className = 'h-4 w-4' }) {
  return (
    <svg className={`animate-spin ${className}`} viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeOpacity="0.25" strokeWidth="4" />
      <path d="M22 12a10 10 0 0 0-10-10" stroke="currentColor" strokeWidth="4" strokeLinecap="round" />
    </svg>
  )
}
