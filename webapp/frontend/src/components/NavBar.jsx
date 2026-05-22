import { NavLink } from 'react-router-dom'

const links = [
  { to: '/',            label: 'Home' },
  { to: '/performance', label: 'Performance' },
  { to: '/diagnosis',   label: 'Live Diagnosis' },
  { to: '/xai',         label: 'XAI Analysis' },
  { to: '/comparison',  label: 'Comparison' },
  { to: '/results',     label: 'Results Gallery' },
]

export default function NavBar() {
  return (
    <nav className="bg-navy-900 bg-slate-950 border-b border-teal-400/20 sticky top-0 z-50 backdrop-blur">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 flex items-center gap-2 h-14">
        <span className="text-teal-400 font-bold text-lg tracking-tight mr-4 shrink-0">
          NeuroScan AI
        </span>
        <div className="flex gap-1 overflow-x-auto scrollbar-none">
          {links.map(l => (
            <NavLink
              key={l.to}
              to={l.to}
              end={l.to === '/'}
              className={({ isActive }) =>
                `px-3 py-1.5 rounded-lg text-sm font-medium whitespace-nowrap transition-colors ${
                  isActive
                    ? 'bg-teal-400/15 text-teal-400 border border-teal-400/30'
                    : 'text-slate-400 hover:text-slate-100 hover:bg-slate-800'
                }`
              }
            >
              {l.label}
            </NavLink>
          ))}
        </div>
      </div>
    </nav>
  )
}
