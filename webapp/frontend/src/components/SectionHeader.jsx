export default function SectionHeader({ title, subtitle }) {
  return (
    <div className="border-l-4 border-teal-400 pl-3 mb-5">
      <h3 className="text-lg font-bold text-white">{title}</h3>
      {subtitle && <p className="text-sm text-slate-400 mt-0.5">{subtitle}</p>}
    </div>
  )
}
