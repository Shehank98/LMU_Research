import { useState, useEffect } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, Legend,
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar,
} from 'recharts'
import SectionHeader from '../components/SectionHeader'

const FALLBACK_REPORTS = {
  'Final Ensemble': {
    Glioma:     { precision: 0.97, recall: 0.97, f1: 0.97 },
    Meningioma: { precision: 0.97, recall: 0.96, f1: 0.97 },
    'No Tumor': { precision: 0.99, recall: 0.99, f1: 0.99 },
    Pituitary:  { precision: 0.99, recall: 1.00, f1: 0.99 },
  },
}
const FALLBACK_CM = {
  'Final Ensemble':            [[1688,34,6,5],[42,1636,8,13],[5,5,1837,0],[4,4,0,1698]],
  'CNN + Classical Ens.':      [[284,136,29,55],[40,442,16,12],[0,4,534,7],[14,17,6,467]],
  'Xception + Classical Ens.': [[226,127,42,109],[69,368,25,48],[16,19,492,18],[58,56,26,364]],
  'InceptionV3 + Classical Ens.':[[225,148,46,85],[48,376,41,45],[15,40,468,22],[43,32,29,400]],
}
const CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
const FALLBACK_FEAT = {
  'CNN features':         { RF: 85.45, DT: 80.99, SVM: 66.60, Ensemble: 83.71 },
  'InceptionV3 features': { RF: 74.64, DT: 68.05, SVM: 57.34, Ensemble: 71.20 },
  'Xception features':    { RF: 73.68, DT: 68.20, SVM: 56.71, Ensemble: 70.28 },
  'Raw pixels':           { RF: 79.52, DT: 80.02, SVM: 88.26, Ensemble: 83.00 },
}

function ConfusionMatrix({ name, matrix }) {
  const mat = matrix.map(row => [...row])
  const rowSums = mat.map(row => row.reduce((a, b) => a + b, 0))
  const maxPct = Math.max(...mat.flat().map((v, i) => v / (rowSums[Math.floor(i / 4)] || 1)))

  return (
    <div className="bg-slate-800/60 rounded-xl p-4">
      <h4 className="text-sm font-semibold text-slate-300 mb-3 text-center">{name}</h4>
      <div className="grid grid-cols-5 gap-0.5 text-center text-xs">
        <div />
        {CLASS_NAMES.map(c => <div key={c} className="text-slate-500 font-medium truncate px-1">{c.slice(0,4)}</div>)}
        {mat.map((row, r) => [
          <div key={`r${r}`} className="text-slate-500 font-medium flex items-center justify-end pr-1 truncate">{CLASS_NAMES[r].slice(0,4)}</div>,
          ...row.map((v, c) => {
            const pct = v / (rowSums[r] || 1)
            const intensity = Math.round((pct / (maxPct || 1)) * 100)
            const isDiag = r === c
            return (
              <div key={`${r}-${c}`}
                className={`rounded py-1.5 font-mono ${isDiag ? 'ring-1 ring-teal-400/50' : ''}`}
                style={{ background: `rgba(0,180,216,${isDiag ? pct * 0.8 + 0.1 : pct * 0.5})` }}
                title={`${CLASS_NAMES[r]} → ${CLASS_NAMES[c]}: ${v} (${(pct * 100).toFixed(1)}%)`}
              >
                <div className="text-white font-bold leading-none">{v}</div>
                <div className="text-teal-200/60 text-[10px]">{(pct * 100).toFixed(0)}%</div>
              </div>
            )
          }),
        ])}
      </div>
    </div>
  )
}

const RADAR_COLORS = ['#00b4d8', '#e74c3c', '#27ae60', '#f39c12']

export default function Performance() {
  const [data, setData] = useState({
    classification_reports: FALLBACK_REPORTS,
    confusion_matrices: FALLBACK_CM,
    feature_ml_accuracy: FALLBACK_FEAT,
  })

  useEffect(() => {
    fetch('/api/research').then(r => r.json()).then(d => setData(prev => ({ ...prev, ...d }))).catch(() => {})
  }, [])

  const reports = data.classification_reports || FALLBACK_REPORTS
  const cms = data.confusion_matrices || FALLBACK_CM
  const featAcc = data.feature_ml_accuracy || FALLBACK_FEAT

  // Grouped bar: standalone vs ensemble per backbone
  const overviewData = [
    { name: 'CNN',         standalone: 94.00, ensemble: 83.71 },
    { name: 'Inception V3', standalone: 84.63, ensemble: 71.20 },
    { name: 'Xception',    standalone: 86.91, ensemble: 70.28 },
  ]

  // Radar F1
  const radarData = CLASS_NAMES.map(cls => {
    const point = { class: cls }
    Object.entries(reports).forEach(([model, rep]) => {
      point[model.replace(' + Classical Ens.', '').replace(' + Classical Ensemble', '')] = rep[cls]?.f1 || 0
    })
    return point
  })
  const radarModels = Object.keys(reports).map(m => m.replace(' + Classical Ens.', '').replace(' + Classical Ensemble', ''))

  // Feature ML chart
  const clfs = ['RF', 'DT', 'SVM', 'Ensemble']
  const clfColors = ['#2980b9', '#e74c3c', '#27ae60', '#f39c12']
  const featSources = Object.keys(featAcc)

  // Class difficulty (recall per class per model)
  const diffData = CLASS_NAMES.map(cls => {
    const point = { class: cls }
    Object.entries(reports).forEach(([model, rep]) => {
      const short = model.split(' ')[0]
      point[short] = rep[cls]?.recall || 0
    })
    return point
  })
  const diffModels = Object.keys(reports).map(m => m.split(' ')[0])

  // Metrics table rows
  const tableRows = []
  Object.entries(reports).forEach(([model, rep]) => {
    CLASS_NAMES.forEach(cls => {
      tableRows.push({ model, cls, ...rep[cls] })
    })
  })

  return (
    <div className="space-y-10">
      <div>
        <h2 className="text-2xl font-extrabold text-white">Model Performance Dashboard</h2>
        <p className="text-slate-400 mt-1">Pre-computed results from full research study (2,063 test images, 4 classes)</p>
      </div>

      {/* 1. Accuracy overview */}
      <div className="card-bordered">
        <SectionHeader title="Accuracy Overview" subtitle="Standalone vs ensemble accuracy per backbone" />
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={overviewData} margin={{ left: 10, right: 20, top: 10, bottom: 5 }}>
            <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 12 }} />
            <YAxis domain={[60, 100]} tickFormatter={v => `${v}%`} tick={{ fill: '#94a3b8', fontSize: 11 }} />
            <Tooltip formatter={v => [`${v.toFixed(2)}%`]}
              contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }} />
            <Legend wrapperStyle={{ color: '#94a3b8', fontSize: 12 }} />
            <Bar dataKey="standalone" name="Standalone" fill="#2980b9" radius={[4,4,0,0]} />
            <Bar dataKey="ensemble" name="+ Classical Ens." fill="#8e44ad" radius={[4,4,0,0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* 2. Confusion matrices */}
      <div className="card-bordered">
        <SectionHeader title="Confusion Matrices" subtitle="Rows = actual class · Columns = predicted class" />
        <div className="grid sm:grid-cols-2 gap-4">
          {Object.entries(cms).map(([name, matrix]) => (
            <ConfusionMatrix key={name} name={name} matrix={matrix} />
          ))}
        </div>
      </div>

      {/* 3. Per-class metrics table */}
      <div className="card-bordered">
        <SectionHeader title="Per-Class Classification Metrics" subtitle="Precision · Recall · F1-Score by model" />
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-xs text-slate-400 uppercase border-b border-slate-700">
                <th className="text-left py-2 pr-4">Model</th>
                <th className="text-left py-2 pr-4">Class</th>
                <th className="text-right py-2 px-3">Precision</th>
                <th className="text-right py-2 px-3">Recall</th>
                <th className="text-right py-2 px-3">F1</th>
              </tr>
            </thead>
            <tbody>
              {tableRows.map(({ model, cls, precision, recall, f1 }, i) => {
                const f1Color = f1 >= 0.95 ? 'text-green-400' : f1 >= 0.80 ? 'text-teal-400' : f1 >= 0.65 ? 'text-amber-400' : 'text-red-400'
                return (
                  <tr key={i} className="border-b border-slate-700/40 hover:bg-slate-700/20">
                    <td className="py-1.5 pr-4 text-slate-400 text-xs">{i % CLASS_NAMES.length === 0 ? model : ''}</td>
                    <td className="py-1.5 pr-4 text-slate-300">{cls}</td>
                    <td className="py-1.5 px-3 text-right font-mono text-slate-300">{precision?.toFixed(2)}</td>
                    <td className="py-1.5 px-3 text-right font-mono text-slate-300">{recall?.toFixed(2)}</td>
                    <td className={`py-1.5 px-3 text-right font-mono font-bold ${f1Color}`}>{f1?.toFixed(2)}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* 4. F1 Radar */}
      <div className="card-bordered">
        <SectionHeader title="F1-Score Radar" subtitle="Per-class F1 across models" />
        <ResponsiveContainer width="100%" height={320}>
          <RadarChart data={radarData}>
            <PolarGrid stroke="#334155" />
            <PolarAngleAxis dataKey="class" tick={{ fill: '#94a3b8', fontSize: 12 }} />
            <PolarRadiusAxis domain={[0, 1]} tick={{ fill: '#94a3b8', fontSize: 10 }} />
            {radarModels.map((m, i) => (
              <Radar key={m} name={m} dataKey={m} stroke={RADAR_COLORS[i % RADAR_COLORS.length]}
                fill={RADAR_COLORS[i % RADAR_COLORS.length]} fillOpacity={0.15} />
            ))}
            <Legend wrapperStyle={{ color: '#94a3b8', fontSize: 11 }} />
            <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }} />
          </RadarChart>
        </ResponsiveContainer>
      </div>

      {/* 5. Classical ML by feature source */}
      <div className="card-bordered">
        <SectionHeader title="Classical ML by Feature Source"
          subtitle="RF / DT / SVM accuracy when trained on different feature extractors" />
        <ResponsiveContainer width="100%" height={280}>
          <BarChart
            data={featSources.map(src => ({ src, ...featAcc[src] }))}
            margin={{ left: 10, right: 20, top: 10, bottom: 5 }}
          >
            <XAxis dataKey="src" tick={{ fill: '#94a3b8', fontSize: 11 }} />
            <YAxis domain={[50, 95]} tickFormatter={v => `${v}%`} tick={{ fill: '#94a3b8', fontSize: 11 }} />
            <Tooltip formatter={v => [`${Number(v).toFixed(1)}%`]}
              contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }} />
            <Legend wrapperStyle={{ color: '#94a3b8', fontSize: 12 }} />
            {clfs.map((clf, i) => (
              <Bar key={clf} dataKey={clf} fill={clfColors[i]} radius={[3,3,0,0]} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* 6. Class difficulty */}
      <div className="card-bordered">
        <SectionHeader title="Class Difficulty Analysis"
          subtitle="Recall per class — lower = harder to correctly identify" />
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={diffData} margin={{ left: 10, right: 20, top: 10, bottom: 5 }}>
            <XAxis dataKey="class" tick={{ fill: '#94a3b8', fontSize: 12 }} />
            <YAxis domain={[0, 1.1]} tickFormatter={v => v.toFixed(1)} tick={{ fill: '#94a3b8', fontSize: 11 }} />
            <Tooltip formatter={v => [Number(v).toFixed(2), 'Recall']}
              contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }} />
            <Legend wrapperStyle={{ color: '#94a3b8', fontSize: 12 }} />
            {diffModels.map((m, i) => (
              <Bar key={m} dataKey={m} fill={RADAR_COLORS[i % RADAR_COLORS.length]} radius={[3,3,0,0]} />
            ))}
          </BarChart>
        </ResponsiveContainer>
        <p className="text-xs text-slate-500 mt-2">
          Glioma consistently has the lowest recall — most frequently misclassified class,
          expected as glioma can appear morphologically similar to meningioma on MRI.
        </p>
      </div>
    </div>
  )
}
