import { useState, useEffect } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
  LineChart, Line, CartesianGrid, PieChart, Pie, Legend,
} from 'recharts'
import KPICard from '../components/KPICard'
import SectionHeader from '../components/SectionHeader'

const FALLBACK = {
  model_accuracy: {
    'All Models Combined': 98.20,
    'CNN': 94.00,
    'Xception': 86.91,
    'Traditional ML + Ensemble Models': 83.00,
    'CNN + Ensemble Models': 83.71,
    'Inception V3': 84.63,
    'Inception V3 + Ensemble Models': 71.20,
    'Xception + Ensemble Models': 70.28,
  },
  family_colors: {
    'All Models Combined': '#f39c12',
    'CNN': '#8e44ad',
    'Inception V3': '#2980b9',
    'Xception': '#16a085',
    'Traditional ML + Ensemble Models': '#7f8c8d',
    'CNN + Ensemble Models': '#8e44ad',
    'Inception V3 + Ensemble Models': '#2980b9',
    'Xception + Ensemble Models': '#16a085',
  },
  ensemble_buildup: [
    { models: 'CNN only', accuracy: 94.00 },
    { models: '+ Inception V3', accuracy: 95.40 },
    { models: '+ Xception', accuracy: 96.20 },
    { models: '+ CNN Ensemble', accuracy: 96.80 },
    { models: '+ Inc Ensemble', accuracy: 97.40 },
    { models: '+ Xcp Ensemble', accuracy: 98.20 },
  ],
  dataset: {
    Training: { Glioma: 1733, Meningioma: 1699, 'No Tumor': 1847, Pituitary: 1705 },
    Testing:  { Glioma: 504,  Meningioma: 510,  'No Tumor': 545,  Pituitary: 504  },
  },
}

const CLASS_PIE_COLORS = ['#e74c3c', '#e67e22', '#27ae60', '#2980b9']

export default function Home() {
  const [data, setData] = useState(FALLBACK)
  const [abstractOpen, setAbstractOpen] = useState(false)

  useEffect(() => {
    fetch('/api/research')
      .then(r => r.json())
      .then(d => setData({ ...FALLBACK, ...d }))
      .catch(() => {})
  }, [])

  const accData = Object.entries(data.model_accuracy)
    .sort((a, b) => a[1] - b[1])
    .map(([name, acc]) => ({
      name,
      accuracy: acc,
      color: data.family_colors?.[name] || '#64748b',
    }))

  const buildupData = (data.ensemble_buildup || FALLBACK.ensemble_buildup).map((e, i) => ({
    step: i + 1,
    label: e.models.replace(/^\+ /, '').slice(0, 18),
    accuracy: e.accuracy,
  }))

  const objectives = [
    ['RO1', 'Evaluate CNN feature extraction for brain tumor identification'],
    ['RO2', 'Assess RF/DT/SVM performance on CNN-extracted features'],
    ['RO3', 'Investigate Xception and Inception V3 for feature extraction'],
    ['RO4', 'Evaluate ML classifiers on pre-trained model features'],
    ['RO5', 'Assess accuracy gains from ensemble combination'],
    ['RO6', 'Compare CNN-based vs fine-tuned feature extraction'],
  ]

  return (
    <div className="space-y-8">
      {/* Hero */}
      <div>
        <h1 className="text-3xl font-extrabold text-white mb-1">NeuroScan AI</h1>
        <p className="text-slate-400">Ensemble Consensus XAI System for Brain Tumor Classification from MRI</p>
      </div>

      {/* KPI row */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
        <KPICard label="Final Ensemble Accuracy" value="98.20%" accent="amber" />
        <KPICard label="MRI Scans"      value="9,047"  subtitle="Kaggle dataset" />
        <KPICard label="Tumor Classes"  value="4" />
        <KPICard label="Combined Models" value="6" accent="purple" />
        <KPICard label="Weighted F1"    value="0.98" accent="green" />
      </div>

      {/* Abstract */}
      <div className="card-bordered">
        <button
          onClick={() => setAbstractOpen(o => !o)}
          className="flex items-center justify-between w-full text-left"
        >
          <span className="font-semibold text-teal-400">Research Abstract</span>
          <span className="text-slate-400 text-sm">{abstractOpen ? 'Collapse' : 'Expand'}</span>
        </button>
        {abstractOpen && (
          <p className="mt-3 text-sm text-slate-300 leading-relaxed">
            This study examines how well fine-tuned models perform in feature extraction compared to ordinary
            Convolutional Neural Networks (CNNs) and how their integration with ensemble techniques affects the
            precision and dependability of identifying brain tumors from MRI images. The purpose is to assess
            the performance of SVM, RF, DT, Xception, and Inception V3 classifiers using a mixed-methods approach.
            The final ensemble model, composed of six saved models, outperformed the CNN's highest individual
            accuracy of 94% with a final accuracy of <strong className="text-white">98.20%</strong>.
          </p>
        )}
      </div>

      {/* Objectives + accuracy table */}
      <div className="grid lg:grid-cols-2 gap-6">
        <div className="card-bordered">
          <SectionHeader title="Research Objectives" subtitle="Mixed-methods, deductive approach" />
          <div className="space-y-2">
            {objectives.map(([code, text]) => (
              <div key={code} className="flex items-start gap-2">
                <span className="shrink-0 bg-teal-500 text-white text-xs font-bold px-2 py-0.5 rounded mt-0.5">{code}</span>
                <span className="text-sm text-slate-300">{text}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="card-bordered">
          <SectionHeader title="Accuracy Summary" subtitle="All models on official test set" />
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-xs text-slate-400 uppercase border-b border-slate-700">
                  <th className="text-left py-2 pr-4">Model</th>
                  <th className="text-right py-2">Accuracy</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(data.model_accuracy)
                  .sort((a, b) => b[1] - a[1])
                  .map(([name, acc]) => (
                    <tr key={name} className="border-b border-slate-700/50 hover:bg-slate-700/30">
                      <td className="py-2 pr-4 text-slate-300">{name}</td>
                      <td className="py-2 text-right font-mono font-semibold text-white">{acc.toFixed(2)}%</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Accuracy bar chart */}
      <div className="card-bordered">
        <SectionHeader title="Model Accuracy Comparison" subtitle="All configurations on 2,063 test images" />
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={accData} layout="vertical" margin={{ left: 20, right: 60, top: 5, bottom: 5 }}>
            <XAxis type="number" domain={[60, 102]} tickFormatter={v => `${v}%`}
              tick={{ fill: '#94a3b8', fontSize: 11 }} />
            <YAxis type="category" dataKey="name" width={210}
              tick={{ fill: '#94a3b8', fontSize: 11 }} />
            <Tooltip
              formatter={v => [`${Number(v).toFixed(2)}%`, 'Accuracy']}
              contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
              labelStyle={{ color: '#e2e8f0' }}
            />
            <Bar dataKey="accuracy" radius={[0, 4, 4, 0]}
              label={{ position: 'right', formatter: v => `${Number(v).toFixed(1)}%`, fill: '#94a3b8', fontSize: 11 }}>
              {accData.map((d, i) => <Cell key={i} fill={d.color} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Ensemble build-up */}
      <div className="card-bordered">
        <SectionHeader title="Ensemble Build-Up" subtitle="Accuracy gain as models are added one by one" />
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={buildupData} margin={{ left: 10, right: 30, top: 10, bottom: 30 }}>
            <CartesianGrid stroke="#334155" strokeDasharray="3 3" />
            <XAxis dataKey="label" tick={{ fill: '#94a3b8', fontSize: 11 }} angle={-20} textAnchor="end" height={50} />
            <YAxis domain={[88, 100]} tickFormatter={v => `${v}%`} tick={{ fill: '#94a3b8', fontSize: 11 }} />
            <Tooltip
              formatter={v => [`${v}%`, 'Accuracy']}
              contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
            />
            <Line type="monotone" dataKey="accuracy" stroke="#00b4d8" strokeWidth={3}
              dot={{ fill: '#00b4d8', r: 5 }}
              label={{ position: 'top', formatter: v => `${v}%`, fill: '#00b4d8', fontSize: 10, dy: -6 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Dataset distribution */}
      <div className="card-bordered">
        <SectionHeader title="Dataset Distribution" subtitle="Kaggle Brain Tumor MRI Dataset — 9,047 scans" />
        <div className="grid sm:grid-cols-2 gap-4">
          {['Training', 'Testing'].map(split => {
            const counts = data.dataset?.[split] || {}
            const pieData = Object.entries(counts).map(([name, value]) => ({ name, value }))
            return (
              <div key={split}>
                <p className="text-sm text-slate-400 text-center mb-2 font-medium">
                  {split} Set — {Object.values(counts).reduce((a, b) => a + b, 0).toLocaleString()} images
                </p>
                <ResponsiveContainer width="100%" height={220}>
                  <PieChart>
                    <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%"
                      innerRadius={50} outerRadius={85} paddingAngle={2}
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                      labelLine={{ stroke: '#64748b' }}>
                      {pieData.map((_, i) => (
                        <Cell key={i} fill={CLASS_PIE_COLORS[i % CLASS_PIE_COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            )
          })}
        </div>
      </div>

      {/* Footer */}
      <p className="text-center text-xs text-slate-500 py-4">
        <strong>Research Demonstration Only.</strong> Built for academic purposes at ESOFT Metro Campus.
        Must not be used for clinical diagnosis. Always consult a qualified radiologist.
      </p>
    </div>
  )
}
