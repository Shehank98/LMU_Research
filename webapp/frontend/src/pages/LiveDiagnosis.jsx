import { useState, useRef, useCallback } from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import ConfidenceBadge from '../components/ConfidenceBadge'
import SectionHeader from '../components/SectionHeader'

const CLASS_COLORS = { Glioma: '#e74c3c', Meningioma: '#e67e22', 'No Tumor': '#27ae60', Pituitary: '#2980b9' }
const IOULabels = { cnn_vs_xception: 'CNN vs Xception', cnn_vs_inception: 'CNN vs InceptionV3', xception_vs_inception: 'Xception vs InceptionV3' }

function DropZone({ onFile }) {
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef()

  const handle = useCallback(file => {
    if (file && (file.type === 'image/jpeg' || file.type === 'image/png')) onFile(file)
  }, [onFile])

  return (
    <div
      onDragOver={e => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={e => { e.preventDefault(); setDragging(false); handle(e.dataTransfer.files[0]) }}
      onClick={() => inputRef.current.click()}
      className={`border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-colors ${
        dragging ? 'border-teal-400 bg-teal-400/10' : 'border-slate-600 hover:border-teal-500 hover:bg-slate-800/50'
      }`}
    >
      <input ref={inputRef} type="file" accept="image/jpeg,image/png" className="hidden"
        onChange={e => handle(e.target.files[0])} />
      <div className="text-4xl mb-3">🧬</div>
      <p className="text-slate-300 font-medium">Drag & drop or click to upload MRI scan</p>
      <p className="text-slate-500 text-sm mt-1">JPEG or PNG · Any resolution</p>
    </div>
  )
}

function IoUMeter({ score, label }) {
  const level = score >= 0.65 ? 'strong' : score >= 0.40 ? 'moderate' : 'low'
  const color = level === 'strong' ? 'text-green-400' : level === 'moderate' ? 'text-amber-400' : 'text-red-400'
  const barColor = level === 'strong' ? 'bg-green-500' : level === 'moderate' ? 'bg-amber-500' : 'bg-red-500'
  return (
    <div className="bg-slate-700/50 rounded-lg p-3">
      <div className="text-xs text-slate-400 mb-1">{label}</div>
      <div className={`text-xl font-extrabold ${color}`}>{score.toFixed(3)}</div>
      <div className="mt-1.5 h-1.5 bg-slate-600 rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${barColor}`} style={{ width: `${score * 100}%` }} />
      </div>
      <div className={`text-xs mt-1 capitalize ${color}`}>{level}</div>
    </div>
  )
}

export default function LiveDiagnosis() {
  const [imageURL, setImageURL] = useState(null)
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [detailOpen, setDetailOpen] = useState(false)

  const handleFile = f => {
    setFile(f)
    setImageURL(URL.createObjectURL(f))
    setResult(null)
    setError(null)
  }

  const runInference = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    try {
      const form = new FormData()
      form.append('file', file)
      const res = await fetch('/api/predict', { method: 'POST', body: form })
      if (!res.ok) {
        const msg = await res.json().catch(() => ({ detail: 'Inference failed' }))
        throw new Error(msg.detail || `HTTP ${res.status}`)
      }
      setResult(await res.json())
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const reset = () => { setFile(null); setImageURL(null); setResult(null); setError(null) }

  const classColor = result ? (CLASS_COLORS[result.final_class] || '#00b4d8') : '#00b4d8'

  const probData = result
    ? (result.class_names || []).map((name, i) => ({
        name,
        value: (result.class_probabilities?.[i] || 0) * 100,
        color: CLASS_COLORS[name] || '#64748b',
      }))
    : []

  const perModelData = result?.per_model
    ? [...result.per_model].sort((a, b) => b.confidence - a.confidence).map(m => ({
        name: m.name.replace(' + Ensemble', '+Ens').replace('InceptionV3', 'Inc'),
        confidence: (m.confidence * 100).toFixed(1),
        predicted: m.predicted_class,
        color: CLASS_COLORS[m.predicted_class] || '#64748b',
      }))
    : []

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-2xl font-extrabold text-white">🔬 Live MRI Diagnosis</h2>
        <p className="text-slate-400 mt-1">Upload a brain MRI scan for ensemble prediction with GRAD-CAM explainability</p>
      </div>

      {!file ? (
        <DropZone onFile={handleFile} />
      ) : (
        <div className="grid lg:grid-cols-2 gap-6">
          <div className="card-bordered">
            <SectionHeader title="Original MRI" />
            <img src={imageURL} alt="Uploaded MRI" className="w-full rounded-lg object-contain max-h-64" />
            <div className="flex gap-3 mt-4">
              {!result && !loading && (
                <button onClick={runInference} className="btn-primary flex-1">
                  ▶ Run Ensemble Analysis
                </button>
              )}
              <button onClick={reset} className="btn-secondary">
                ↩ New Image
              </button>
            </div>
          </div>

          <div className="card-bordered">
            <SectionHeader title="Prediction" />
            {loading && (
              <div className="flex flex-col items-center justify-center py-12 gap-4">
                <div className="w-10 h-10 border-4 border-teal-400 border-t-transparent rounded-full animate-spin" />
                <p className="text-slate-400 text-sm">Running 6-model ensemble inference…</p>
              </div>
            )}
            {error && (
              <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4 text-red-400">
                <strong>Error:</strong> {error}
                {error.includes('No models') && (
                  <p className="text-sm mt-2 text-red-300">
                    Configure <code className="bg-red-500/20 px-1 rounded">HF_REPO_ID</code> and
                    <code className="bg-red-500/20 px-1 rounded ml-1">HF_TOKEN</code> env vars on Railway.
                  </p>
                )}
              </div>
            )}
            {result && (
              <div className="space-y-4">
                <div>
                  <div className="text-4xl font-extrabold" style={{ color: classColor }}>
                    {result.final_class}
                  </div>
                  <div className="text-slate-400 text-sm mt-1">Ensemble prediction</div>
                </div>
                <div>
                  <div className="flex items-center justify-between text-sm mb-1">
                    <span className="text-slate-400">Confidence</span>
                    <span className="font-mono font-bold text-white">{(result.confidence * 100).toFixed(1)}%</span>
                  </div>
                  <div className="h-3 bg-slate-700 rounded-full overflow-hidden">
                    <div className="h-full rounded-full transition-all duration-700"
                      style={{ width: `${result.confidence * 100}%`, backgroundColor: classColor }} />
                  </div>
                </div>
                <ConfidenceBadge level={result.confidence_level} pct={result.confidence * 100} />
                {result.confidence_level === 'high' && (
                  <p className="text-green-400 text-sm">✔ Models strongly agree on this prediction.</p>
                )}
                {result.confidence_level === 'moderate' && (
                  <p className="text-amber-400 text-sm">⚠ Moderate agreement. Radiologist review recommended.</p>
                )}
                {result.confidence_level === 'low' && (
                  <p className="text-red-400 text-sm">✖ Low confidence. Expert review required.</p>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {result && (
        <>
          {/* Class probability chart */}
          <div className="card-bordered">
            <SectionHeader title="Class Probability Distribution" />
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={probData} margin={{ left: 10, right: 40, top: 5, bottom: 5 }}>
                <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 12 }} />
                <YAxis tickFormatter={v => `${v.toFixed(0)}%`} tick={{ fill: '#94a3b8', fontSize: 11 }} />
                <Tooltip formatter={v => [`${Number(v).toFixed(1)}%`, 'Probability']}
                  contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }} />
                <Bar dataKey="value" radius={[4, 4, 0, 0]}
                  label={{ position: 'top', formatter: v => `${Number(v).toFixed(1)}%`, fill: '#94a3b8', fontSize: 11 }}>
                  {probData.map((d, i) => <Cell key={i} fill={d.color} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* GRAD-CAM heatmaps */}
          {(result.heatmaps && Object.keys(result.heatmaps).length > 0) && (
            <div className="card-bordered">
              <SectionHeader title="GRAD-CAM Heatmaps"
                subtitle="Red = high attention · Blue = low attention" />
              <div className={`grid gap-4 ${
                Object.keys(result.heatmaps).length + (result.consensus_heatmap ? 1 : 0) >= 3
                  ? 'grid-cols-2 lg:grid-cols-4'
                  : 'grid-cols-2'
              }`}>
                {Object.entries(result.heatmaps).map(([key, b64]) => (
                  <div key={key} className="text-center">
                    <p className="text-xs text-slate-400 font-medium mb-1 uppercase tracking-wide">
                      {key === 'cnn' ? 'CNN' : key === 'xception' ? 'Xception' : 'InceptionV3'}
                    </p>
                    <img src={b64} alt={key} className="w-full rounded-lg" />
                  </div>
                ))}
                {result.consensus_heatmap && (
                  <div className="text-center">
                    <p className="text-xs text-teal-400 font-bold mb-1 uppercase tracking-wide">Consensus</p>
                    <img src={result.consensus_heatmap} alt="consensus" className="w-full rounded-lg ring-2 ring-teal-400/40" />
                    <p className="text-xs mt-1 font-mono text-teal-400">
                      EAA-IoU: {result.average_iou?.toFixed(3)}
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Pairwise IoU */}
          {result.iou_scores && Object.keys(result.iou_scores).length > 0 && (
            <div className="card-bordered">
              <SectionHeader title="Pairwise IoU Breakdown"
                subtitle="Inter-model attention agreement (EAA-IoU) — ≥0.65 strong · 0.40–0.65 moderate · <0.40 low" />
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                {Object.entries(result.iou_scores).map(([key, score]) => (
                  <IoUMeter key={key} score={score}
                    label={IOULabels[key] || key.replace(/_vs_/g, ' vs ')} />
                ))}
                <IoUMeter score={result.average_iou || 0} label="Average EAA-IoU" />
              </div>
            </div>
          )}

          {/* Per-model votes */}
          {perModelData.length > 0 && (
            <div className="card-bordered">
              <SectionHeader title="Per-Model Vote Breakdown"
                subtitle="Top-class confidence per model" />
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={perModelData} layout="vertical"
                  margin={{ left: 10, right: 60, top: 5, bottom: 5 }}>
                  <XAxis type="number" domain={[0, 105]} tickFormatter={v => `${v}%`}
                    tick={{ fill: '#94a3b8', fontSize: 11 }} />
                  <YAxis type="category" dataKey="name" width={120}
                    tick={{ fill: '#94a3b8', fontSize: 11 }} />
                  <Tooltip
                    formatter={(v, _, props) => [`${v}% → ${props.payload.predicted}`, 'Confidence']}
                    contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }} />
                  <Bar dataKey="confidence" radius={[0, 4, 4, 0]}
                    label={{ position: 'right', formatter: v => `${v}%`, fill: '#94a3b8', fontSize: 11 }}>
                    {perModelData.map((d, i) => <Cell key={i} fill={d.color} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Detailed table */}
          <div className="card-bordered">
            <button
              onClick={() => setDetailOpen(o => !o)}
              className="flex items-center justify-between w-full text-left"
            >
              <SectionHeader title="Detailed Probability Table" />
              <span className="text-slate-400 text-sm mb-5">{detailOpen ? '▲' : '▼'}</span>
            </button>
            {detailOpen && (
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="text-slate-400 border-b border-slate-700">
                      <th className="text-left py-2 pr-3">Model</th>
                      <th className="text-left py-2 pr-3">Predicted</th>
                      {(result.class_names || []).map(c => (
                        <th key={c} className="text-right py-2 px-1">{c}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {result.per_model?.map(m => (
                      <tr key={m.name} className="border-b border-slate-700/50">
                        <td className="py-2 pr-3 text-slate-300">{m.name}</td>
                        <td className="py-2 pr-3 font-medium" style={{ color: CLASS_COLORS[m.predicted_class] }}>
                          {m.predicted_class}
                        </td>
                        {m.probabilities.map((p, i) => (
                          <td key={i} className="py-2 px-1 text-right font-mono text-slate-300">
                            {(p * 100).toFixed(1)}%
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </>
      )}

      <p className="text-xs text-slate-500 text-center">
        ⚠ <strong>Research Demonstration Only.</strong> Not for clinical use. Always consult a qualified radiologist.
      </p>
    </div>
  )
}
