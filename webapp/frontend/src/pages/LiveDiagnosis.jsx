import { useState, useRef, useCallback, useEffect } from 'react'
import ConfidenceBadge from '../components/ConfidenceBadge'

/* ── Constants ────────────────────────────────────────────────────────────── */
const CLASS_COLORS = {
  Glioma:     '#ef4444',
  Meningioma: '#f97316',
  'No Tumor': '#22c55e',
  Pituitary:  '#3b82f6',
}

const TUMOR_INFO = {
  Glioma: {
    icdCode: 'C71',
    urgency: 'high',
    urgencyLabel: 'HIGH PRIORITY',
    description:
      'Gliomas arise from glial cells within the brain or spinal cord. As the most common primary brain tumour, gliomas range from low-grade (slow-growing, WHO Grade I–II) to high-grade (rapidly progressive, Grade III–IV including glioblastoma multiforme). Presentation may include seizures, headache, focal neurological deficits, and progressive cognitive changes.',
    recommendation:
      'Immediate neurosurgical and neuro-oncology consultation recommended. Advanced MRI with gadolinium contrast, MR spectroscopy, and perfusion imaging may be indicated for further characterisation and grading. Biopsy or surgical resection is typically required for definitive diagnosis.',
  },
  Meningioma: {
    icdCode: 'D32',
    urgency: 'moderate',
    urgencyLabel: 'MODERATE PRIORITY',
    description:
      'Meningiomas originate from the arachnoid cells of the meninges surrounding the brain and spinal cord. The majority are benign (WHO Grade I) and slow-growing. Atypical (Grade II) and malignant (Grade III) variants exist. Common locations include the cerebral convexity, sphenoid wing, and olfactory groove. Symptoms depend on size and location.',
    recommendation:
      'Neurosurgical evaluation recommended. Watchful waiting with serial imaging may be appropriate for small, asymptomatic lesions. Surgical resection is considered for symptomatic or enlarging tumours. Stereotactic radiosurgery may be an alternative for select cases.',
  },
  'No Tumor': {
    icdCode: 'Z03.89',
    urgency: 'normal',
    urgencyLabel: 'NO ABNORMALITY DETECTED',
    description:
      'No intracranial mass lesion was detected in this MRI scan. The 6-model ensemble did not identify features characteristic of glioma, meningioma, or pituitary adenoma. Structural brain anatomy appears within normal limits for the regions assessed by the model architectures used.',
    recommendation:
      "Clinical correlation is required. Results must be interpreted alongside the patient's symptoms, clinical history, and assessment by a qualified radiologist. A negative AI prediction does not exclude pathology — the model has not been validated for all neurological conditions.",
  },
  Pituitary: {
    icdCode: 'D35.2',
    urgency: 'moderate',
    urgencyLabel: 'MODERATE PRIORITY',
    description:
      'Pituitary adenomas are typically benign epithelial neoplasms of the anterior pituitary gland. Classified as microadenomas (<10 mm) or macroadenomas (≥10 mm). May be functioning (hormone-secreting) or non-functioning. Common presentations include bitemporal hemianopia from optic chiasm compression, headache, and endocrine disturbances.',
    recommendation:
      'Endocrinology consultation for hormonal evaluation (GH, prolactin, TSH, ACTH, cortisol). Neurosurgical assessment recommended for macroadenomas or tumours causing mass effect. Dedicated pituitary protocol MRI with gadolinium contrast is advised for definitive characterisation.',
  },
}

const URGENCY_STYLES = {
  high:     { bg: 'bg-red-500/10',    border: 'border-red-500/40',    text: 'text-red-400',    dot: 'bg-red-500' },
  moderate: { bg: 'bg-orange-500/10', border: 'border-orange-500/40', text: 'text-orange-400', dot: 'bg-orange-500' },
  normal:   { bg: 'bg-green-500/10',  border: 'border-green-500/40',  text: 'text-green-400',  dot: 'bg-green-500' },
}

const HEATMAP_LABELS = {
  cnn:      'CNN',
  xception: 'Xception',
  inception:'InceptionV3',
  cnn_feat: 'CNN Features',
  inc_feat: 'Inc Features',
  xcp_feat: 'Xcp Features',
}

const IOU_LABELS = {
  cnn_vs_xception:            'CNN ↔ Xception',
  cnn_vs_inception:           'CNN ↔ InceptionV3',
  xception_vs_inception:      'Xception ↔ InceptionV3',
  cnn_vs_cnn_feat:            'CNN ↔ CNN Feat',
  cnn_vs_inc_feat:            'CNN ↔ Inc Feat',
  cnn_vs_xcp_feat:            'CNN ↔ Xcp Feat',
  xception_vs_cnn_feat:       'Xception ↔ CNN Feat',
  xception_vs_inc_feat:       'Xception ↔ Inc Feat',
  xception_vs_xcp_feat:       'Xception ↔ Xcp Feat',
  inception_vs_cnn_feat:      'InceptionV3 ↔ CNN Feat',
  inception_vs_inc_feat:      'InceptionV3 ↔ Inc Feat',
  inception_vs_xcp_feat:      'InceptionV3 ↔ Xcp Feat',
  cnn_feat_vs_inc_feat:       'CNN Feat ↔ Inc Feat',
  cnn_feat_vs_xcp_feat:       'CNN Feat ↔ Xcp Feat',
  inc_feat_vs_xcp_feat:       'Inc Feat ↔ Xcp Feat',
}

/* ── Sub-components ───────────────────────────────────────────────────────── */
function UploadZone({ onFile }) {
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef()

  const handle = useCallback(
    file => {
      if (file && (file.type === 'image/jpeg' || file.type === 'image/png')) onFile(file)
    },
    [onFile],
  )

  return (
    <div className="flex flex-col items-center justify-center py-10 sm:py-16 gap-7 px-2">
      <div className="text-center space-y-2">
        <div className="inline-flex items-center gap-2 bg-teal-500/10 border border-teal-500/20 rounded-full px-4 py-1.5 mb-1">
          <span className="w-1.5 h-1.5 rounded-full bg-teal-400 animate-pulse" />
          <span className="text-teal-400 text-xs font-semibold tracking-widest uppercase">
            AI Diagnostic System — Research Demo
          </span>
        </div>
        <h3 className="text-xl sm:text-2xl font-bold text-white">Upload Brain MRI Scan</h3>
        <p className="text-slate-400 text-sm">Axial · Sagittal · Coronal · JPEG or PNG</p>
      </div>

      <div
        onDragOver={e => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={e => { e.preventDefault(); setDragging(false); handle(e.dataTransfer.files[0]) }}
        onClick={() => inputRef.current.click()}
        className={`w-full max-w-md border-2 border-dashed rounded-2xl p-10 sm:p-14 text-center cursor-pointer
          transition-all duration-200 ${
            dragging
              ? 'border-teal-400 bg-teal-400/10 scale-[1.02]'
              : 'border-slate-600 hover:border-teal-500 hover:bg-slate-800/40'
          }`}
      >
        <input
          ref={inputRef} type="file" accept="image/jpeg,image/png"
          className="hidden" onChange={e => handle(e.target.files[0])}
        />
        {/* Upload icon */}
        <div className="w-14 h-14 mx-auto mb-4 rounded-2xl bg-slate-700/60 flex items-center justify-center">
          <svg className="w-8 h-8 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round"
              d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
          </svg>
        </div>
        <p className="text-slate-200 font-semibold">Drag MRI scan here or tap to browse</p>
        <p className="text-slate-500 text-sm mt-1">JPEG · PNG · Any resolution</p>
      </div>

      {/* Model chips */}
      <div className="flex flex-wrap justify-center gap-2 max-w-lg">
        <span className="text-slate-600 text-xs self-center">Powered by:</span>
        {['CNN', 'InceptionV3', 'Xception', 'SVM', 'Random Forest', 'Decision Tree'].map(m => (
          <span key={m}
            className="px-2.5 py-1 bg-slate-800 border border-slate-700 rounded-full text-xs text-slate-400 font-mono">
            {m}
          </span>
        ))}
      </div>
    </div>
  )
}

function SegmentBar({ value, color }) {
  const total = 20
  const filled = Math.round((value / 100) * total)
  return (
    <div className="flex gap-0.5">
      {Array.from({ length: total }).map((_, i) => (
        <div key={i}
          className="h-2.5 flex-1 rounded-sm transition-all duration-700"
          style={i < filled ? { backgroundColor: color } : { backgroundColor: '#334155' }}
        />
      ))}
    </div>
  )
}

function ProbBar({ label, value, color, isMax }) {
  return (
    <div className={`rounded-lg px-3 py-2.5 transition-colors ${isMax ? 'bg-slate-700/60' : 'bg-slate-800/30'}`}>
      <div className="flex justify-between items-center mb-1.5">
        <span className={`text-sm font-medium ${isMax ? 'text-white' : 'text-slate-400'}`}>{label}</span>
        <span className={`text-sm font-mono font-bold ${isMax ? 'text-white' : 'text-slate-500'}`}>
          {value.toFixed(1)}%
        </span>
      </div>
      <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{ width: `${value}%`, backgroundColor: isMax ? color : '#475569' }}
        />
      </div>
    </div>
  )
}

function IoUAgreement({ score }) {
  if (score == null) return null
  const level = score >= 0.65 ? 'strong' : score >= 0.40 ? 'moderate' : 'low'
  const map = {
    strong:   { text: 'text-green-400',  border: 'border-green-500/30',  bg: 'bg-green-500/10',  label: 'Models strongly agree on tumour location' },
    moderate: { text: 'text-amber-400',  border: 'border-amber-500/30',  bg: 'bg-amber-500/10',  label: 'Moderate inter-model attention agreement' },
    low:      { text: 'text-red-400',    border: 'border-red-500/30',    bg: 'bg-red-500/10',    label: 'Low agreement — radiologist review advised' },
  }
  const s = map[level]
  return (
    <div className={`flex items-center gap-3 rounded-xl border px-4 py-3 ${s.bg} ${s.border}`}>
      <div className="text-center shrink-0">
        <div className={`text-2xl font-black font-mono leading-none ${s.text}`}>{score.toFixed(3)}</div>
        <div className={`text-xs mt-0.5 ${s.text}`}>EAA-IoU</div>
      </div>
      <div>
        <div className={`font-semibold text-sm ${s.text}`}>{s.label}</div>
        <div className="text-xs text-slate-500 mt-0.5">
          {score >= 0.65 ? '≥0.65 threshold — high spatial consensus' :
           score >= 0.40 ? '0.40–0.65 — partial agreement' :
           '<0.40 — escalation recommended'}
        </div>
      </div>
    </div>
  )
}

function ModelStatusPill({ modelStatus }) {
  if (!modelStatus) return null
  const full = modelStatus.loaded_count === modelStatus.total_count
  const none = modelStatus.loaded_count === 0
  return (
    <div className={`inline-flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full border font-medium ${
      full ? 'bg-green-500/10 border-green-500/30 text-green-400' :
      none ? 'bg-red-500/10 border-red-500/30 text-red-400' :
             'bg-amber-500/10 border-amber-500/30 text-amber-400'
    }`}>
      <span className={`w-1.5 h-1.5 rounded-full ${full ? 'bg-green-400' : none ? 'bg-red-400' : 'bg-amber-400'}`} />
      {modelStatus.loaded_count}/{modelStatus.total_count} Models
    </div>
  )
}

/* ── Main page ────────────────────────────────────────────────────────────── */
export default function LiveDiagnosis() {
  const [file, setFile]           = useState(null)
  const [imageURL, setImageURL]   = useState(null)
  const [loading, setLoading]     = useState(false)
  const [result, setResult]       = useState(null)
  const [error, setError]         = useState(null)
  const [modelStatus, setModelStatus] = useState(null)
  const [activeTab, setActiveTab] = useState('heatmaps')
  const reportRef = useRef()

  useEffect(() => {
    fetch('/api/models/status').then(r => r.json()).then(d => setModelStatus(d)).catch(() => {})
  }, [])

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
      const data = await res.json()
      setResult(data)
      setActiveTab('heatmaps')
      setTimeout(() => reportRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 150)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const reset = () => { setFile(null); setImageURL(null); setResult(null); setError(null) }

  const info         = result ? TUMOR_INFO[result.final_class] : null
  const urgStyle     = info   ? URGENCY_STYLES[info.urgency]   : null
  const classColor   = result ? (CLASS_COLORS[result.final_class] || '#14b8a6') : '#14b8a6'
  const confPct      = result ? result.confidence * 100 : 0

  const probData = result
    ? (result.class_names || [])
        .map((name, i) => ({
          name, value: (result.class_probabilities?.[i] || 0) * 100,
          color: CLASS_COLORS[name] || '#64748b', isMax: name === result.final_class,
        }))
        .sort((a, b) => b.value - a.value)
    : []

  const reportTime = new Date().toLocaleString('en-GB', {
    day: '2-digit', month: 'short', year: 'numeric',
    hour: '2-digit', minute: '2-digit', hour12: false,
  })

  /* ── render ─────────────────────────────────────────────────────────────── */
  return (
    <div className="space-y-5">

      {/* ── Page title bar ── */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-2xl font-extrabold text-white">Live Diagnosis</h2>
          <p className="text-slate-400 text-sm mt-0.5">
            6-model ensemble · GRAD-CAM explainability · EAA-IoU consensus
          </p>
        </div>
        <ModelStatusPill modelStatus={modelStatus} />
      </div>

      {/* ── Upload view ── */}
      {!file && <UploadZone onFile={handleFile} />}

      {/* ── Report view ── */}
      {file && (
        <div className="space-y-5" ref={reportRef}>

          {/* Report header bar */}
          <div className="flex flex-wrap items-center justify-between gap-3
            bg-slate-800/60 border border-slate-700 rounded-xl px-4 py-3">
            <div className="flex items-center gap-3 min-w-0">
              <div className="w-8 h-8 shrink-0 rounded-lg bg-teal-500/20 flex items-center justify-center">
                <svg className="w-4 h-4 text-teal-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round"
                    d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              </div>
              <div className="min-w-0">
                <div className="text-white font-semibold text-sm">AI Diagnostic Report</div>
                <div className="text-slate-500 text-xs truncate">
                  {reportTime}
                  {result && (
                    <span className="ml-2 font-mono text-slate-600 hidden sm:inline">· {file?.name}</span>
                  )}
                </div>
              </div>
            </div>
            <button onClick={reset} className="btn-secondary text-xs px-3 py-1.5 shrink-0">
              New Scan
            </button>
          </div>

          {/* Main grid: scan panel (2 cols) + diagnosis (3 cols) */}
          <div className="grid lg:grid-cols-5 gap-5">

            {/* ── Left: Scan panel ── */}
            <div className="lg:col-span-2 card-bordered flex flex-col gap-4">
              <div>
                <p className="text-xs text-slate-500 uppercase tracking-wider font-semibold mb-2">
                  Input MRI Scan
                </p>
                <img
                  src={imageURL} alt="MRI scan"
                  className="w-full rounded-lg object-contain bg-black max-h-64 lg:max-h-72"
                />
              </div>

              {/* File metadata */}
              {file && (
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="bg-slate-700/30 rounded-lg p-2">
                    <p className="text-slate-500 mb-0.5">File</p>
                    <p className="text-slate-300 truncate font-mono" title={file.name}>{file.name}</p>
                  </div>
                  <div className="bg-slate-700/30 rounded-lg p-2">
                    <p className="text-slate-500 mb-0.5">Size</p>
                    <p className="text-slate-300 font-mono">{(file.size / 1024).toFixed(1)} KB</p>
                  </div>
                </div>
              )}

              {!result && !loading && (
                <button
                  onClick={runInference}
                  disabled={modelStatus?.loaded_count === 0}
                  className="btn-primary w-full text-sm"
                >
                  {modelStatus?.loaded_count > 0
                    ? `Run ${modelStatus.loaded_count}-Model Ensemble`
                    : 'Run Analysis'}
                </button>
              )}

              {loading && (
                <div className="flex flex-col items-center justify-center py-5 gap-3">
                  <div className="flex gap-1">
                    {[0, 1, 2, 3, 4].map(i => (
                      <div key={i}
                        className="w-1.5 h-5 bg-teal-400 rounded-full animate-bounce"
                        style={{ animationDelay: `${i * 0.1}s` }} />
                    ))}
                  </div>
                  <p className="text-slate-400 text-sm text-center">
                    Analysing across {modelStatus?.loaded_count ?? 6} models…
                  </p>
                </div>
              )}

              {error && (
                <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3 text-red-400 text-sm">
                  <strong>Error:</strong> {error}
                </div>
              )}

              {result && (
                <p className="text-xs text-slate-500 text-center">
                  Analysis complete · {result.models_used} model{result.models_used !== 1 ? 's' : ''} consulted
                </p>
              )}
            </div>

            {/* ── Right: Diagnosis panel ── */}
            <div className="lg:col-span-3 space-y-4">

              {!result && !loading && !error && (
                <div className="card-bordered flex items-center justify-center min-h-[220px]">
                  <div className="text-center space-y-2">
                    <div className="w-12 h-12 mx-auto rounded-2xl bg-slate-700/50 flex items-center justify-center">
                      <svg className="w-6 h-6 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                        <path strokeLinecap="round" strokeLinejoin="round"
                          d="M9.75 9.75l4.5 4.5m0-4.5l-4.5 4.5M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                    </div>
                    <p className="text-slate-500 text-sm">Press "Run Ensemble" to analyse the scan</p>
                  </div>
                </div>
              )}

              {loading && (
                <div className="card-bordered flex items-center justify-center min-h-[220px]">
                  <p className="text-slate-500 text-sm animate-pulse">Running inference…</p>
                </div>
              )}

              {result && (
                <>
                  {/* ── Primary diagnosis card ── */}
                  <div className="card-bordered border-l-4" style={{ borderLeftColor: classColor }}>
                    <div className="flex flex-wrap items-start justify-between gap-3 mb-5">
                      <div>
                        <p className="text-xs text-slate-500 uppercase tracking-wider font-semibold mb-1">
                          Primary Diagnosis
                        </p>
                        <h3 className="text-4xl sm:text-5xl font-black leading-none" style={{ color: classColor }}>
                          {result.final_class}
                        </h3>
                        {info?.icdCode && (
                          <span className="text-xs text-slate-500 font-mono mt-1 block">
                            ICD-10: {info.icdCode}
                          </span>
                        )}
                      </div>
                      <ConfidenceBadge level={result.confidence_level} pct={confPct} />
                    </div>

                    {/* Confidence */}
                    <div className="space-y-2 mb-4">
                      <div className="flex justify-between items-center text-sm">
                        <span className="text-slate-400 font-medium">Ensemble Confidence</span>
                        <span className="font-mono font-bold text-white text-base">{confPct.toFixed(1)}%</span>
                      </div>
                      <SegmentBar value={confPct} color={classColor} />
                      <div className="flex justify-between text-xs text-slate-600">
                        <span>0%</span><span>50%</span><span>100%</span>
                      </div>
                    </div>

                    {/* EAA-IoU */}
                    {result.average_iou != null && (
                      <IoUAgreement score={result.average_iou} />
                    )}

                    <p className="text-xs text-slate-600 mt-3">
                      {result.models_used} of {result.models_used + (result.models_missing?.length || 0)} models consulted
                      {result.models_missing?.length > 0 && (
                        <span className="text-amber-600 ml-1">· {result.models_missing.length} pending</span>
                      )}
                    </p>
                  </div>

                  {/* ── Urgency / recommendation ── */}
                  {info && urgStyle && (
                    <div className={`rounded-xl border p-4 ${urgStyle.bg} ${urgStyle.border}`}>
                      <div className="flex items-center gap-2 mb-2">
                        <span className={`w-2 h-2 rounded-full shrink-0 ${urgStyle.dot}`} />
                        <span className={`text-xs font-bold tracking-widest uppercase ${urgStyle.text}`}>
                          {info.urgencyLabel}
                        </span>
                      </div>
                      <p className="text-slate-300 text-sm leading-relaxed">{info.recommendation}</p>
                    </div>
                  )}

                  {/* ── Probability distribution ── */}
                  <div className="card-bordered">
                    <p className="text-xs text-slate-500 uppercase tracking-wider font-semibold mb-3">
                      Class Probability Distribution
                    </p>
                    <div className="space-y-2">
                      {probData.map(d => (
                        <ProbBar key={d.name} label={d.name} value={d.value}
                          color={d.color} isMax={d.isMax} />
                      ))}
                    </div>
                  </div>
                </>
              )}
            </div>
          </div>

          {/* ── Detail tabs — only after prediction ── */}
          {result && (
            <>
              <div className="card-bordered">
                {/* Tab bar */}
                <div className="flex gap-1 border-b border-slate-700 mb-5 -mx-5 px-5 overflow-x-auto">
                  {[
                    { key: 'heatmaps', label: 'GRAD-CAM Heatmaps' },
                    { key: 'models',   label: 'Per-Model Votes' },
                    { key: 'iou',      label: 'EAA-IoU Breakdown' },
                  ].map(tab => (
                    <button
                      key={tab.key}
                      onClick={() => setActiveTab(tab.key)}
                      className={`px-4 py-3 text-sm font-medium whitespace-nowrap border-b-2 -mb-px transition-colors ${
                        activeTab === tab.key
                          ? 'border-teal-400 text-teal-400'
                          : 'border-transparent text-slate-500 hover:text-slate-300'
                      }`}
                    >
                      {tab.label}
                    </button>
                  ))}
                </div>

                {/* ── GRAD-CAM tab ── */}
                {activeTab === 'heatmaps' && (
                  <div>
                    <p className="text-xs text-slate-500 mb-4 leading-relaxed">
                      Class Activation Maps (GRAD-CAM) show which regions each model attended to when making its prediction.{' '}
                      <span className="text-red-400">Red</span> = high attention ·{' '}
                      <span className="text-blue-400">Blue</span> = low attention.
                      The consensus map is a weighted average of all three heatmaps.
                    </p>
                    {result.heatmaps && Object.keys(result.heatmaps).length > 0 ? (
                      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-7 gap-4">
                        {Object.entries(result.heatmaps).map(([key, b64]) => (
                          <div key={key}>
                            <p className="text-xs text-slate-400 font-semibold mb-2 uppercase tracking-wide text-center">
                              {HEATMAP_LABELS[key] || key}
                            </p>
                            <img src={b64} alt={key}
                              className="w-full rounded-lg border border-slate-700" />
                          </div>
                        ))}
                        {result.consensus_heatmap && (
                          <div>
                            <p className="text-xs text-teal-400 font-bold mb-2 uppercase tracking-wide text-center">
                              Consensus
                            </p>
                            <img src={result.consensus_heatmap} alt="consensus"
                              className="w-full rounded-lg ring-2 ring-teal-400/50" />
                            <p className="text-xs mt-1.5 font-mono text-teal-400 text-center">
                              EAA-IoU: {result.average_iou?.toFixed(3)}
                            </p>
                          </div>
                        )}
                      </div>
                    ) : (
                      <div className="text-center py-8">
                        <p className="text-slate-500 text-sm">GRAD-CAM unavailable — model files not loaded</p>
                        <p className="text-slate-600 text-xs mt-1">Upload model files to HuggingFace to enable heatmaps</p>
                      </div>
                    )}
                  </div>
                )}

                {/* ── Per-model votes tab ── */}
                {activeTab === 'models' && (
                  <div className="overflow-x-auto -mx-1">
                    <table className="w-full text-sm min-w-[480px]">
                      <thead>
                        <tr className="text-slate-500 border-b border-slate-700 text-xs uppercase tracking-wider">
                          <th className="text-left py-2.5 pr-4 pl-1">Model</th>
                          <th className="text-left py-2.5 pr-4">Prediction</th>
                          <th className="text-right py-2.5 pr-4">Confidence</th>
                          {(result.class_names || []).map(c => (
                            <th key={c} className="text-right py-2.5 px-2 hidden sm:table-cell">{c}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {result.per_model?.map(m => (
                          <tr key={m.name}
                            className="border-b border-slate-700/40 hover:bg-slate-700/20 transition-colors">
                            <td className="py-3 pr-4 pl-1 text-slate-300 font-medium text-sm">{m.name}</td>
                            <td className="py-3 pr-4">
                              <span className="font-semibold text-sm"
                                style={{ color: CLASS_COLORS[m.predicted_class] || '#64748b' }}>
                                {m.predicted_class}
                              </span>
                            </td>
                            <td className="py-3 pr-4 text-right">
                              <div className="flex items-center justify-end gap-2">
                                <div className="w-16 h-1.5 bg-slate-700 rounded-full overflow-hidden hidden sm:block">
                                  <div className="h-full rounded-full"
                                    style={{
                                      width: `${m.confidence * 100}%`,
                                      backgroundColor: CLASS_COLORS[m.predicted_class] || '#64748b',
                                    }} />
                                </div>
                                <span className="font-mono text-sm text-slate-300 w-12 text-right">
                                  {(m.confidence * 100).toFixed(1)}%
                                </span>
                              </div>
                            </td>
                            {m.probabilities.map((p, i) => (
                              <td key={i} className="py-3 px-2 text-right font-mono text-xs text-slate-500 hidden sm:table-cell">
                                {(p * 100).toFixed(1)}%
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    <p className="text-xs text-slate-600 mt-3 px-1">
                      Full probability breakdown visible on wider screens (≥sm).
                    </p>
                  </div>
                )}

                {/* ── EAA-IoU tab ── */}
                {activeTab === 'iou' && (
                  <div>
                    <p className="text-xs text-slate-500 mb-5 leading-relaxed">
                      <strong className="text-slate-400">Ensemble Attention Agreement (EAA-IoU)</strong> — Novel contribution.
                      Measures pairwise spatial overlap between each model's binarised attention map.
                      High agreement (≥0.65) indicates the models consistently localise the same brain region,
                      adding confidence to the prediction. Low agreement (&lt;0.40) suggests uncertainty and
                      is used as a clinical escalation trigger.
                    </p>
                    {result.iou_scores && Object.keys(result.iou_scores).length > 0 ? (
                      <div className="space-y-3">
                        {Object.entries(result.iou_scores).map(([key, score]) => {
                          const level  = score >= 0.65 ? 'strong' : score >= 0.40 ? 'moderate' : 'low'
                          const color  = level === 'strong' ? '#22c55e' : level === 'moderate' ? '#f59e0b' : '#ef4444'
                          const label  = IOU_LABELS[key] || key.replace(/_vs_/g, ' ↔ ')
                          return (
                            <div key={key} className="flex items-center gap-4">
                              <span className="text-sm text-slate-400 w-48 shrink-0">{label}</span>
                              <div className="flex-1 h-2 bg-slate-700 rounded-full overflow-hidden">
                                <div className="h-full rounded-full transition-all duration-700"
                                  style={{ width: `${score * 100}%`, backgroundColor: color }} />
                              </div>
                              <span className="font-mono text-sm font-bold w-14 text-right"
                                style={{ color }}>
                                {score.toFixed(3)}
                              </span>
                            </div>
                          )
                        })}

                        {/* Average row */}
                        <div className="border-t border-slate-700 pt-3 flex items-center gap-4">
                          <span className="text-sm font-semibold text-white w-48 shrink-0">Average EAA-IoU</span>
                          <div className="flex-1 h-2.5 bg-slate-700 rounded-full overflow-hidden">
                            <div className="h-full rounded-full transition-all duration-700 bg-teal-400"
                              style={{ width: `${(result.average_iou || 0) * 100}%` }} />
                          </div>
                          <span className="font-mono text-sm font-black text-teal-400 w-14 text-right">
                            {result.average_iou?.toFixed(3)}
                          </span>
                        </div>

                        {/* Threshold legend */}
                        <div className="mt-3 flex flex-wrap gap-4 text-xs text-slate-600">
                          <span><span className="text-green-400">■</span> ≥0.65 Strong agreement</span>
                          <span><span className="text-amber-400">■</span> 0.40–0.65 Moderate</span>
                          <span><span className="text-red-400">■</span> &lt;0.40 Low — escalate</span>
                        </div>
                      </div>
                    ) : (
                      <div className="text-center py-8">
                        <p className="text-slate-500 text-sm">IoU data unavailable</p>
                        <p className="text-slate-600 text-xs mt-1">Requires ≥2 GRAD-CAM heatmaps (≥2 deep models loaded)</p>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* ── Clinical information ── */}
              {info && (
                <div className="card-bordered">
                  <div className="flex items-center gap-2 mb-3">
                    <div className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: classColor }} />
                    <p className="text-xs text-slate-500 uppercase tracking-wider font-semibold">
                      Clinical Information — {result.final_class}
                    </p>
                  </div>
                  <p className="text-slate-300 text-sm leading-relaxed">{info.description}</p>

                  {/* Quick facts grid */}
                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 mt-4">
                    <div className="bg-slate-700/30 rounded-lg p-3">
                      <p className="text-xs text-slate-500 mb-1">ICD-10 Code</p>
                      <p className="text-white font-mono font-bold text-sm">{info.icdCode}</p>
                    </div>
                    <div className="bg-slate-700/30 rounded-lg p-3">
                      <p className="text-xs text-slate-500 mb-1">Priority Level</p>
                      <p className={`font-bold text-sm capitalize ${urgStyle?.text}`}>{info.urgency}</p>
                    </div>
                    <div className="bg-slate-700/30 rounded-lg p-3 col-span-2 sm:col-span-1">
                      <p className="text-xs text-slate-500 mb-1">Ensemble Accuracy</p>
                      <p className="text-teal-400 font-bold text-sm">98.20% (6-model)</p>
                    </div>
                  </div>
                </div>
              )}

              {/* ── Research methodology note ── */}
              <div className="card-bordered">
                <p className="text-xs text-slate-500 uppercase tracking-wider font-semibold mb-3">
                  About This Analysis
                </p>
                <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-3 text-xs">
                  {[
                    { label: 'Ensemble Architecture', value: '6-model majority vote + confidence weighting' },
                    { label: 'Deep Models', value: 'CNN · InceptionV3 · Xception' },
                    { label: 'Classical Models', value: 'SVM · Random Forest · Decision Tree' },
                    { label: 'Training Dataset', value: '9,047 MRI scans · 4 classes' },
                    { label: 'Explainability', value: 'GRAD-CAM + EAA-IoU (novel)' },
                    { label: 'Validated Accuracy', value: '98.20% on 2,063 test images' },
                  ].map(item => (
                    <div key={item.label} className="bg-slate-700/30 rounded-lg p-3">
                      <p className="text-slate-500 mb-1">{item.label}</p>
                      <p className="text-slate-300 font-medium">{item.value}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* ── Disclaimer ── */}
              <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 p-4">
                <div className="flex items-start gap-3">
                  <svg className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24"
                    stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round"
                      d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
                  </svg>
                  <div>
                    <p className="text-amber-400 font-bold text-sm">Research Demonstration Only — Not for Clinical Use</p>
                    <p className="text-slate-400 text-xs mt-1.5 leading-relaxed">
                      This system is a proof-of-concept developed for academic research (HND Data Analytics, ESOFT Metro Campus, 2024).
                      It has <strong className="text-slate-300">not been validated for clinical diagnosis</strong> and must not substitute
                      professional medical judgement. AI-generated predictions may contain errors.
                      Always consult a qualified radiologist or neurologist before making any clinical decisions.
                      Image analysis is performed in-browser via a research API — no patient data is stored.
                    </p>
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  )
}
