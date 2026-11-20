import { useRef, useState } from 'react'
import { analyseImage } from '../api'
import { Card, PageHeader, Spinner } from '../components/ui'
import { ExplanationPanel, PredictionPanel, VotesPanel } from '../components/Analysis'

function UploadPanel({ file, preview, busy, onFile, onClear }) {
  const input = useRef(null)
  const [drag, setDrag] = useState(false)

  const accept = (f) => f && ['image/jpeg', 'image/png', 'image/jpg'].includes(f.type) && onFile(f)

  return (
    <Card title="MRI scan" subtitle="Upload one axial brain MRI slice (JPEG or PNG).">
      {!preview ? (
        <button type="button" onClick={() => input.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => { e.preventDefault(); setDrag(false); accept(e.dataTransfer.files[0]) }}
          className={`flex w-full flex-col items-center justify-center rounded-lg border-2 border-dashed px-6 py-14 text-center transition-colors ${
            drag ? 'border-brand-600 bg-brand-50' : 'border-slate-300 hover:border-slate-400 hover:bg-slate-50'}`}>
          <svg className="h-10 w-10 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
            <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5m-13.5-9L12 3m0 0 4.5 4.5M12 3v13.5" />
          </svg>
          <span className="mt-3 text-sm font-medium text-slate-700">Drop an image here, or click to choose</span>
          <span className="mt-1 text-xs text-slate-500">JPEG or PNG</span>
        </button>
      ) : (
        <div>
          <div className="overflow-hidden rounded-lg border border-slate-200 bg-slate-900">
            <img src={preview} alt="Uploaded MRI scan" className="mx-auto max-h-80 w-auto object-contain" />
          </div>
          <p className="mt-3 truncate text-sm text-slate-500" title={file?.name}>{file?.name}</p>
          <div className="mt-4 flex gap-3">
            <button className="btn-primary flex-1" onClick={() => input.current?.click()} disabled={busy}>
              {busy ? <><Spinner /> Analysing</> : 'Upload another scan'}
            </button>
            <button className="btn-secondary" onClick={onClear} disabled={busy}>Clear</button>
          </div>
        </div>
      )}
      <input ref={input} type="file" accept="image/jpeg,image/png" className="hidden"
        onChange={(e) => { accept(e.target.files[0]); e.target.value = '' }} />
    </Card>
  )
}

export default function Diagnose() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const analyse = async (f) => {
    setBusy(true); setError('')
    try { setResult(await analyseImage(f)) }
    catch (e) { setError(e.message) }
    finally { setBusy(false) }
  }
  const choose = (f) => {                 // a new upload starts the analysis straight away
    setFile(f); setResult(null); setError('')
    setPreview(URL.createObjectURL(f))
    analyse(f)
  }
  const clear = () => { setFile(null); setPreview(null); setResult(null); setError('') }

  return (
    <>
      <PageHeader title="Diagnose an MRI scan" />
      <div className="grid gap-6 lg:grid-cols-5">
        <div className="lg:col-span-2">
          <UploadPanel file={file} preview={preview} busy={busy} onFile={choose} onClear={clear} />
        </div>
        <div className="lg:col-span-3">
          <PredictionPanel result={result} busy={busy} error={error} />
        </div>
      </div>
      {result && !busy && (
        <div className="mt-6 space-y-6">
          <ExplanationPanel xai={result.xai} />
          <VotesPanel perModel={result.per_model} />
        </div>
      )}
    </>
  )
}
