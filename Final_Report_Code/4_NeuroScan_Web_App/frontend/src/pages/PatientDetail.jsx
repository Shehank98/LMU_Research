import { useEffect, useRef, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { getPatient, reviewScan, uploadPatientScan } from '../api'
import { Badge, Card, CLASS_COLORS, PageHeader, Spinner, levelTone, pct } from '../components/ui'
import { ExplanationPanel, PredictionPanel, VotesPanel } from '../components/Analysis'
import { Initials, StatusBadge, fmtDate } from './Patients'

const CLASSES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

function Field({ label, children }) {
  return (
    <div>
      <p className="text-xs text-slate-500">{label}</p>
      <p className="mt-0.5 text-sm font-medium text-slate-800">{children || '–'}</p>
    </div>
  )
}

function Profile({ p }) {
  const pending = p.scans.filter((s) => s.review.status === 'pending').length
  return (
    <Card>
      <div className="flex flex-col gap-6 lg:flex-row">
        <div className="flex items-center gap-4 lg:w-72">
          <Initials name={p.name} size="h-14 w-14 text-lg" />
          <div>
            <p className="text-lg font-semibold text-slate-900">{p.name}</p>
            {p.name_si && <p className="text-sm text-slate-600">{p.name_si}</p>}
            <p className="mt-1 text-xs tabular-nums text-slate-500">{p.id}</p>
          </div>
        </div>
        <div className="grid flex-1 grid-cols-2 gap-4 sm:grid-cols-4">
          <Field label="Age">{p.age} years ({fmtDate(p.dob)})</Field>
          <Field label="Sex">{p.sex}</Field>
          <Field label="Blood group">{p.blood_group}</Field>
          <Field label="Allergies">{p.allergies}</Field>
          <Field label="Phone">{p.phone}</Field>
          <Field label="City">{p.city}</Field>
          <Field label="Referring clinician">{p.referring_clinician}</Field>
          <Field label="Scans / pending reviews">{p.scans.length} / {pending}</Field>
        </div>
      </div>
      {p.history && (
        <div className="mt-5 border-t border-slate-100 pt-4">
          <p className="text-xs text-slate-500">Clinical history</p>
          <p className="mt-1 text-sm text-slate-700">{p.history}</p>
        </div>
      )}
    </Card>
  )
}

function NewScan({ patientId, onDone }) {
  const input = useRef(null)
  const [file, setFile] = useState(null)
  const [study, setStudy] = useState('')
  const [notes, setNotes] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const submit = async () => {
    setBusy(true); setError('')
    try { onDone(await uploadPatientScan(patientId, file, study, notes)) }
    catch (e) { setError(e.message) }
    finally { setBusy(false) }
  }

  return (
    <div className="mb-5 grid gap-3 rounded-lg border border-slate-200 bg-slate-50 p-4 sm:grid-cols-4">
      <div><label className="label">MRI image</label>
        <button type="button" className="btn-secondary w-full truncate" onClick={() => input.current?.click()}>
          {file ? file.name : 'Choose JPEG or PNG'}</button>
        <input ref={input} type="file" accept="image/jpeg,image/png" className="hidden" onChange={(e) => setFile(e.target.files[0])} /></div>
      <div><label className="label">Study</label><input className="input" placeholder="MRI brain, contrast-enhanced" value={study} onChange={(e) => setStudy(e.target.value)} /></div>
      <div><label className="label">Reason / notes</label><input className="input" value={notes} onChange={(e) => setNotes(e.target.value)} /></div>
      <div className="flex items-end">
        <button className="btn-primary w-full" disabled={!file || busy} onClick={submit}>{busy ? <><Spinner /> Analysing</> : 'Analyse and save'}</button>
      </div>
      {error && <p className="text-sm text-rose-700 sm:col-span-4">{error}</p>}
    </div>
  )
}

function History({ scans, selected, onSelect }) {
  return (
    <div className="-mx-5 -mb-5 overflow-x-auto sm:-mx-6 sm:-mb-6">
      <table className="min-w-full divide-y divide-slate-100">
        <thead className="bg-slate-50">
          <tr><th className="th">Date</th><th className="th">Study</th><th className="th">AI result</th><th className="th">Agreement</th>
            <th className="th">Radiologist review</th></tr>
        </thead>
        <tbody className="divide-y divide-slate-100">
          {scans.map((s) => {
            const r = s.result
            return (
              <tr key={s.id} onClick={() => onSelect(s.id)}
                className={`cursor-pointer ${selected === s.id ? 'bg-brand-50' : 'hover:bg-slate-50'}`}>
                <td className="td whitespace-nowrap font-medium text-slate-900">{fmtDate(s.date)}</td>
                <td className="td">{s.study}</td>
                <td className="td whitespace-nowrap">
                  <span className="inline-flex items-center gap-2">
                    <span className="h-2 w-2 rounded-full" style={{ backgroundColor: CLASS_COLORS[r.final_class] }} />
                    {r.final_class} <span className="text-slate-400">{pct(r.confidence, 0)}</span>
                  </span>
                </td>
                <td className="td whitespace-nowrap">
                  <span className="mr-2 tabular-nums">{r.xai.eaa_iou?.toFixed(2)}</span>
                  <Badge tone={levelTone(r.xai.agreement_level)}>{r.xai.alert ? 'alert' : r.xai.agreement_level}</Badge>
                </td>
                <td className="td">
                  {s.review.status === 'pending'
                    ? <StatusBadge alert={r.xai.alert} pending />
                    : <span className="inline-flex items-center gap-2">
                        <Badge tone={s.review.status === 'corrected' ? 'amber' : 'green'}>{s.review.status}</Badge>
                        {s.review.final_diagnosis}
                      </span>}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function ReviewPanel({ patientId, scan, onSaved }) {
  const [radiologist, setRadiologist] = useState('')
  const [final, setFinal] = useState(scan.result.final_class)
  const [note, setNote] = useState('')
  const [busy, setBusy] = useState(false)
  const r = scan.review

  if (r.status !== 'pending') {
    return (
      <Card title="Radiologist review" subtitle={`${fmtDate(r.date)} · ${r.radiologist}`}>
        <div className="flex items-center gap-2">
          <Badge tone={r.status === 'corrected' ? 'amber' : 'green'}>{r.status}</Badge>
          <span className="text-sm font-semibold text-slate-900">{r.final_diagnosis}</span>
        </div>
        {r.note && <p className="mt-3 text-sm text-slate-600">{r.note}</p>}
        {r.status === 'corrected' && (
          <p className="mt-3 text-xs text-slate-500">NeuroScan predicted {scan.result.final_class}; the radiologist's diagnosis is recorded as final.</p>
        )}
      </Card>
    )
  }

  const save = async () => {
    setBusy(true)
    try {
      onSaved(await reviewScan(patientId, scan.id, {
        status: final === scan.result.final_class ? 'confirmed' : 'corrected',
        final_diagnosis: final, radiologist, note,
      }))
    } finally { setBusy(false) }
  }

  return (
    <Card title="Radiologist review" subtitle="Awaiting review. The AI result is not a diagnosis until a radiologist confirms it.">
      <div className="space-y-3">
        <div><label className="label">Radiologist</label><input className="input" value={radiologist} onChange={(e) => setRadiologist(e.target.value)} /></div>
        <div><label className="label">Final diagnosis</label>
          <select className="input" value={final} onChange={(e) => setFinal(e.target.value)}>{CLASSES.map((c) => <option key={c}>{c}</option>)}</select></div>
        <div><label className="label">Note</label><textarea className="input" rows="3" value={note} onChange={(e) => setNote(e.target.value)} /></div>
        <button className="btn-primary w-full" disabled={!radiologist || busy} onClick={save}>
          {busy ? <><Spinner /> Saving</> : final === scan.result.final_class ? 'Confirm AI result' : 'Save corrected diagnosis'}
        </button>
      </div>
    </Card>
  )
}

export default function PatientDetail() {
  const { id } = useParams()
  const [patient, setPatient] = useState(null)
  const [selected, setSelected] = useState(null)
  const [adding, setAdding] = useState(false)
  const [error, setError] = useState('')

  const load = (pick) => getPatient(id).then((p) => {
    setPatient(p)
    setSelected(pick || p.scans[0]?.id || null)
  }).catch((e) => setError(e.message))
  useEffect(() => { load() }, [id])

  if (error) return <p className="text-sm text-rose-700">{error}</p>
  if (!patient) return <div className="flex justify-center py-20"><Spinner className="h-6 w-6 text-brand-700" /></div>
  const scan = patient.scans.find((s) => s.id === selected)

  return (
    <>
      <Link to="/patients" className="text-sm text-slate-500 hover:text-slate-900">← All patients</Link>
      <PageHeader title="Patient record" />
      <div className="space-y-6">
        <Profile p={patient} />
        <Card title="Scan history" subtitle="Every NeuroScan analysis for this patient, newest first. Select a scan to see its details."
          action={<button className="btn-primary" onClick={() => setAdding(!adding)}>{adding ? 'Close' : 'Add new scan'}</button>}>
          {adding && <NewScan patientId={patient.id} onDone={(s) => { setAdding(false); load(s.id) }} />}
          {patient.scans.length ? <History scans={patient.scans} selected={selected} onSelect={setSelected} />
            : <p className="text-sm text-slate-500">No scans yet. Use "Add new scan" to analyse the first MRI.</p>}
        </Card>
        {scan && (
          <>
            <div>
              <h2 className="text-lg font-semibold">Scan of {fmtDate(scan.date)}</h2>
              <p className="text-sm text-slate-500">{scan.study}{scan.notes ? ` · ${scan.notes}` : ''}</p>
            </div>
            <div className="grid gap-6 lg:grid-cols-5">
              <div className="space-y-6 lg:col-span-2">
                <Card title="MRI scan">
                  <div className="overflow-hidden rounded-lg border border-slate-200 bg-slate-900">
                    <img src={scan.result.xai.original} alt="MRI scan" className="mx-auto max-h-72 w-auto object-contain" />
                  </div>
                </Card>
                <ReviewPanel key={scan.id} patientId={patient.id} scan={scan} onSaved={(s) => load(s.id)} />
              </div>
              <div className="lg:col-span-3"><PredictionPanel result={scan.result} /></div>
            </div>
            <ExplanationPanel xai={scan.result.xai} />
            <VotesPanel perModel={scan.result.per_model} />
          </>
        )}
      </div>
    </>
  )
}
