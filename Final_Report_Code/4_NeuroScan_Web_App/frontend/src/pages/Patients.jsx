import { useEffect, useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { createPatient, getPatients } from '../api'
import { Badge, Card, CLASS_COLORS, PageHeader, Spinner, pct } from '../components/ui'

const fmtDate = (d) => (d ? new Date(d).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' }) : '–')

export function Initials({ name, size = 'h-9 w-9 text-sm' }) {
  const text = name.split(' ').map((w) => w[0]).slice(0, 2).join('')
  return <span className={`grid shrink-0 place-items-center rounded-full bg-brand-50 font-semibold text-brand-700 ${size}`}>{text}</span>
}

export function StatusBadge({ alert, pending }) {
  if (alert && pending) return <Badge tone="red">Alert · review pending</Badge>
  if (pending) return <Badge tone="amber">Review pending</Badge>
  return <Badge tone="green">Reviewed</Badge>
}

function NewPatientForm({ onCreated, onCancel }) {
  const [form, setForm] = useState({ name: '', name_si: '', sex: 'Female', dob: '', phone: '', city: '', referring_clinician: '', history: '' })
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const set = (k) => (e) => setForm({ ...form, [k]: e.target.value })

  const submit = async (e) => {
    e.preventDefault()
    setBusy(true); setError('')
    try { onCreated(await createPatient(form)) }
    catch (err) { setError(err.message) }
    finally { setBusy(false) }
  }

  return (
    <Card title="New patient" subtitle="Create a record, then add MRI scans from the patient's page." className="mb-6">
      <form onSubmit={submit} className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <div className="sm:col-span-2"><label className="label">Full name</label><input className="input" required value={form.name} onChange={set('name')} /></div>
        <div className="sm:col-span-2"><label className="label">Name in Sinhala (optional)</label><input className="input" value={form.name_si} onChange={set('name_si')} /></div>
        <div><label className="label">Sex</label>
          <select className="input" value={form.sex} onChange={set('sex')}><option>Female</option><option>Male</option></select></div>
        <div><label className="label">Date of birth</label><input className="input" type="date" required value={form.dob} onChange={set('dob')} /></div>
        <div><label className="label">Phone</label><input className="input" value={form.phone} onChange={set('phone')} /></div>
        <div><label className="label">City</label><input className="input" value={form.city} onChange={set('city')} /></div>
        <div className="sm:col-span-2"><label className="label">Referring clinician</label><input className="input" value={form.referring_clinician} onChange={set('referring_clinician')} /></div>
        <div className="sm:col-span-2"><label className="label">Clinical history</label><input className="input" value={form.history} onChange={set('history')} /></div>
        {error && <p className="text-sm text-rose-700 sm:col-span-2 lg:col-span-4">{error}</p>}
        <div className="flex gap-3 sm:col-span-2 lg:col-span-4">
          <button className="btn-primary" disabled={busy}>{busy ? <><Spinner /> Saving</> : 'Save patient'}</button>
          <button type="button" className="btn-secondary" onClick={onCancel}>Cancel</button>
        </div>
      </form>
    </Card>
  )
}

export default function Patients() {
  const [data, setData] = useState(null)
  const [error, setError] = useState('')
  const [query, setQuery] = useState('')
  const [adding, setAdding] = useState(false)
  const navigate = useNavigate()

  useEffect(() => { getPatients().then(setData).catch((e) => setError(e.message)) }, [])

  const rows = useMemo(() => {
    const q = query.trim().toLowerCase()
    return (data?.patients || []).filter((p) =>
      !q || [p.name, p.name_si, p.id, p.city].some((v) => v && v.toLowerCase().includes(q)))
  }, [data, query])

  return (
    <>
      <PageHeader title="Patients" description="Patient records with every previous MRI analysis and the radiologist's review.">
        <button className="btn-primary" onClick={() => setAdding(true)}>Add patient</button>
      </PageHeader>
      {data?.demo && (
        <div className="mb-6 rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800">
          <span className="font-semibold">Demonstration data.</span> These patients are fictitious. Their MRI slices come from the public
          test set and were analysed by NeuroScan's trained models.
        </div>
      )}
      {adding && <NewPatientForm onCancel={() => setAdding(false)} onCreated={(p) => navigate(`/patients/${p.id}`)} />}
      <Card title={`${rows.length} patient${rows.length === 1 ? '' : 's'}`}
        action={<input className="input w-56" placeholder="Search name, ID or city" value={query} onChange={(e) => setQuery(e.target.value)} />}>
        {error && <p className="text-sm text-rose-700">{error}</p>}
        {!data && !error && <div className="flex justify-center py-10"><Spinner className="h-6 w-6 text-brand-700" /></div>}
        {data && (
          <div className="-mx-5 -my-5 overflow-x-auto sm:-mx-6 sm:-my-6">
            <table className="min-w-full divide-y divide-slate-100">
              <thead className="bg-slate-50">
                <tr><th className="th">Patient</th><th className="th">ID</th><th className="th">Age / sex</th><th className="th">City</th>
                  <th className="th">Scans</th><th className="th">Last scan</th><th className="th">Last AI result</th><th className="th">Status</th></tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {rows.map((p) => (
                  <tr key={p.id} className="cursor-pointer hover:bg-slate-50" onClick={() => navigate(`/patients/${p.id}`)}>
                    <td className="td">
                      <div className="flex items-center gap-3">
                        <Initials name={p.name} />
                        <div>
                          <Link to={`/patients/${p.id}`} className="font-medium text-slate-900 hover:text-brand-700">{p.name}</Link>
                          {p.name_si && <p className="text-xs text-slate-500">{p.name_si}</p>}
                        </div>
                      </div>
                    </td>
                    <td className="td tabular-nums text-slate-500">{p.id}</td>
                    <td className="td">{p.age} · {p.sex}</td>
                    <td className="td">{p.city || '–'}</td>
                    <td className="td tabular-nums">{p.n_scans}</td>
                    <td className="td">{fmtDate(p.last_scan_date)}</td>
                    <td className="td">
                      {p.last_result ? (
                        <span className="inline-flex items-center gap-2">
                          <span className="h-2 w-2 rounded-full" style={{ backgroundColor: CLASS_COLORS[p.last_result] }} />
                          {p.last_result} <span className="text-slate-400">{pct(p.last_confidence, 0)}</span>
                        </span>
                      ) : '–'}
                    </td>
                    <td className="td">{p.n_scans ? <StatusBadge alert={p.last_alert} pending={p.pending_reviews > 0} /> : <Badge>No scans</Badge>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </>
  )
}

export { fmtDate }
