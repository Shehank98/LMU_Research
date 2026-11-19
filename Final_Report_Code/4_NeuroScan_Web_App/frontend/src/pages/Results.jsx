import { useEffect, useMemo, useState } from 'react'
import { getResults } from '../api'
import { Badge, Bar, Card, CLASS_COLORS, PageHeader, Spinner, Stat, pct } from '../components/ui'

const GROUP_COLORS = {
  'Final ensemble': '#0f766e',
  'Deep model': '#334155',
  'Classical ML': '#94a3b8',
  Reference: '#334155',
  Other: '#94a3b8',
}

// Headline view: deep models, ensembles and the final vote. Every single classifier is in the table below.
const isHeadline = (m) => m.group !== 'Classical ML' || /ensemble/i.test(m.name)

function AccuracyChart({ models }) {
  const headline = models.filter(isHeadline)
  const sorted = [...(headline.length ? headline : models)].sort((a, b) => b.accuracy - a.accuracy)
  return (
    <div className="space-y-2.5">
      {sorted.map((m) => (
        <div key={m.id} className="grid grid-cols-[minmax(0,16rem)_1fr_4rem] items-center gap-3 text-sm">
          <span className="truncate text-slate-700" title={m.name}>{m.name}</span>
          <Bar value={m.accuracy} color={GROUP_COLORS[m.group] || '#334155'} height="h-2.5" />
          <span className="text-right tabular-nums font-medium text-slate-900">{pct(m.accuracy, 2)}</span>
        </div>
      ))}
    </div>
  )
}

function ResultsTable({ models }) {
  const sorted = [...models].sort((a, b) => b.accuracy - a.accuracy)
  return (
    <div className="-mx-5 -my-5 overflow-x-auto sm:-mx-6 sm:-my-6">
      <table className="min-w-full divide-y divide-slate-100">
        <thead className="bg-slate-50">
          <tr>
            <th className="th">Model</th><th className="th">Group</th>
            <th className="th text-right">Accuracy</th><th className="th text-right">Precision</th>
            <th className="th text-right">Recall</th><th className="th text-right">F1</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-100">
          {sorted.map((m) => (
            <tr key={m.id}>
              <td className="td font-medium text-slate-900">{m.name}</td>
              <td className="td text-slate-500">{m.group}</td>
              <td className="td text-right tabular-nums">{pct(m.accuracy, 2)}</td>
              <td className="td text-right tabular-nums">{pct(m.precision, 2)}</td>
              <td className="td text-right tabular-nums">{pct(m.recall, 2)}</td>
              <td className="td text-right tabular-nums">{pct(m.f1, 2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function EaaCard({ eaa }) {
  const byClass = Object.entries(eaa.mean_eaa_by_true_class || {})
  return (
    <Card title="Explainability: EAA-IoU on the test set"
      subtitle="Does low agreement between the models' Grad-CAM maps signal a wrong prediction?">
      <div className="grid gap-4 sm:grid-cols-4">
        <Stat label="Mean EAA, correct" value={eaa.mean_eaa_correct?.toFixed(3) ?? '–'} />
        <Stat label="Mean EAA, wrong" value={eaa.mean_eaa_misclassified?.toFixed(3) ?? '–'} />
        <Stat label="AUROC" value={eaa.auroc_low_eaa_predicts_error?.toFixed(3) ?? '–'} hint="low EAA as an error detector" />
        <Stat label="Mann-Whitney p" value={typeof eaa.mann_whitney_p_value === 'number' ? eaa.mann_whitney_p_value.toExponential(2) : '–'} />
      </div>
      <div className="mt-6 grid gap-8 lg:grid-cols-2">
        <div>
          <p className="mb-3 text-xs font-medium uppercase tracking-wide text-slate-500">Mean EAA-IoU by true class</p>
          <div className="space-y-2.5">
            {byClass.map(([name, v]) => {
              const label = name === 'notumor' ? 'No Tumor' : name.charAt(0).toUpperCase() + name.slice(1)
              return (
                <div key={name} className="grid grid-cols-[7rem_1fr_3rem] items-center gap-3 text-sm">
                  <span className="text-slate-700">{label}</span>
                  <Bar value={v} color={CLASS_COLORS[label] || '#334155'} />
                  <span className="text-right tabular-nums text-slate-600">{v.toFixed(2)}</span>
                </div>
              )
            })}
          </div>
        </div>
        <div>
          <p className="mb-3 text-xs font-medium uppercase tracking-wide text-slate-500">Escalation threshold</p>
          <table className="min-w-full text-sm">
            <thead><tr className="text-left text-xs text-slate-500">
              <th className="py-1.5 font-medium">Flag if EAA below</th><th className="py-1.5 text-right font-medium">Scans flagged</th>
              <th className="py-1.5 text-right font-medium">Errors caught</th></tr></thead>
            <tbody className="divide-y divide-slate-100">
              {(eaa.threshold_table || []).map((r) => (
                <tr key={r.threshold}>
                  <td className="py-1.5 tabular-nums">{r.threshold}</td>
                  <td className="py-1.5 text-right tabular-nums">{r.flagged_pct}%</td>
                  <td className="py-1.5 text-right tabular-nums">{r.errors_caught_pct}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </Card>
  )
}

function DatasetCard({ dataset }) {
  const classes = Object.keys(dataset.Training)
  const total = (split) => Object.values(dataset[split]).reduce((a, b) => a + b, 0)
  return (
    <Card title="Dataset" subtitle="Kaggle Brain Tumor MRI dataset. The official test split is never used for training.">
      <div className="-mx-5 -my-5 overflow-x-auto sm:-mx-6 sm:-my-6">
        <table className="min-w-full divide-y divide-slate-100">
          <thead className="bg-slate-50"><tr><th className="th">Class</th><th className="th text-right">Training</th><th className="th text-right">Testing</th></tr></thead>
          <tbody className="divide-y divide-slate-100">
            {classes.map((c) => (
              <tr key={c}>
                <td className="td"><span className="inline-flex items-center gap-2">
                  <span className="h-2 w-2 rounded-full" style={{ backgroundColor: CLASS_COLORS[c] }} />{c}</span></td>
                <td className="td text-right tabular-nums">{dataset.Training[c].toLocaleString()}</td>
                <td className="td text-right tabular-nums">{dataset.Testing[c].toLocaleString()}</td>
              </tr>
            ))}
            <tr className="bg-slate-50 font-semibold">
              <td className="td text-slate-900">Total</td>
              <td className="td text-right tabular-nums text-slate-900">{total('Training').toLocaleString()}</td>
              <td className="td text-right tabular-nums text-slate-900">{total('Testing').toLocaleString()}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </Card>
  )
}

export default function Results() {
  const [data, setData] = useState(null)
  const [error, setError] = useState('')
  useEffect(() => { getResults().then(setData).catch((e) => setError(e.message)) }, [])

  const summary = useMemo(() => {
    if (!data) return null
    const models = data.models
    const best = [...models].sort((a, b) => b.accuracy - a.accuracy)[0]
    const finals = models.filter((m) => m.group === 'Final ensemble')
    const final = finals.sort((a, b) => b.accuracy - a.accuracy)[0]
    const testTotal = Object.values(data.dataset.Testing).reduce((a, b) => a + b, 0)
    return { best, final, testTotal }
  }, [data])

  if (error) return <div className="card card-pad text-sm text-rose-700">{error}</div>
  if (!data) return <div className="flex items-center gap-2 text-sm text-slate-500"><Spinner /> Loading results</div>

  const live = data.source !== 'reference'
  return (
    <>
      <PageHeader title="Results"
        description="Test-set performance of every model, computed in the Colab training run and published to Hugging Face.">
        {live ? <Badge tone="green">Live results · {data.run}</Badge>
              : <Badge tone="amber">Reference values · new run not uploaded yet</Badge>}
      </PageHeader>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="Best model" value={pct(summary.best?.accuracy, 2)} hint={summary.best?.name} />
        <Stat label="Final ensemble" value={summary.final ? pct(summary.final.accuracy, 2) : '–'} hint={summary.final?.name || 'not in this run yet'} />
        <Stat label="Models evaluated" value={data.models.length} />
        <Stat label="Test images" value={summary.testTotal.toLocaleString()} />
      </div>

      <div className="mt-6 grid gap-6">
        <Card title="Test accuracy by model" subtitle={live ? "Deep models, ensembles and the final vote. Individual classifiers are listed in the table below." : undefined}
          action={<div className="hidden gap-3 text-xs text-slate-500 sm:flex">
            {['Final ensemble', 'Deep model', 'Classical ML'].map((g) => (
              <span key={g} className="inline-flex items-center gap-1.5">
                <span className="h-2 w-2 rounded-full" style={{ backgroundColor: GROUP_COLORS[g] }} />{g}</span>))}
          </div>}>
          <AccuracyChart models={data.models} />
        </Card>
        {live && <Card title="All metrics" subtitle="Macro-averaged over the four classes."><ResultsTable models={data.models} /></Card>}
        {data.eaa && <EaaCard eaa={data.eaa} />}
        <DatasetCard dataset={data.dataset} />
      </div>
    </>
  )
}
