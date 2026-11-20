import { Badge, Bar, Card, CLASS_COLORS, Spinner, levelTone, pct } from './ui'

export function PredictionPanel({ result, busy, error }) {
  if (error) {
    return (
      <Card title="Prediction">
        <div className="rounded-lg border border-rose-200 bg-rose-50 p-4 text-sm text-rose-700">{error}</div>
      </Card>
    )
  }
  if (busy) {
    return (
      <Card title="Prediction">
        <div className="flex flex-col items-center justify-center py-16 text-center">
          <Spinner className="h-6 w-6 text-brand-700" />
          <p className="mt-3 text-sm font-medium text-slate-700">Running six models and Grad-CAM</p>
          <p className="mt-1 text-xs text-slate-500">The first analysis after start-up can take up to a minute.</p>
        </div>
      </Card>
    )
  }
  if (!result) {
    return (
      <Card title="Prediction">
        <div className="py-16 text-center">
          <p className="text-sm font-medium text-slate-700">No scan analysed yet</p>
          <p className="mt-1 text-sm text-slate-500">Upload an MRI image and it is analysed automatically.</p>
        </div>
      </Card>
    )
  }
  const { xai } = result
  return (
    <Card title="Prediction" subtitle={`Confidence-weighted vote of ${result.models_used} predictors`}>
      <div className="flex items-end justify-between gap-4">
        <div>
          <p className="text-xs font-medium uppercase tracking-wide text-slate-500">Predicted class</p>
          <p className="mt-1 flex items-center gap-2.5 text-3xl font-semibold text-slate-900">
            <span className="h-3 w-3 rounded-full" style={{ backgroundColor: CLASS_COLORS[result.final_class] }} />
            {result.final_class}
          </p>
        </div>
        <div className="text-right">
          <p className="text-xs font-medium uppercase tracking-wide text-slate-500">Confidence</p>
          <p className="mt-1 text-3xl font-semibold text-slate-900">{pct(result.confidence)}</p>
        </div>
      </div>

      <div className="mt-5 grid grid-cols-2 gap-3">
        <div className="rounded-lg border border-slate-200 px-4 py-3">
          <p className="text-xs text-slate-500">Ensemble confidence</p>
          <div className="mt-1"><Badge tone={levelTone(result.confidence_level)}>{result.confidence_level}</Badge></div>
        </div>
        <div className="rounded-lg border border-slate-200 px-4 py-3">
          <p className="text-xs text-slate-500">Model agreement (EAA-IoU)</p>
          <div className="mt-1 flex items-center gap-2">
            <span className="text-sm font-semibold text-slate-900">
              {xai.eaa_iou === null ? '–' : xai.eaa_iou.toFixed(2)}
            </span>
            {xai.agreement_level && <Badge tone={levelTone(xai.agreement_level)}>{xai.agreement_level}</Badge>}
          </div>
        </div>
      </div>

      {xai.alert && (
        <div className="mt-4 rounded-lg border border-rose-200 bg-rose-50 p-4 text-sm text-rose-700">
          <p className="font-semibold">Radiologist review recommended</p>
          <p className="mt-0.5">The three deep models focus on different regions of this scan (EAA-IoU below {xai.thresholds.low}). Treat this prediction with caution.</p>
        </div>
      )}

      <div className="mt-6 space-y-3">
        <p className="text-xs font-medium uppercase tracking-wide text-slate-500">Class probabilities</p>
        {result.class_names.map((name, i) => (
          <div key={name} className="grid grid-cols-[7rem_1fr_3.5rem] items-center gap-3 text-sm">
            <span className="text-slate-700">{name}</span>
            <Bar value={result.class_probabilities[i]} color={CLASS_COLORS[name]} />
            <span className="text-right tabular-nums text-slate-600">{pct(result.class_probabilities[i])}</span>
          </div>
        ))}
      </div>
    </Card>
  )
}

export function ExplanationPanel({ xai }) {
  const tiles = [
    { name: 'Original', image: xai.original },
    ...xai.heatmaps,
    ...(xai.consensus ? [{ name: 'Consensus', image: xai.consensus }] : []),
  ]
  return (
    <Card title="Where the models looked"
      subtitle="Grad-CAM attention for each deep model. Warm colours show the regions that drove the prediction.">
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">
        {tiles.map((t) => (
          <figure key={t.name}>
            <img src={t.image} alt={`${t.name} attention map`} className="aspect-square w-full rounded-lg border border-slate-200 object-cover" />
            <figcaption className="mt-2 text-center text-xs font-medium text-slate-600">{t.name}</figcaption>
          </figure>
        ))}
      </div>
      {xai.pairwise_iou.length > 0 && (
        <div className="mt-6 flex flex-col gap-4 border-t border-slate-100 pt-5 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex flex-wrap gap-2">
            {xai.pairwise_iou.map((p) => (
              <span key={p.pair} className="rounded-md bg-slate-100 px-2.5 py-1 text-xs text-slate-600">
                {p.pair}: <span className="font-semibold text-slate-900">{p.iou.toFixed(2)}</span>
              </span>
            ))}
          </div>
          <p className="max-w-md text-xs text-slate-500">
            EAA-IoU is the mean overlap between the three maps. At or above {xai.thresholds.high} the models agree;
            below {xai.thresholds.low} the case is flagged for review.
          </p>
        </div>
      )}
    </Card>
  )
}

export function VotesPanel({ perModel }) {
  return (
    <Card title="Individual model votes" subtitle="Each predictor's own prediction before fusion.">
      <div className="-mx-5 -my-5 overflow-x-auto sm:-mx-6 sm:-my-6">
        <table className="min-w-full divide-y divide-slate-100">
          <thead className="bg-slate-50">
            <tr><th className="th">Model</th><th className="th">Type</th><th className="th">Prediction</th><th className="th w-1/3">Confidence</th></tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {perModel.map((m) => (
              <tr key={m.name}>
                <td className="td font-medium text-slate-900">{m.name}</td>
                <td className="td text-slate-500">{m.family}</td>
                <td className="td">
                  <span className="inline-flex items-center gap-2">
                    <span className="h-2 w-2 rounded-full" style={{ backgroundColor: CLASS_COLORS[m.predicted_class] }} />
                    {m.predicted_class}
                  </span>
                </td>
                <td className="td">
                  <div className="flex items-center gap-3">
                    <Bar value={m.confidence} color="#334155" />
                    <span className="w-14 text-right tabular-nums text-slate-600">{pct(m.confidence)}</span>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

