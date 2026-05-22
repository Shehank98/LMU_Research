import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, Legend,
  LineChart, Line, CartesianGrid, ScatterChart, Scatter, ZAxis,
} from 'recharts'
import SectionHeader from '../components/SectionHeader'

const LITERATURE = [
  { author: 'Talo et al.', year: 2019, method: 'ResNet34 transfer learning', accuracy: 100.0, dataset: '613 MRIs (5-fold CV)', note: 'Small dataset, 5-fold CV inflates result' },
  { author: 'Cheng et al.', year: 2015, method: 'Distance learning + offline DB', accuracy: 94.68, dataset: 'Multi-class MRI', note: '' },
  { author: 'Zhou et al.', year: 2018, method: 'DenseNet + LSTM (3D)', accuracy: 92.13, dataset: '3D MRI', note: '3D modality — different task' },
  { author: 'Abiwinanda et al.', year: 2018, method: 'Simple CNN (3 classes)', accuracy: 84.19, dataset: '3,064 T1 images', note: '3-class only' },
  { author: 'Sultan et al.', year: 2019, method: 'Deep neural network', accuracy: 96.13, dataset: 'Multi-class', note: '' },
  { author: 'Minz & Mahobiya', year: 2017, method: 'Adaboost + GLCM features', accuracy: 89.90, dataset: 'MR images', note: 'Hand-crafted features' },
  { author: 'This Study', year: 2024, method: '6-model ensemble (CNN+Inc+Xcp+3 ML)', accuracy: 98.20, dataset: '9,047 MRI (4 classes)', note: 'Novel EAA-IoU XAI layer', isOwn: true },
]

const ALL_MODELS = [
  { name: 'All Models Combined', accuracy: 98.20, color: '#f39c12' },
  { name: 'CNN', accuracy: 94.00, color: '#8e44ad' },
  { name: 'Xception', accuracy: 86.91, color: '#16a085' },
  { name: 'Inception V3', accuracy: 84.63, color: '#2980b9' },
  { name: 'Traditional ML + Ensemble Models', accuracy: 83.00, color: '#7f8c8d' },
  { name: 'CNN + Ensemble Models', accuracy: 83.71, color: '#8e44ad' },
  { name: 'Inception V3 + Ensemble Models', accuracy: 71.20, color: '#2980b9' },
  { name: 'Xception + Ensemble Models', accuracy: 70.28, color: '#16a085' },
].sort((a, b) => a.accuracy - b.accuracy)

const BUILDUP = [
  { label: 'CNN only', accuracy: 94.00 },
  { label: '+ Inception V3', accuracy: 95.40 },
  { label: '+ Xception', accuracy: 96.20 },
  { label: '+ CNN Ens.', accuracy: 96.80 },
  { label: '+ Inc Ens.', accuracy: 97.40 },
  { label: '+ Xcp Ens.', accuracy: 98.20 },
]

const FEATURE_SOURCES = [
  { src: 'CNN features',     RF: 85.45, DT: 80.99, SVM: 66.60, Ens: 83.71 },
  { src: 'InceptionV3 feat', RF: 74.64, DT: 68.05, SVM: 57.34, Ens: 71.20 },
  { src: 'Xception feat',    RF: 73.68, DT: 68.20, SVM: 56.71, Ens: 70.28 },
  { src: 'Raw pixels',       RF: 79.52, DT: 80.02, SVM: 88.26, Ens: 83.00 },
]

const COMPLEXITY = [
  { name: 'CNN', params: 1.2, accuracy: 94.00, size: 200 },
  { name: 'InceptionV3', params: 23.8, accuracy: 84.63, size: 200 },
  { name: 'Xception', params: 22.9, accuracy: 86.91, size: 200 },
  { name: 'SVM (raw)', params: 0.001, accuracy: 88.26, size: 200 },
  { name: 'Final Ens.', params: 48, accuracy: 98.20, size: 300 },
]

const CLF_COLORS = ['#2980b9', '#e74c3c', '#27ae60', '#f39c12']

export default function ModelComparison() {
  return (
    <div className="space-y-10">
      <div>
        <h2 className="text-2xl font-extrabold text-white">Model Comparison</h2>
        <p className="text-slate-400 mt-1">Literature benchmarking, ensemble build-up, and accuracy rankings</p>
      </div>

      {/* Literature table */}
      <div className="card-bordered">
        <SectionHeader title="Literature Benchmark"
          subtitle="Positioning against published brain tumor classification results" />
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-xs text-slate-400 uppercase border-b border-slate-700">
                <th className="text-left py-2 pr-3">Author</th>
                <th className="text-left py-2 pr-3">Year</th>
                <th className="text-left py-2 pr-3">Method</th>
                <th className="text-left py-2 pr-3">Dataset</th>
                <th className="text-right py-2 px-3">Accuracy</th>
              </tr>
            </thead>
            <tbody>
              {LITERATURE.sort((a, b) => b.accuracy - a.accuracy).map(row => (
                <tr key={row.author + row.year}
                  className={`border-b border-slate-700/40 ${row.isOwn ? 'bg-teal-500/5' : 'hover:bg-slate-700/20'}`}>
                  <td className={`py-2 pr-3 ${row.isOwn ? 'text-teal-300 font-bold' : 'text-slate-300'}`}>
                    {row.author}
                  </td>
                  <td className="py-2 pr-3 text-slate-400">{row.year}</td>
                  <td className="py-2 pr-3 text-slate-400 text-xs">{row.method}</td>
                  <td className="py-2 pr-3 text-slate-500 text-xs">{row.dataset}</td>
                  <td className={`py-2 px-3 text-right font-mono font-bold ${
                    row.isOwn ? 'text-teal-300' :
                    row.accuracy >= 95 ? 'text-green-400' :
                    row.accuracy >= 90 ? 'text-white' : 'text-slate-400'
                  }`}>
                    {row.accuracy.toFixed(2)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-xs text-slate-500 mt-3">
          Note: Talo et al. 100% result is on 613 images with 5-fold CV — not directly comparable to this
          study's 2,063-image held-out test set evaluation.
        </p>
      </div>

      {/* Full accuracy ranking */}
      <div className="card-bordered">
        <SectionHeader title="All Model Accuracy Rankings"
          subtitle="Every configuration evaluated on the official 2,063-image test set" />
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={ALL_MODELS} layout="vertical"
            margin={{ left: 20, right: 70, top: 5, bottom: 5 }}>
            <XAxis type="number" domain={[60, 102]} tickFormatter={v => `${v}%`}
              tick={{ fill: '#94a3b8', fontSize: 11 }} />
            <YAxis type="category" dataKey="name" width={190}
              tick={{ fill: '#94a3b8', fontSize: 11 }} />
            <Tooltip
              formatter={v => [`${Number(v).toFixed(2)}%`, 'Accuracy']}
              contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
              labelStyle={{ color: '#e2e8f0' }} />
            <Bar dataKey="accuracy" radius={[0, 4, 4, 0]}
              label={{ position: 'right', formatter: v => `${Number(v).toFixed(1)}%`, fill: '#94a3b8', fontSize: 11 }}>
              {ALL_MODELS.map((d, i) => <Cell key={i} fill={d.color} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Ensemble build-up line chart */}
      <div className="card-bordered">
        <SectionHeader title="Ensemble Build-Up Curve"
          subtitle="Incremental accuracy gain as each model is added to the ensemble" />
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={BUILDUP} margin={{ left: 10, right: 30, top: 15, bottom: 30 }}>
            <CartesianGrid stroke="#334155" strokeDasharray="3 3" />
            <XAxis dataKey="label" tick={{ fill: '#94a3b8', fontSize: 11 }}
              angle={-15} textAnchor="end" height={50} />
            <YAxis domain={[88, 100]} tickFormatter={v => `${v}%`}
              tick={{ fill: '#94a3b8', fontSize: 11 }} />
            <Tooltip formatter={v => [`${v}%`, 'Accuracy']}
              contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }} />
            <Line type="monotone" dataKey="accuracy" stroke="#00b4d8" strokeWidth={3}
              dot={{ fill: '#00b4d8', r: 5 }}
              label={{ position: 'top', formatter: v => `${v}%`, fill: '#00b4d8', fontSize: 10, dy: -8 }} />
          </LineChart>
        </ResponsiveContainer>
        <p className="text-xs text-slate-500 mt-2">
          Each deep model adds ~2% accuracy gain. Classical ML sub-ensembles contribute a further ~1% each
          by providing diverse voting signals trained on extracted features rather than raw pixels.
        </p>
      </div>

      {/* Feature source comparison */}
      <div className="card-bordered">
        <SectionHeader title="Classical ML Performance by Feature Source"
          subtitle="RF / DT / SVM accuracy with different feature extractors vs raw pixels" />
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={FEATURE_SOURCES} margin={{ left: 10, right: 20, top: 10, bottom: 5 }}>
            <XAxis dataKey="src" tick={{ fill: '#94a3b8', fontSize: 11 }} />
            <YAxis domain={[50, 95]} tickFormatter={v => `${v}%`}
              tick={{ fill: '#94a3b8', fontSize: 11 }} />
            <Tooltip formatter={v => [`${Number(v).toFixed(1)}%`]}
              contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }} />
            <Legend wrapperStyle={{ color: '#94a3b8', fontSize: 12 }} />
            {['RF', 'DT', 'SVM', 'Ens'].map((clf, i) => (
              <Bar key={clf} dataKey={clf} fill={CLF_COLORS[i]} radius={[3, 3, 0, 0]}
                name={clf === 'Ens' ? 'Ensemble' : clf} />
            ))}
          </BarChart>
        </ResponsiveContainer>
        <p className="text-xs text-slate-500 mt-2">
          Surprising finding: SVM on raw pixels (88.26%) outperforms SVM on CNN features (66.60%).
          CNN features compress spatial information that SVM's linear boundary cannot exploit,
          while raw pixels preserve the full 65,536-dimensional signal.
        </p>
      </div>

      {/* Model complexity */}
      <div className="card-bordered">
        <SectionHeader title="Model Complexity vs Accuracy"
          subtitle="Approximate parameter count (millions) plotted against test accuracy" />
        <ResponsiveContainer width="100%" height={260}>
          <ScatterChart margin={{ left: 20, right: 30, top: 20, bottom: 20 }}>
            <CartesianGrid stroke="#334155" strokeDasharray="3 3" />
            <XAxis dataKey="params" name="Parameters (M)" type="number"
              tickFormatter={v => `${v}M`} tick={{ fill: '#94a3b8', fontSize: 11 }}
              label={{ value: 'Parameters (M)', position: 'insideBottom', offset: -10, fill: '#94a3b8', fontSize: 11 }} />
            <YAxis dataKey="accuracy" name="Accuracy" domain={[80, 100]}
              tickFormatter={v => `${v}%`} tick={{ fill: '#94a3b8', fontSize: 11 }} />
            <ZAxis dataKey="size" range={[80, 300]} />
            <Tooltip
              cursor={{ strokeDasharray: '3 3', stroke: '#334155' }}
              content={({ payload }) => {
                if (!payload?.length) return null
                const d = payload[0].payload
                return (
                  <div className="bg-slate-800 border border-slate-600 rounded-lg p-3 text-xs">
                    <p className="font-bold text-white">{d.name}</p>
                    <p className="text-slate-400">Params: ~{d.params}M</p>
                    <p className="text-teal-400">Accuracy: {d.accuracy}%</p>
                  </div>
                )
              }}
            />
            <Scatter data={COMPLEXITY} fill="#00b4d8"
              label={{ dataKey: 'name', fill: '#94a3b8', fontSize: 10, dy: -12 }} />
          </ScatterChart>
        </ResponsiveContainer>
        <p className="text-xs text-slate-500 mt-2">
          The Final Ensemble (~48M total parameters across 6 models) achieves the best accuracy but at
          significantly higher computational cost. SVM on raw pixels offers the best
          accuracy-to-complexity ratio among single models.
        </p>
      </div>

      {/* Key findings */}
      <div className="card-bordered">
        <SectionHeader title="Key Comparative Findings" />
        <div className="grid sm:grid-cols-2 gap-4 text-sm">
          {[
            ['RO5 — Ensemble Superiority', 'Majority vote across 6 models (98.20%) significantly outperforms any individual model. The synergistic effect reduces individual model biases that are class-specific.'],
            ['RO6 — CNN vs Fine-Tuned', 'CNN standalone (94%) outperforms both Xception (86.91%) and Inception V3 (84.63%) standalone despite being custom-trained. Pre-trained ImageNet features are not always optimal for medical imaging.'],
            ['RO2 — CNN Features for ML', 'CNN-extracted features give the best classical ML results (RF 85.45%). The CNN learns domain-specific features more useful for classical classifiers than generic ImageNet features.'],
            ['RO4 — Raw Pixels Surprise', 'SVM on raw pixels (88.26%) beats SVM on any deep features. Classical ML can exploit full spatial resolution that deep feature compression discards.'],
          ].map(([title, text]) => (
            <div key={title} className="bg-slate-800/60 rounded-xl p-4">
              <p className="text-teal-400 font-bold text-xs mb-2">{title}</p>
              <p className="text-slate-300 text-xs leading-relaxed">{text}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
