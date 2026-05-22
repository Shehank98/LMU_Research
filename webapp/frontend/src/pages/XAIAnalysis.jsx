import SectionHeader from '../components/SectionHeader'

const NOVELTY_TABLE = [
  {
    paper: 'Kakon et al. Cancers 2025',
    arch: 'EfficientNetB7 + Inc + Xcp',
    voting: 'Soft + LightGBM meta',
    perModelCam: true,
    heatmapFusion: false,
    interModelIou: false,
  },
  {
    paper: 'ScienceDirect Comput Biol Med 2025',
    arch: 'MobileNetV2 + DenseNet121',
    voting: 'Soft vote',
    perModelCam: true,
    heatmapFusion: false,
    interModelIou: false,
  },
  {
    paper: 'Advanced Dynamic Ens. Sci Rep 2025',
    arch: 'CNN + ResNet-50 + EfficientNet-B5',
    voting: 'Adaptive weights',
    perModelCam: true,
    heatmapFusion: false,
    interModelIou: false,
  },
  {
    paper: 'Two-step Majority Vote ResearchGate 2025',
    arch: 'Inc+Xcp+DenseNet+EffNet+ResNet',
    voting: 'Majority vote',
    perModelCam: true,
    heatmapFusion: false,
    interModelIou: false,
  },
  {
    paper: 'This Study',
    arch: 'CNN + Xception + InceptionV3 + Classical ML',
    voting: 'Confidence-weighted',
    perModelCam: true,
    heatmapFusion: true,
    interModelIou: true,
    isOwn: true,
  },
]

const IOU_THRESHOLDS = [
  { range: '≥ 0.65', level: 'Strong Agreement', color: 'text-green-400', bg: 'bg-green-500/10 border-green-500/30', desc: 'Models focus on the same region. High clinical confidence — proceed with ensemble prediction.' },
  { range: '0.40 – 0.65', level: 'Moderate Agreement', color: 'text-amber-400', bg: 'bg-amber-500/10 border-amber-500/30', desc: 'Some divergence in attention regions. Radiologist review recommended, especially for borderline predictions.' },
  { range: '< 0.40', level: 'Low Agreement', color: 'text-red-400', bg: 'bg-red-500/10 border-red-500/30', desc: 'Models disagree on the tumor region. Escalate to expert review regardless of predicted class.' },
]

const CLASS_IOU = [
  { cls: 'Glioma', expected: 'Low (most variable)', reason: 'Morphologically similar to meningioma on T1 MRI. Highest misclassification rate in all models.' },
  { cls: 'Meningioma', expected: 'Moderate', reason: 'Well-defined extra-axial location, but can be confused with glioma in boundary cases.' },
  { cls: 'No Tumor', expected: 'High', reason: 'Models reliably agree on absence of lesion. Lowest misclassification across all architectures.' },
  { cls: 'Pituitary', expected: 'High', reason: 'Distinctive sella turcica location. Models converge on the same anatomical region.' },
]

function Check({ yes }) {
  return yes
    ? <span className="text-green-400 font-bold">✓</span>
    : <span className="text-slate-600">✗</span>
}

export default function XAIAnalysis() {
  return (
    <div className="space-y-10">
      <div>
        <h2 className="text-2xl font-extrabold text-white">🔍 XAI Analysis</h2>
        <p className="text-slate-400 mt-1">Explainable AI methodology, novelty positioning, and EAA-IoU analysis</p>
      </div>

      {/* EAA-IoU explanation */}
      <div className="card-bordered">
        <SectionHeader title="Ensemble Attention Agreement (EAA-IoU)"
          subtitle="Novel inter-model saliency consensus metric — no ground truth annotations required" />
        <div className="space-y-4 text-sm text-slate-300 leading-relaxed">
          <p>
            Standard GRAD-CAM compares a model's attention map against a radiologist-drawn bounding box.
            This requires labelled segmentation data that the Kaggle Brain Tumor MRI dataset does not provide.
          </p>
          <p>
            <strong className="text-white">EAA-IoU</strong> instead asks: <em>do the three architecturally distinct
            models agree with each other</em> on which region drove the prediction? Low inter-model agreement is
            a self-supervised uncertainty signal — no annotation required.
          </p>
          <div className="bg-slate-800/80 rounded-xl p-4 font-mono text-xs text-teal-300 space-y-1">
            <div>Input MRI → CNN   → GRAD-CAM heatmap A (256×256, normalised 0–1)</div>
            <div>Input MRI → Xception    → GRAD-CAM heatmap B</div>
            <div>Input MRI → InceptionV3 → GRAD-CAM heatmap C</div>
            <div className="mt-2">Binarise each at threshold t = 0.5</div>
            <div>Pairwise IoU: IoU(A,B), IoU(A,C), IoU(B,C)</div>
            <div className="text-white font-bold mt-1">EAA-IoU = mean( IoU(A,B), IoU(A,C), IoU(B,C) )</div>
          </div>
          <p>
            A weighted-average pixel map of the three heatmaps forms the <strong className="text-white">Consensus
            Heatmap</strong> — a single overlay showing where the ensemble collectively focused its attention.
          </p>
        </div>
      </div>

      {/* IoU threshold table */}
      <div className="card-bordered">
        <SectionHeader title="EAA-IoU Clinical Thresholds"
          subtitle="Three-tier escalation rule based on inter-model attention agreement" />
        <div className="space-y-3">
          {IOU_THRESHOLDS.map(t => (
            <div key={t.range} className={`rounded-xl border p-4 ${t.bg}`}>
              <div className="flex items-center gap-3 mb-1">
                <span className={`font-mono font-extrabold text-lg ${t.color}`}>{t.range}</span>
                <span className={`font-bold ${t.color}`}>{t.level}</span>
              </div>
              <p className="text-sm text-slate-300">{t.desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Research questions */}
      <div className="card-bordered">
        <SectionHeader title="Research Questions Addressed by EAA-IoU" />
        <ol className="space-y-3 text-sm text-slate-300 list-decimal list-inside">
          <li>Does low EAA-IoU correlate with misclassification on the 2,063 official test images?</li>
          <li>Which tumor class produces the lowest average EAA-IoU? (Hypothesis: glioma — matches confusion matrix pattern)</li>
          <li>What precision/recall tradeoff does an EAA-IoU escalation threshold give?</li>
          <li>Can EAA-IoU flag the 1.8% of cases the ensemble misclassifies without retraining?</li>
        </ol>
        <div className="mt-4 bg-teal-500/10 border border-teal-500/30 rounded-xl p-4">
          <p className="text-sm text-teal-300 font-medium">New Research Question</p>
          <p className="text-sm text-slate-300 mt-1">
            Does inter-model GRAD-CAM agreement (EAA-IoU) correlate with misclassification in a heterogeneous
            6-model brain tumor ensemble, and can it serve as a per-sample clinical uncertainty trigger?
          </p>
        </div>
      </div>

      {/* Class-conditional analysis */}
      <div className="card-bordered">
        <SectionHeader title="Expected Class-Conditional EAA-IoU Patterns"
          subtitle="Predicted attention agreement by tumor class based on morphological characteristics" />
        <div className="grid sm:grid-cols-2 gap-4">
          {CLASS_IOU.map(c => (
            <div key={c.cls} className="bg-slate-800/60 rounded-xl p-4">
              <div className="text-white font-bold mb-1">{c.cls}</div>
              <div className={`text-xs font-semibold mb-2 ${
                c.expected.startsWith('Low') ? 'text-red-400' :
                c.expected.startsWith('Moderate') ? 'text-amber-400' : 'text-green-400'
              }`}>{c.expected} IoU</div>
              <p className="text-xs text-slate-400">{c.reason}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Confidence-weighted voting */}
      <div className="card-bordered">
        <SectionHeader title="Confidence-Weighted Voting"
          subtitle="Known method — implemented and cited correctly" />
        <div className="space-y-3 text-sm text-slate-300">
          <p>
            Softmax-probability-weighted voting (summing 6 probability vectors) replaces majority vote.
            This is an <strong className="text-amber-400">established method</strong> — published in
            TSO-Optimised Voting (Informatica 2024), arXiv 2603.28357, and MDPI Diagnostics 2025.
            It is not a novel contribution on its own.
          </p>
          <div className="bg-slate-800/80 rounded-xl p-4 font-mono text-xs text-teal-300 space-y-1">
            <div className="text-slate-500"># Replace majority vote with confidence-weighted sum:</div>
            <div>combined_probs = prob_cnn + prob_inc + prob_xcp</div>
            <div>                + prob_cnn_ens + prob_inc_ens + prob_xcp_ens</div>
            <div>final_class    = argmax(combined_probs)</div>
            <div>vote_confidence = max(combined_probs) / sum(combined_probs)</div>
          </div>
          <p>
            The combination of <strong className="text-white">confidence-weighted vote + EAA-IoU uncertainty
            flag</strong> is the novel system. EAA-IoU adds a safety layer on top of the voting result
            without retraining any model.
          </p>
        </div>
      </div>

      {/* Novelty comparison table */}
      <div className="card-bordered">
        <SectionHeader title="Novelty Gap Analysis — Published Literature"
          subtitle="Columns 4 and 5 are unoccupied in all published brain tumor ensemble papers" />
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-slate-400 border-b border-slate-700 uppercase text-[10px]">
                <th className="text-left py-2 pr-3">Paper</th>
                <th className="text-left py-2 pr-3">Architecture</th>
                <th className="text-left py-2 pr-3">Voting</th>
                <th className="text-center py-2 px-2">Per-model<br/>GRAD-CAM</th>
                <th className="text-center py-2 px-2">Heatmap<br/>Fusion</th>
                <th className="text-center py-2 px-2">Inter-model<br/>IoU</th>
              </tr>
            </thead>
            <tbody>
              {NOVELTY_TABLE.map(row => (
                <tr key={row.paper}
                  className={`border-b border-slate-700/40 ${row.isOwn ? 'bg-teal-500/5' : 'hover:bg-slate-700/20'}`}>
                  <td className={`py-2 pr-3 ${row.isOwn ? 'text-teal-300 font-bold' : 'text-slate-300'}`}>{row.paper}</td>
                  <td className="py-2 pr-3 text-slate-400">{row.arch}</td>
                  <td className="py-2 pr-3 text-slate-400">{row.voting}</td>
                  <td className="py-2 px-2 text-center"><Check yes={row.perModelCam} /></td>
                  <td className="py-2 px-2 text-center"><Check yes={row.heatmapFusion} /></td>
                  <td className="py-2 px-2 text-center"><Check yes={row.interModelIou} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="mt-4 bg-slate-800/60 rounded-xl p-4">
          <p className="text-xs text-slate-400 leading-relaxed italic">
            "Unlike existing ensemble-CAM methods that compare model attention against radiologist-annotated
            bounding boxes, this work introduces Ensemble Attention Agreement (EAA-IoU) — a per-sample
            inter-model saliency consensus score computed from architecturally distinct deep models
            (CNN, Xception, InceptionV3). EAA-IoU is used not as an accuracy metric but as a clinical
            uncertainty signal: low inter-model agreement triggers escalation to radiologist review,
            independent of the ensemble's final class prediction."
          </p>
        </div>
      </div>

      {/* Papers to cite */}
      <div className="card-bordered">
        <SectionHeader title="Key Citations for XAI Section" />
        <div className="space-y-4">
          <div>
            <p className="text-xs text-teal-400 font-semibold uppercase tracking-wide mb-2">Confidence-Weighted Voting (known method — cite these)</p>
            <ul className="space-y-1 text-xs text-slate-400">
              <li>• TSO-Optimised Weighted Soft Voting (2024) <em>Informatica</em> — InceptionV3+Xception, 99.92%</li>
              <li>• Optimised Weighted Voting arXiv 2603.28357 (2025)</li>
              <li>• Majority Voting Ensemble <em>MDPI Diagnostics</em> 2025 (PMC12293199) — 14 models, 99.8%</li>
            </ul>
          </div>
          <div>
            <p className="text-xs text-teal-400 font-semibold uppercase tracking-wide mb-2">Standard GRAD-CAM in Brain Tumor (known method)</p>
            <ul className="space-y-1 text-xs text-slate-400">
              <li>• Grad-CAM + SHAP + LIME <em>Med Eng Phys</em> ScienceDirect 2025 — accuracy 97.2% → 99.4%</li>
              <li>• Grad-CAM ResNet50 <em>BMC Medical Imaging</em> 2024 (PMC11088067)</li>
            </ul>
          </div>
          <div>
            <p className="text-xs text-teal-400 font-semibold uppercase tracking-wide mb-2">Closest Prior Work for Heatmap Fusion (different domain)</p>
            <ul className="space-y-1 text-xs text-slate-400">
              <li>• Aasem & Iqbal "Ensemble-CAM" <em>Frontiers in Big Data</em> 2024 (PMC11096460) — chest X-ray, IoU vs annotation</li>
              <li>• Multi-Model Heatmap Fusion arXiv 2507.00234 (2025) — time-series, CNN+Transformer</li>
            </ul>
          </div>
        </div>
      </div>

      <p className="text-xs text-slate-500 text-center">
        ⚠ <strong>Research Demonstration Only.</strong> EAA-IoU thresholds are empirically derived from the Kaggle test set.
        Validate on independent clinical data before any medical application.
      </p>
    </div>
  )
}
