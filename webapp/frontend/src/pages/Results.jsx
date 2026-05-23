import { useState, useCallback } from 'react'
import SectionHeader from '../components/SectionHeader'

const HF_BASE = 'https://huggingface.co/shehank98/brain-tumor-mri-models/resolve/main/results'

const NOTEBOOKS = [
  {
    id: '01_CNN_Standalone',
    label: '01 — CNN Standalone',
    color: 'border-purple-500',
    accent: 'text-purple-400',
    charts: [
      { file: '01_CNN_Standalone_training_history.jpg', title: 'Training History' },
      { file: '01_cnn_standalone_confusion_matrix.jpg', title: 'Confusion Matrix' },
      { file: '01_cnn_standalone_roc_curve.jpg',        title: 'ROC Curve' },
      { file: '01_cnn_standalone_pr_curve.jpg',         title: 'Precision-Recall Curve' },
    ],
  },
  {
    id: '02_CNN_Ensemble',
    label: '02 — CNN + Ensemble',
    color: 'border-purple-400',
    accent: 'text-purple-300',
    charts: [
      { file: 'cnn_random_forest_confusion_matrix.jpg',       title: 'RF — Confusion Matrix' },
      { file: 'cnn_random_forest_roc_curve.jpg',              title: 'RF — ROC Curve' },
      { file: 'cnn_decision_tree_confusion_matrix.jpg',       title: 'DT — Confusion Matrix' },
      { file: 'cnn_svm_confusion_matrix.jpg',                 title: 'SVM — Confusion Matrix' },
      { file: 'cnn_soft_vote_ensemble_confusion_matrix.jpg',  title: 'Ensemble — Confusion Matrix' },
      { file: 'cnn_soft_vote_ensemble_roc_curve.jpg',         title: 'Ensemble — ROC Curve' },
    ],
  },
  {
    id: '03_InceptionV3_Standalone',
    label: '03 — InceptionV3 Standalone',
    color: 'border-blue-500',
    accent: 'text-blue-400',
    charts: [
      { file: '03_InceptionV3_Standalone_training_history.jpg', title: 'Training History' },
      { file: '03_inceptionv3_standalone_confusion_matrix.jpg', title: 'Confusion Matrix' },
      { file: '03_inceptionv3_standalone_roc_curve.jpg',        title: 'ROC Curve' },
      { file: '03_inceptionv3_standalone_pr_curve.jpg',         title: 'Precision-Recall Curve' },
    ],
  },
  {
    id: '04_InceptionV3_Ensemble',
    label: '04 — InceptionV3 + Ensemble',
    color: 'border-blue-400',
    accent: 'text-blue-300',
    charts: [
      { file: 'inceptionv3_random_forest_confusion_matrix.jpg',      title: 'RF — Confusion Matrix' },
      { file: 'inceptionv3_random_forest_roc_curve.jpg',             title: 'RF — ROC Curve' },
      { file: 'inceptionv3_soft_vote_ensemble_confusion_matrix.jpg', title: 'Ensemble — Confusion Matrix' },
      { file: 'inceptionv3_soft_vote_ensemble_roc_curve.jpg',        title: 'Ensemble — ROC Curve' },
    ],
  },
  {
    id: '05_Xception_Standalone',
    label: '05 — Xception Standalone',
    color: 'border-teal-500',
    accent: 'text-teal-400',
    charts: [
      { file: '05_Xception_Standalone_training_history.jpg', title: 'Training History' },
      { file: '05_xception_standalone_confusion_matrix.jpg', title: 'Confusion Matrix' },
      { file: '05_xception_standalone_roc_curve.jpg',        title: 'ROC Curve' },
      { file: '05_xception_standalone_pr_curve.jpg',         title: 'Precision-Recall Curve' },
    ],
  },
  {
    id: '06_Xception_Ensemble',
    label: '06 — Xception + Ensemble',
    color: 'border-teal-400',
    accent: 'text-teal-300',
    charts: [
      { file: 'xception_random_forest_confusion_matrix.jpg',      title: 'RF — Confusion Matrix' },
      { file: 'xception_random_forest_roc_curve.jpg',             title: 'RF — ROC Curve' },
      { file: 'xception_decision_tree_confusion_matrix.jpg',      title: 'DT — Confusion Matrix' },
      { file: 'xception_svm_confusion_matrix.jpg',                title: 'SVM — Confusion Matrix' },
      { file: 'xception_soft_vote_ensemble_confusion_matrix.jpg', title: 'Ensemble — Confusion Matrix' },
      { file: 'xception_soft_vote_ensemble_roc_curve.jpg',        title: 'Ensemble — ROC Curve' },
    ],
  },
  {
    id: '07_TraditionalML_Ensemble',
    label: '07 — Traditional ML Baseline',
    color: 'border-slate-500',
    accent: 'text-slate-300',
    charts: [
      { file: 'traditional_ml_random_forest_confusion_matrix.jpg',      title: 'RF — Confusion Matrix' },
      { file: 'traditional_ml_svm_confusion_matrix.jpg',                title: 'SVM — Confusion Matrix' },
      { file: 'traditional_ml_decision_tree_confusion_matrix.jpg',      title: 'DT — Confusion Matrix' },
      { file: 'traditional_ml_soft_vote_ensemble_confusion_matrix.jpg', title: 'Ensemble — Confusion Matrix' },
      { file: 'traditional_ml_soft_vote_ensemble_roc_curve.jpg',        title: 'Ensemble — ROC Curve' },
    ],
  },
  {
    id: '08_AllModelsCombined_FinalEnsemble',
    label: '08 — Final 6-Model Ensemble',
    color: 'border-amber-500',
    accent: 'text-amber-400',
    charts: [
      { file: 'all_models_accuracy_comparison.jpg',     title: 'All Models Accuracy Comparison' },
      { file: 'final_ensemble_confusion_matrix.jpg',    title: 'Final Ensemble — Confusion Matrix' },
      { file: 'final_ensemble_roc_curve.jpg',           title: 'Final Ensemble — ROC Curve' },
      { file: 'final_ensemble_pr_curve.jpg',            title: 'Final Ensemble — PR Curve' },
    ],
  },
  {
    id: '09_ConsensusGradCAM',
    label: '09 — Consensus GRAD-CAM (EAA-IoU)',
    color: 'border-cyan-500',
    accent: 'text-cyan-400',
    charts: [
      { file: 'sample_gradcam_grid.jpg',      title: 'All-6-Model GRAD-CAM Grid' },
      { file: 'eaa_iou_distribution.jpg',     title: 'EAA-IoU Distribution by Class' },
      { file: 'pairwise_iou_heatmap.jpg',     title: 'Pairwise IoU Matrix (6×6)' },
      { file: 'eaa_iou_low_vs_high.jpg',      title: 'Low vs High EAA-IoU Examples' },
    ],
  },
  {
    id: '10_ConfidenceVoting',
    label: '10 — Confidence-Weighted Voting',
    color: 'border-lime-500',
    accent: 'text-lime-400',
    charts: [
      { file: 'voting_comparison_confusion_matrix.jpg', title: 'Voting — Confusion Matrices' },
      { file: 'voting_comparison_per_class_f1.jpg',     title: 'Per-Class F1 Improvement' },
      { file: 'voting_confidence_distribution.jpg',     title: 'Confidence Distribution' },
    ],
  },
  {
    id: '11_XAIAnalysis',
    label: '11 — XAI Analysis',
    color: 'border-rose-500',
    accent: 'text-rose-400',
    charts: [
      { file: 'eaa_iou_roc_pr_curve.jpg',           title: 'EAA-IoU ROC & PR Curves' },
      { file: 'eaa_iou_by_class_boxplot.jpg',        title: 'EAA-IoU by Class (Box Plots)' },
      { file: 'escalation_threshold_analysis.jpg',   title: 'Escalation Threshold Analysis' },
      { file: 'eaa_iou_vs_confidence_scatter.jpg',   title: 'EAA-IoU vs Vote Confidence' },
    ],
  },
]

function ChartCard({ notebook_id, file, title, accent, cacheBust }) {
  const [status, setStatus] = useState('loading') // 'loading' | 'ok' | 'missing'
  const url = `${HF_BASE}/${notebook_id}/${file}${cacheBust ? `?v=${cacheBust}` : ''}`
  return (
    <div className="bg-slate-800 rounded-xl overflow-hidden border border-slate-700 flex flex-col">
      <div className="relative bg-slate-900 flex items-center justify-center min-h-[200px]">
        {status === 'loading' && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="w-8 h-8 border-2 border-teal-400 border-t-transparent rounded-full animate-spin" />
          </div>
        )}
        {status === 'missing' && (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-500 gap-2 p-4 text-center">
            <svg className="w-10 h-10 opacity-40" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                d="M9 13h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414A1 1 0 0119 9.414V19a2 2 0 01-2 2z" />
            </svg>
            <span className="text-sm">Not generated yet<br />Run notebook to produce this chart</span>
          </div>
        )}
        <img
          src={url}
          alt={title}
          className={`w-full object-contain transition-opacity duration-300 ${status === 'ok' ? 'opacity-100' : 'opacity-0 absolute'}`}
          onLoad={() => setStatus('ok')}
          onError={() => setStatus('missing')}
        />
      </div>
      <div className="px-3 py-2 text-center">
        <p className={`text-xs font-medium ${accent}`}>{title}</p>
      </div>
    </div>
  )
}

export default function Results() {
  const [active, setActive] = useState(null)
  const [cacheBust, setCacheBust] = useState(0)
  const refresh = useCallback(() => setCacheBust(Date.now()), [])

  const displayed = active ? NOTEBOOKS.filter(n => n.id === active) : NOTEBOOKS

  return (
    <div className="space-y-10">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <SectionHeader
          title="Notebook Results Gallery"
          subtitle="Training charts, confusion matrices, ROC curves, and precision-recall curves generated by each notebook. Images are hosted on HuggingFace and appear here automatically after each Colab run."
        />
        <button
          onClick={refresh}
          className="shrink-0 flex items-center gap-2 px-4 py-2 rounded-lg bg-teal-500/10 border border-teal-500/30 text-teal-400 text-sm font-medium hover:bg-teal-500/20 transition-colors"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round"
              d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          Refresh Images
        </button>
      </div>

      {/* Filter pills */}
      <div className="flex flex-wrap gap-2">
        <button
          onClick={() => setActive(null)}
          className={`px-3 py-1.5 rounded-full text-xs font-medium border transition-colors ${
            active === null
              ? 'bg-teal-400/15 text-teal-400 border-teal-400/40'
              : 'text-slate-400 border-slate-600 hover:border-slate-400'
          }`}
        >
          All notebooks
        </button>
        {NOTEBOOKS.map(nb => (
          <button
            key={nb.id}
            onClick={() => setActive(active === nb.id ? null : nb.id)}
            className={`px-3 py-1.5 rounded-full text-xs font-medium border transition-colors ${
              active === nb.id
                ? `bg-teal-400/15 text-teal-400 border-teal-400/40`
                : `text-slate-400 border-slate-600 hover:border-slate-400`
            }`}
          >
            {nb.label.split(' — ')[0]}
          </button>
        ))}
      </div>

      {/* Notice */}
      <div className="flex items-start gap-3 bg-slate-800/50 border border-slate-700 rounded-xl p-4 text-sm text-slate-400">
        <svg className="w-5 h-5 text-teal-400 mt-0.5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
            d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        <p>
          Charts appear here automatically after you run each notebook in Google Colab —
          Section 10 uploads them to HuggingFace. If a chart shows a placeholder,
          that notebook has not been run yet.
        </p>
      </div>

      {/* Gallery per notebook */}
      {displayed.map(nb => (
        <section key={nb.id} className={`border-l-4 ${nb.color} pl-5`}>
          <h2 className={`text-lg font-semibold mb-4 ${nb.accent}`}>{nb.label}</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {nb.charts.map(chart => (
              <ChartCard
                key={`${chart.file}-${cacheBust}`}
                notebook_id={nb.id}
                file={chart.file}
                title={chart.title}
                accent={nb.accent}
                cacheBust={cacheBust}
              />
            ))}
          </div>
        </section>
      ))}
    </div>
  )
}
