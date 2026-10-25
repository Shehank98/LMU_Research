import { Card, PageHeader } from '../components/ui'

const STEPS = [
  ['Train and evaluate', 'All models are trained in Google Colab with one shared data pipeline, seed and test split. Every metric is logged as CSV and JSON.'],
  ['Publish', 'The trained models and the result files are uploaded to the Hugging Face Hub.'],
  ['Serve', 'This web application downloads the models from Hugging Face at start-up and runs them on each uploaded scan.'],
]

const MODELS = [
  ['Custom CNN', 'Four convolution and pooling blocks trained from scratch on 256 x 256 images.'],
  ['InceptionV3 and Xception', 'ImageNet-pretrained networks with a frozen backbone and a new four-class head (transfer learning).'],
  ['Feature ensembles', 'A 256-value feature vector is read from each network and classified by Random Forest, Decision Tree and SVM with soft voting.'],
  ['Final vote', 'The six probability vectors are summed and the highest class wins (confidence-weighted voting).'],
]

export default function About() {
  return (
    <>
      <PageHeader title="About this project"
        description="A comparative study of CNNs, fine-tuned models and ensemble learning for multi-class brain tumour MRI classification, with an explainable agreement signal." />
      <div className="grid gap-6 lg:grid-cols-3">
        <div className="space-y-6 lg:col-span-2">
          <Card title="How a prediction is made">
            <ol className="space-y-4">
              {MODELS.map(([name, text], i) => (
                <li key={name} className="flex gap-4">
                  <span className="grid h-7 w-7 shrink-0 place-items-center rounded-full bg-brand-50 text-xs font-semibold text-brand-700">{i + 1}</span>
                  <div><p className="text-sm font-medium text-slate-900">{name}</p><p className="mt-0.5 text-sm text-slate-600">{text}</p></div>
                </li>
              ))}
            </ol>
          </Card>
          <Card title="Ensemble Attention Agreement (EAA-IoU)">
            <div className="space-y-3 text-sm leading-relaxed text-slate-600">
              <p>Grad-CAM produces a heatmap of the image regions each deep model relied on. The CNN, Xception and InceptionV3 maps are thresholded and compared in pairs using Intersection over Union. Their mean is the EAA-IoU score.</p>
              <p>A high score means the three architecturally different models look at the same region. A low score means they disagree about where the tumour is, so the prediction is flagged for radiologist review. Unlike earlier work, no radiologist-drawn tumour boundaries are needed.</p>
            </div>
          </Card>
          <Card title="From training to this application">
            <div className="grid gap-4 sm:grid-cols-3">
              {STEPS.map(([t, d], i) => (
                <div key={t} className="rounded-lg border border-slate-200 p-4">
                  <p className="text-xs font-semibold text-brand-700">Step {i + 1}</p>
                  <p className="mt-1 text-sm font-medium text-slate-900">{t}</p>
                  <p className="mt-1 text-sm text-slate-600">{d}</p>
                </div>
              ))}
            </div>
          </Card>
        </div>
        <div className="space-y-6">
          <Card title="Project">
            <dl className="space-y-3 text-sm">
              {[['Student', 'Pattiyage Shehan Kavishka (E187041)'], ['Programme', 'BSc (Hons) Data Science'],
                ['Supervisor', 'Ms. Hiruni Wimalaweera'], ['Institution', 'London Metropolitan University, ESOFT Metro Campus'],
                ['Dataset', 'Kaggle Brain Tumor MRI, 9,047 images, 4 classes']].map(([k, v]) => (
                <div key={k}><dt className="text-xs text-slate-500">{k}</dt><dd className="font-medium text-slate-800">{v}</dd></div>
              ))}
            </dl>
          </Card>
          <Card title="Technology">
            <ul className="space-y-1.5 text-sm text-slate-600">
              <li>TensorFlow and Keras for the deep models</li>
              <li>scikit-learn for the classical models</li>
              <li>FastAPI backend, React frontend</li>
              <li>Hugging Face Hub for model hosting</li>
            </ul>
          </Card>
          <div className="rounded-xl border border-amber-200 bg-amber-50 p-5 text-sm text-amber-800">
            This is a research prototype and is not approved for clinical use. Results must always be reviewed by a qualified radiologist.
          </div>
        </div>
      </div>
    </>
  )
}
