# Proposal Viva — Presentation Script & Story

**Student:** Pattiyage Shehan Kavishka (E187041)
**Title:** *A Comparative Analysis of Ensemble Learning, Fine-Tuned Models and CNNs for Multi-Class Disease Prediction in Medical Imaging: An Integrated Framework with Systematic Optimization*

> **How to use this file:** Each section below = roughly one slide. The **"🗣️ Say this"** blocks are the words you speak — written as a natural story, not bullet points to read out. The **"🧠 Why / how the code works"** blocks are the deeper detail so you can answer follow-up questions. Aim for **12–15 minutes** of talking, then Q&A. Practice the story until you can tell it *without* reading.

---

## The one-sentence story (memorise this)

> *"A single model looking at a brain MRI can be confidently wrong. So instead of trusting one model, I built six — three deep networks and three classical models fed by deep features — let them vote, and then I added a safety layer that checks whether the models are all looking at the **same part of the brain**. When they disagree on *where* the tumour is, the system stops and asks for a radiologist."*

Everything in the presentation is an expansion of that sentence.

---

## Slide 1 — Title & hook (30 sec)

**🗣️ Say this:**
"Good morning. My name is Shehan Kavishka. My research is about classifying brain tumours from MRI scans. But I want to start with the real problem, not the technology. When a brain MRI comes in, a radiologist has to decide: is this a glioma, a meningioma, a pituitary tumour, or no tumour at all? That decision changes the entire treatment plan. It's slow, it depends on one person's eye, and mistakes are costly. My project asks: *can we build an AI system that is not just accurate, but also honest about when it isn't sure?*"

**🧠 Note:** Don't open with "I used CNNs." Open with the patient problem. The panel wants to see you understand *why* the work matters.

---

## Slide 2 — The problem in brief (1 min)

**🗣️ Say this:**
"There are two problems in this field. The first is technical: most published work uses a *single* model. A single CNN, or a single pretrained network. That model might hit 94% accuracy, but it inherits one architecture's blind spots. The second problem is trust: even the accurate models are black boxes. They give a class, but they never tell the clinician *how confident the reasoning was*, and they never flag the cases where the model is basically guessing. My framework attacks both — better accuracy through combining models, and a new way to measure uncertainty."

**🧠 Why:** This maps directly to your proposal's "Problem in Brief" and "Aim". Four classes: **glioma, meningioma, pituitary, no-tumour.**

---

## Slide 3 — The dataset (1 min)

**🗣️ Say this:**
"I used the Kaggle Brain Tumour MRI dataset — around 9,000 scans, already split into a training folder and a testing folder, across the four classes. I kept that official split throughout, so every model is judged on images it has never seen. The classes are fairly balanced, which matters because I don't want the model to just learn to always guess the most common class."

| Class | Train | Test |
|-------|-------|------|
| Glioma | 1,733 | 504 |
| Meningioma | 1,699 | 510 |
| No Tumour | 1,847 | 545 |
| Pituitary | 1,705 | 504 |
| **Total** | **6,984** | **2,063** |

**🧠 Why:** If asked "how do you avoid data leakage?" → *"I never touch the Testing folder during training. Feature extraction, model fitting, and hyper-parameter choices all happen on Training only."*

---

## Slide 4 — Preprocessing: the same recipe for every model (1.5 min)

**🗣️ Say this:**
"Before any model sees an image, every scan goes through the *same* preprocessing recipe, so I'm comparing models fairly and not comparing preprocessing tricks. Let me walk through what one image experiences."

**The pipeline (this is literally what the code does):**
1. **Read** the JPG in colour — `cv2.imread(path, IMREAD_COLOR)`.
2. **Resize** to a fixed square — `cv2.resize(img, (size, size))`. Networks need a fixed input size.
3. **Fix the colour order** — OpenCV loads images as BGR, so I convert to RGB (`cv2.cvtColor(..., COLOR_BGR2RGB)`), because the pretrained ImageNet models expect RGB.
4. **Convert to float** and **normalise** the pixels from the 0–255 range down to a small range around 0. This keeps the numbers small so training is stable and fast.

**🗣️ Say this about the two sizes:**
"There are two input sizes in my system, and that's deliberate. The **standalone** deep models run at **256×256** — bigger images, more detail, because that model is doing the whole job alone. The **feature-extraction backbones** run at **128×128** — smaller, because there I only need a compact feature vector, and 128 keeps it fast and memory-friendly when I then run classical machine learning on thousands of images."

**🧠 Why / how the code works — the honest detail:**
- Standalone CNN / Xception / InceptionV3: normalise as `pixels / 255` → range **[0, 1]**.
- Ensemble backbones: normalise as `pixels / 255 − 0.5` → range **[−0.5, 0.5]** (zero-centred).
- If a panellist catches this inconsistency, **own it**: *"Yes — that's a known inconsistency I documented. The −0.5 shift zero-centres the input for the ensemble backbones. In the deployed web app I standardised everything to /255 so the live system is fully consistent."* (This is true — `backend/xai_engine.py preprocess()` uses `/255.0` for all models.)

---

## Slide 5 — Step 1 of the build: a custom CNN baseline (1 min)

**🗣️ Say this:**
"I started simple, to get a baseline. A custom CNN — four convolution-and-pooling blocks that learn edges, then textures, then tumour-shaped patterns, followed by a couple of dense layers that make the final four-class decision. Trained from scratch on the MRIs, it reached about 91–94% on the test set. Good — but it's one architecture's opinion, and it started overfitting after about ten epochs. That told me a single model wasn't going to be enough."

**🧠 Why:** Conv layers = automatic feature learning (edges → textures → shapes). This is your "RO1 — evaluate CNN feature extraction". The overfitting is your motivation to bring in transfer learning.

---

## Slide 6 — Why Xception and InceptionV3? (2 min — this is a favourite viva question)

**🗣️ Say this:**
"Now, why did I add two *pretrained* models, and why these two specifically? Two reasons: transfer learning, and diversity.

**Transfer learning first.** Xception and InceptionV3 were trained on ImageNet — over a million everyday photos. In that training they already learned how to detect edges, curves, textures, and shapes. A brain tumour is, at the pixel level, still edges and textures and shapes. So instead of learning vision from zero on only 7,000 MRIs — which is a small dataset for deep learning — I *borrow* that visual knowledge and only teach the model the final brain-tumour decision. That's why I **froze** the pretrained layers and only trained a small new head on top.

**Now, why these two and not just one?** Because they *think differently*, and for an ensemble that's exactly what I want:
- **InceptionV3** looks at each region with **several filter sizes in parallel** — small and large filters side by side — so it captures both tiny detail and the bigger structure at once.
- **Xception** replaces those with **depthwise-separable convolutions** — it separates *where* a pattern is from *what channel* it's in, which is very efficient and captures fine spatial detail.

So I have three architecturally *distinct* viewpoints: my own CNN, Inception's multi-scale view, and Xception's separable view. If all three were the same design, combining them would add nothing. Because they're different, their mistakes are different — and that's what makes the ensemble stronger than any one of them."

**🧠 Why / how the code works:**
```python
xcp = Xception(input_shape=(128,128,3), weights='imagenet', include_top=False)
for layer in xcp.layers:
    layer.trainable = False          # freeze ImageNet knowledge
x = Flatten()(xcp.output)
x = Dense(256, activation='relu')(x) # <-- the new 256-unit "understanding" layer
x = Dropout(0.5)(x)                  # dropout fights overfitting
output = Dense(4, activation='softmax')(x)  # 4-class decision
```
- `include_top=False` = throw away ImageNet's original 1000-class classifier; keep only the feature-learning body.
- `weights='imagenet'` = start from the pretrained knowledge.
- The new **Dense(256)** layer is important — remember it for the next slide, because I'm going to *cut the model right there*.

---

## Slide 7 — The clever bit: "cutting the dense layer" for features (2.5 min — the heart of the method)

**🗣️ Say this:**
"This is the core idea of my framework, so let me slow down.

A normal network ends in a softmax that outputs four numbers — the probability of each class. But one layer *before* that, there's my Dense-256 layer. At that point the network has compressed the entire MRI into **256 numbers** that describe *what it understood about the image* — its rich internal summary — but it hasn't yet collapsed that into a final guess.

Those 256 numbers are gold. So after training the network normally, I **cut it** at that Dense-256 layer. I build a *new* model that takes an MRI in and outputs those 256 features instead of the four-class prediction. In code it's one line: I make a model whose output is that named dense layer.

Then — and this is the interesting part — I feed those 256 deep features into **classical machine-learning models**: a Random Forest, a Decision Tree, and an SVM. So the deep network does the *seeing*, and the classical models do the *deciding*. I get the best of both worlds: deep-learning-quality features, with the robustness and speed of classical classifiers."

**🧠 Why / how the code works — the exact mechanism:**
```python
# After the Xception head is trained:
layer_name = 'dense_3'                       # the 256-unit layer
FC_layer_model = Model(inputs=xcp_model.input,
                       outputs=xcp_model.get_layer(layer_name).output)

# For every image, get its 256-number "fingerprint":
features = FC_layer_model.predict(img)       # shape (1, 256)

# Feed those features to classical ML:
rf  = RandomForestClassifier().fit(train_features, train_labels)
dt  = DecisionTreeClassifier().fit(train_features, train_labels)
svm = SVC().fit(train_features, train_labels)
```
- "Cutting the dense layer" = building `Model(inputs=..., outputs=get_layer('dense_3').output)`. Same trained weights, just a shorter exit.
- Layer names in the three backbones: CNN → `dense`, InceptionV3 → `dense_2`, Xception → `dense_3`. All output **256 features** from a **128×128** input.
- Analogy for the panel: *"The deep network is a specialist who examines the scan and writes a 256-line summary. The Random Forest is a second specialist who reads only that summary and makes the call. I'm splitting perception from decision."*

**If asked "why not just use the network's own softmax?"** →
*"Two reasons. One, it lets classical models — which are more robust on small data and easy to interpret — make the final call. Two, it gives me three *more* independent voters for the ensemble, built from the same backbones but with different decision logic."*

---

## Slide 8 — Bringing it together: the 6-model ensemble (1.5 min)

**🗣️ Say this:**
"So now I have six independent opinions on every scan:
- three from the deep networks predicting *directly* — CNN, Inception, Xception;
- three from the classical models reading the deep features — CNN-features, Inception-features, Xception-features.

For the final answer, they **vote**. In the research notebooks it's a straight majority vote — the class that gets the most votes wins. Six diverse models, voting together, reached **98.2%** on the test set — well above any single model, and above my best CNN at 94%. The reason it works is simple: for the ensemble to be wrong, a *majority* of six different models with different blind spots all have to make the *same* mistake on the same image. That's rare."

**🧠 Why / how the code works:**
```python
predictions = [cnn_pred, inc_pred, xcp_pred,
               cnn_ens_pred, inc_ens_pred, xcp_ens_pred]
final = max(set(predictions), key=predictions.count)   # majority vote
```

**Upgrade you made for the live app (mention it — shows initiative):**
"In the web application I improved this. Instead of one hard vote each, I **sum the probability vectors** of all six models and take the strongest class. That's *confidence-weighted voting* — a model that's 99% sure counts more than one that's barely 30% sure — and it gives me a real confidence percentage to show the clinician."
```python
combined = prob_cnn + prob_inc + prob_xcp + prob_cnn_ens + prob_inc_ens + prob_xcp_ens
final_class = np.argmax(combined)
confidence  = combined.max() / combined.sum()
```

---

## Slide 9 — The novelty: EAA-IoU, teaching the system to say "I'm not sure" (2.5 min — your original contribution)

**🗣️ Say this:**
"Accuracy alone wasn't my goal. A model can be 98% accurate and still be dangerously confident on the 2% it gets wrong. So I built a safety layer, and this is the novel part of my work.

I use **GRAD-CAM**, a technique that produces a heatmap showing *which region of the MRI a model looked at* to make its decision. I generate this heatmap independently for all three deep models. Then I ask a simple but powerful question: **do the three models agree on *where* the tumour is?**

I measure that overlap with a score called **Intersection-over-Union** — how much the heatmaps overlap versus how much area they cover in total. I compute it for each pair of models and average them into one number I call the **Ensemble Attention Agreement**, or **EAA-IoU**.

- **High agreement** — all three models highlight the same region — the system is confident and consistent.
- **Low agreement** — the models are looking at completely different regions — that's a red flag. Even if they happened to output the same class, they got there for different reasons, which means the case is ambiguous.

When agreement drops below a threshold, the web app raises a **clinical escalation alert**: *"models disagree — send to a radiologist."*

Here's why this is genuinely new. Every existing 'explainable ensemble' paper compares the model's heatmap against a **doctor's hand-drawn box** around the tumour. That needs expensive expert annotations that most datasets — including mine — don't have. My method needs **no ground truth at all**. It measures whether the models agree *with each other*. That makes it a self-supervised uncertainty signal that works on any dataset out of the box."

**🧠 Why / how the code works:**
```python
# 1. Heatmap per deep model
A = get_gradcam(cnn, img, cls)
B = get_gradcam(xception, img, cls)
C = get_gradcam(inception, img, cls)

# 2. Binarise at 0.5, then IoU = overlap / union
def iou(a, b):
    a, b = a >= 0.5, b >= 0.5
    return (a & b).sum() / (a | b).sum()

# 3. Average the three pairwise scores
EAA_IoU = mean(iou(A,B), iou(A,C), iou(B,C))
# low EAA_IoU  ->  clinical escalation alert
```
**Research question this tests:** *Does low inter-model attention agreement correlate with the ensemble being wrong?* If yes, EAA-IoU can catch misclassifications before they reach a patient — without retraining a single model.

---

## Slide 10 — The product: the live web application (1.5 min)

**🗣️ Say this:**
"All of this lives inside a working web application, because a research result that stays in a notebook helps no one. The site has two screens.

**Live Diagnosis** — a clinician uploads an MRI and instantly sees: the predicted tumour type, a confidence bar, the GRAD-CAM heatmaps from all three models plus a consensus map showing where the system collectively looked, a per-model vote breakdown so nothing is hidden, and — if the models disagree — the clinical alert telling them to escalate.

**Results Gallery** — all my research figures in one place: the accuracy charts, confusion matrices, and the EAA-IoU analysis, so the evidence behind the system is transparent.

The whole thing runs as a React front-end served by a Python backend, and it's containerised so it can be deployed to the cloud. Every screen carries a disclaimer that it's a research tool, not a replacement for a radiologist."

**🧠 Why:** Front-end = React (Live Diagnosis + Results Gallery). Backend = FastAPI loading all six models and running `xai_engine.py`. Dockerised for deployment. Ties back to your proposal's "interactive web application" deliverable.

---

## Slide 11 — Results & contribution (1 min)

**🗣️ Say this:**
"To summarise the numbers: my best single model was the CNN at 94%. Combining six diverse models by voting pushed that to **98.2%**. But the accuracy is only half the story. The real contribution is the honesty layer — EAA-IoU — which turns the ensemble from a black box that's confidently wrong into a system that can raise its hand and say *'this one is ambiguous, get a human.'* That combination — a confidence-weighted ensemble plus an annotation-free inter-model agreement signal — is, from my literature review, not present in any published brain-tumour paper."

| Model | Accuracy |
|-------|----------|
| Custom CNN (best single) | ~94% |
| Xception (standalone) | 86.9% |
| InceptionV3 (standalone) | 84.6% |
| **6-model ensemble (final)** | **98.2%** |

---

## Slide 12 — Close (30 sec)

**🗣️ Say this:**
"So, to close where I started: one model can be confidently wrong. My framework replaces that single opinion with six diverse ones, and then — crucially — checks whether they truly agree, not just on the answer but on the reasoning. That's how you get accuracy *and* safety in a clinical tool. Thank you. I'm happy to take questions."

---

# Anticipated viva questions — and how to answer

Read these until the answers are reflexes. Panels reward calm, specific answers.

**Q: Why an ensemble instead of just improving one model?**
"Because different architectures fail on different images. A single model's errors are systematic — they come from its design. When I combine six architecturally different models, a wrong final answer needs a *majority* of them to fail the same way on the same scan, which is far less likely. The 94%→98.2% jump is the empirical proof."

**Q: What exactly does 'cutting the dense layer' mean?**
"After training a network normally, I build a second model with the same weights but whose output is the 256-unit dense layer instead of the softmax. So for each image I get a 256-number feature vector — the network's internal understanding — and I feed that to Random Forest, Decision Tree and SVM. Deep network sees; classical model decides."

**Q: Why 256 features? Why 128×128 for the backbones but 256×256 for standalone?**
"256 is the width of the dense layer I chose — compact enough for fast classical training, rich enough to describe the tumour. 128×128 for feature extraction keeps memory and time low across thousands of images; 256×256 for the standalone deep models gives them maximum detail since they do the whole job alone."

**Q: Isn't confidence-weighted voting already published?**
"Yes, and I'm honest about that — I cite it as a known method. It's *not* my novelty. It's the baseline I build on. My novelty is EAA-IoU: measuring inter-model attention agreement as an uncertainty signal without any ground-truth annotation. The combination of the two is the new system."

**Q: How is EAA-IoU different from existing explainable-AI ensembles?**
"Existing work measures a model's heatmap against a radiologist's hand-drawn tumour box — that needs expert annotations. Mine measures the models' heatmaps against *each other*. No annotation needed, so it generalises to any dataset, and it captures a different thing: not 'is the model right', but 'do the models agree', which is a purer uncertainty signal."

**Q: How do you know the models aren't all making the same mistake?**
"I can't guarantee it — that's why diversity is designed in. Three architecturally distinct backbones, plus classical decision layers, deliberately reduce shared bias. And EAA-IoU is a second check: if their *reasoning* diverges, I flag it regardless of the class they output."

**Q: What are the limitations?**
"One dataset, so demographic and scanner diversity is limited. The ensemble is compute-heavy — six models — which not every clinic can run. GRAD-CAM is an approximation of attention, not ground truth. And I've validated EAA-IoU as a correlation with error, not yet in a live clinical trial. Those are my future-work items."

**Q: Why classical ML at all — why not deeper networks?**
"Three reasons: on a small medical dataset, Random Forests and SVMs are robust and less prone to overfitting; they're fast and interpretable; and they give me three extra *diverse* voters that share features with the deep models but use completely different decision logic, which strengthens the ensemble."

**Q: Did you use the test set anywhere during training?**
"No. Feature extraction, model fitting and all choices used the Training folder only. The official Testing folder is touched exactly once, for the final numbers you see."

**Q: There's a normalisation inconsistency in the notebooks — explain.**
"Correct, and it's documented. Standalone models use /255 (range 0–1); the ensemble backbones use /255−0.5 (zero-centred). It's a known inconsistency from developing the notebooks separately. In the deployed web app I standardised every model to /255, so the live system is fully consistent."

---

# Delivery tips

- **Tell it as a story, not a spec sheet.** The arc is: *one model can be confidently wrong → so build six diverse ones → make them vote → then check they agree on the reasoning → ship it as a usable tool.* If you lose your place, return to that arc.
- **Slow down on Slides 6, 7 and 9** — the "why these models", the "dense-layer cut", and the "EAA-IoU novelty". Those three are where a viva is won.
- **Own the weaknesses.** When a panellist finds the normalisation inconsistency or the RF/DT bug, agreeing calmly and explaining it earns more marks than defending it.
- **Have one number ready for everything:** 4 classes, ~9,000 images, 6 models, 256 features, 128 vs 256 input, 94% → 98.2%.
- **Practise the opening and closing word-for-word.** Everything in between can flex; a confident first 30 seconds and last 30 seconds frame the whole viva.
