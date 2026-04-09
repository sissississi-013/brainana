# brainana — Hackathon Research Document

> Paradigm Automated Research Hackathon, April 9, 2026
> All scouting, references, prior art, tool inventory, and project concepts collected here.

---

## Table of Contents

1. [Hackathon Context](#hackathon-context)
2. [The Strategic Gap](#the-strategic-gap)
3. [Fresh Papers and Key Developments (2025-2026)](#fresh-papers-and-key-developments)
4. [Tools and APIs for 8-Hour Build](#tools-and-apis)
5. [Novel Project Concepts](#novel-project-concepts)
6. [MAIA Deep Dive and Brain Encoding Adaptation](#maia-deep-dive)
7. [Benchmarks and Objective Functions](#benchmarks-and-objective-functions)
8. [Surrogate Models and Fast Proxies](#surrogate-models-and-fast-proxies)
9. [EEG Foundation Models Landscape](#eeg-foundation-models)
10. [Prior Art: Brain-as-Reward Optimization](#brain-as-reward-prior-art)
11. [Key Repos and Code Links](#key-repos)
12. [Inspirational Reading List](#inspirational-reading)
13. [Project Direction Evolution](#project-direction-evolution)

---

## Hackathon Context

**Event**: Paradigm Automated Research Hackathon
**Date**: April 9, 2026, 8 AM – 5:30 PM PT
**Host**: Dan Robinson (Paradigm)
**Prizes**: $9k total
- Challenge Track: $1k per leaderboard (3 challenges × in-person + online = $6k)
- Project Track: $1k each for top 3 projects in-person ($3k)

**Project Track Rules**:
- "Use automated research to investigate a question"
- Submit a GitHub link
- Present in person at 4 PM
- Open-ended: challenges, harnesses, tooling, agents, evals

**What Judges Care About** (Dan Robinson's framing):
1. Advances the frontier of automated research
2. Actually works (demo > slides)
3. Novel — not just another chatbot or dashboard
4. Would be useful to autoresearch practitioners
5. Connects to the broader theme (Optimization Arena, Parameter Golf, autoresearch)

**Key Quote from Dan Robinson** (Apr 3):
> "Automated research is one of the most important frontiers of AI. It's increasingly accessible to anyone, thanks to the power of Claude and Codex and tools like autoresearch from @karpathy."

**benedict (@bqbrady)**: Giving a talk about "building the best harness for hill climbing and other things we learned running Optimization Arena."

---

## The Strategic Gap

**The single most important finding: nobody has yet connected the autoresearch loop to brain encoding optimization — and all the pieces now exist to do it today.**

AlphaEvolve/FunSearch-style evolutionary code search, Karpathy's autoresearch commit-or-revert loop, Brain-Score as a programmatic fitness function, TRIBE v2 as a fast in-silico brain simulator, and BERG for instant neural response prediction are all available, open-source, and Colab-compatible. Combining them into a self-improving brain encoding research agent would be genuinely novel. No published work as of April 2026 attempts this synthesis.

---

## Fresh Papers and Key Developments

### The Autoresearch Paradigm

**Karpathy's autoresearch** (March 7, 2026)
- LLM agent iteratively edits training code, runs 5-min experiments, evaluates val_bpb, commits or reverts
- 700 experiments in 48 hours, 20 improvements, 11% gain, zero human intervention
- 66,000+ GitHub stars in one month
- Pattern is domain-agnostic: anything with a measurable metric can be optimized
- Architecture: `program.md` (instructions) + `train.py` (mutable) + `prepare.py` (fixed evaluator)
- GitHub: https://github.com/karpathy/autoresearch

**AlphaEvolve** (Google DeepMind, May 2025, arXiv:2506.13131)
- Evolves codebases using Gemini LLMs + evolutionary search
- Improved on Strassen's 56-year-old matrix multiplication algorithm
- Not applied to neuroscience yet
- Results repo: https://github.com/google-deepmind/alphaevolve_results

**OpenEvolve** — open-source AlphaEvolve
- 5,800+ stars, Apache 2.0
- Supports Python, Rust, R, Metal shaders
- pip installable, CLI or library
- GitHub: https://github.com/algorithmicsuperintelligence/openevolve

**DeepEvolve** (arXiv:2510.06056, October 2025)
- Augmented AlphaEvolve with web-based literature search ("deep research")
- Applied to molecular property prediction and PDEs
- Demonstrates paradigm extends to science domains

**AutoRA** (Brown University, JOSS December 2024)
- Closest existing system to automated neuroscience research loop
- Automates model discovery, experimental design, data collection
- "Theorist" agent + "Experimentalist" agent
- GitHub: https://github.com/AutoResearch/autora

### LLMs as Brain Encoding Features

**LLM embeddings characterize fMRI brain activity** (Doerig, Kietzmann et al., Nature Machine Intelligence, 2025)
- LLM embeddings of scene captions predict fMRI voxel activity
- Linear encoding models reconstruct accurate scene captions from brain responses

**Whisper model alignments** (Google Research, Nature Neuroscience)
- Whisper embeddings align with neural activity during conversation
- Brain's language areas perform next-word prediction mirroring LLM architecture

**LLM-based neurological signal interpretation** (PMC survey, 2025)
- LLaMA2 for fMRI encoding
- MindFormer for EEG-to-image generation
- Unified BCI frameworks using foundation models

### Digital Twins of the Brain

**Stanford Foundation Model of Neural Activity** (Wang, Fahey, Tolias et al., Nature, April 2025)
- AI "digital twin" of mouse visual cortex
- 900+ minutes of brain activity from 8 mice watching movies
- Predicts tens of thousands of neurons to unseen stimuli
- Discovered new connectivity rule: neurons prefer connections based on shared stimulus preferences
- URL: https://www.enigmaproject.ai/

**Real-time ECoG brain simulator** (npj Digital Medicine, 2025)
- Variational Bayesian RNNs for real-time simulation

**Digital Twin Brain** (2026)
- Hypernetwork translates individual connectomes into multitask behavioral predictions
- 228 individuals, >90% accuracy on behavioral choices, r>0.85 on reaction times
- Enables in silico interventions that selectively modulate cognitive/emotional functions

**EBRAINS Virtual Brain Twins**
- Personalized brain models integrating structural MRI, diffusion MRI, EEG/MEG/fMRI

### AI Scientists That Do Neuroscience

**Kosmos** (FutureHouse/Edison, November 2025, arXiv:2511.02824)
- 12-hour autonomous research campaigns, ~42,000 lines of code, ~1,500 papers per run
- Neuroscience campaign: analyzed neuron morphology from 8 connectome reconstructions across 5 species
- Independently discovered lognormal distributions in morphological metrics
- 79.4% statement accuracy, equivalent to ~6 months of PhD-level work

**BrainGPT** (Luo et al., Nature Human Behaviour, November 2024)
- LLM fine-tuned on 20 years of neuroscience literature
- 86% accuracy at predicting neuroscience experimental results vs. 63.4% for 171 human experts
- Could serve as hypothesis generator in automated research loop

**The AI Scientist v2** (Sakana AI, April 2025)
- AI-generated papers passed human peer review at ICLR 2025
- Progressive agentic tree-search methodology

**PaperOrchestra** (Google Cloud AI Research, April 2026)
- Multi-agent framework for automated research paper writing
- 5 specialized agents, ~6,070 LLM calls per paper
- Converts unstructured pre-writing into submission-ready LaTeX

**AutoResearchClaw** (aiming-lab)
- Fully autonomous: ideas → complete papers
- Human-in-the-loop co-pilot modes, anti-hallucination verification
- 10,791+ stars
- GitHub: https://github.com/aiming-lab/AutoResearchClaw

### Self-Improving BCI Systems

**Self-evolving invasive BCI** (Chinese Academy of Sciences, December 2025)
- Quadriplegic patient controls wheelchair and robotic dog via thought
- Online recalibration: real-time parameter adjustment without stopping
- "Neural manifold alignment" for stable decoding despite environmental changes

### Brain Response Optimization (Brain-as-Reward)

**NeuroVolve** (arXiv:2512.00557, November 2025)
- Evolving visual stimuli toward programmable neural objectives
- Optimizes in VLM embedding space to activate/deactivate brain regions
- Recovers known category selectivity, captures subject-specific preferences
- Authors: Haomiao Chen, Keith W Jamison, Mert R. Sabuncu, Amy Kuceyeski

**MindPilot** (ICLR 2026, ncclab-sustech, arXiv:2602.10552)
- Closed-loop visual stimulation optimization with EEG-guided diffusion
- First framework using EEG as optimization feedback for image generation
- Treats brain as black-box function, pseudo-model guidance mechanism
- Validated in simulation AND human experiments
- Applications: semantic target retrieval, EEG feature optimization, emotion regulation
- GitHub: https://github.com/ncclab-sustech/MindPilot

**XDream** (PLOS Computational Biology, 2020)
- Finding preferred stimuli for visual neurons using GANs + gradient-free optimization
- Closed-loop: shows images → receives neural response scores → optimizes
- Classic work in the space
- GitHub: https://github.com/willwx/XDream

**Voxel-weighted Activation Maximization** (arXiv:2506.04379)
- DNN encoding models generate images optimized for predicted cortical voxel responses
- Successfully drives activity in targeted visual regions

**ReAlnet** (Communications Biology, 2026)
- Vision model aligned with human EEG representations
- ~3% average similarity improvement, up to 40% relative improvement over standard CV models

**In Silico Mapping of Visual Categorical Selectivity** (OpenReview)
- Transformer encoder-decoder predicts whole-brain activity
- Diffusion models synthesize images that maximally activate specific brain parcels

### Neuroplasticity and Cognitive Training

**CogSimulator** (arXiv:2412.14188)
- Simulates user cognition with minimal data for tailored cognitive enhancement
- Few-shot predictions in new game scenarios

**DecNefLab** (arXiv:2511.14555)
- Simulation framework for decoded neurofeedback
- Models how different protocols influence neuroplasticity

**NeuroWeaver** (arXiv:2602.13473, February 2026)
- Autonomous evolutionary agent for EEG analysis pipeline optimization
- Domain-informed subspace initialization + multi-objective evolutionary optimization
- Outperforms task-specific methods across 5 benchmarks with fewer parameters

---

## Tools and APIs

### Brain-Score — Programmatic Fitness Function

- 100+ benchmarks scoring DNNs on similarity to primate visual processing (V1, V2, V4, IT)
- Public benchmarks run locally via Python API
- Each evaluation: ~1-5 minutes after data download (~9.8GB)
- ~50 evaluations feasible in 8 hours

```python
pip install git+https://github.com/brain-score/vision
import brainscore_vision
score = brainscore_vision.score(
    model_identifier='alexnet',
    benchmark_identifier='MajajHong2015public.IT-pls'
)
```

- URL: https://brain-score.org

### TRIBE v2 — In-Silico Brain Simulator

- Meta FAIR, released March 26, 2026
- Tri-modal: LLaMA 3.2 (text) + V-JEPA2 (video) + Wav2Vec-BERT (audio)
- 20,484 cortical vertices on fsaverage5 — 70× higher resolution than prior SOTA
- Won 1st place among 263 teams in Algonauts 2025
- Zero-shot to unseen subjects and languages
- CC BY-NC 4.0 license
- Caveat: Requires gated LLaMA 3.2 access; may need A100 80GB

```python
from tribev2 import TribeModel
model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")
events = model.get_events_dataframe(text_path="./story.txt")
preds, segments = model.predict(events=events)
# preds.shape = (n_timesteps, ~20k_vertices)
```

- GitHub: https://github.com/facebookresearch/tribev2
- HuggingFace: https://huggingface.co/facebook/tribev2
- Colab demo: tribe_demo.ipynb in repo
- Interactive demo: https://aidemos.atmeta.com/tribev2

### BERG — Instant In-Silico Neural Responses

- Pretrained encoding models for fMRI, EEG, MEG
- ROI-level predictions (V1, V2, V3, V4, EBA, FFA, PPA)
- Trained on Natural Scenes Dataset
- Lightweight, runs on Colab

```python
berg = BERG(path)
model = berg.get_encoding_model("fmri-nsd-fwrf", subject=1, selection={"roi": "V1"})
responses = berg.encode(model, stimuli)  # shape: (batch, n_voxels)
```

- GitHub: https://github.com/gifale95/BERG
- URL: https://www.alegifford.com/projects/berg/

### Net2Brain — 600+ DNNs vs Brain Data

- All-in-one toolbox, published Frontiers in Neuroinformatics May 2025
- 600+ pretrained models, built-in datasets (NSD, BOLDMoments, Algonauts 2019)
- RSA, weighted RSA, linear encoding, variance partitioning, CKA
- Designed for Colab with tutorial notebooks
- Setup: ~10-15 minutes

```python
pip install net2brain
from net2brain.feature_extraction import FeatureExtractor
fx = FeatureExtractor(model='AlexNet', netset='standard', device='cuda')
fx.extract(data_path='images/', save_path='features/')
```

- GitHub: https://github.com/cvai-roig-lab/Net2Brain

### Microsoft Automated Brain Explanations

- QA encoding models + Generative Causal Testing
- 35 yes-no questions outperform black-box encoding models on fMRI + ECoG
- LLM-generated stimuli for causal hypothesis testing
- MIT License

- GitHub: https://github.com/microsoft/automated-explanations
- URL: https://microsoft.github.io/automated-brain-explanations/

### Brain Visualization

**Nilearn** (Python)
- Publication-grade brain maps, 5 min install
- `view_img_on_surf()` and `view_img()` produce interactive HTML
- `plot_surf_stat_map` on fsaverage5 for TRIBE v2 output
- Destrieux atlas for ROI parcellation on fsaverage5

**NiiVue** (Web)
- WebGL 2.0 neuroimaging viewer, 30+ formats
- niivue-react component for React apps
- ipyniivue for Jupyter
- URL: https://niivue.github.io

### LLM Agent Frameworks

- **AIDE** (github.com/WecoAI/aideml) — ML engineering agent, tree-search in code space, SOTA on MLE-Bench
- **LangGraph** (LangChain) — best for custom observe→hypothesize→experiment→evaluate loop
- **OpenAI Agents SDK** (March 2025) — simplest if using OpenAI models

### braindecode — EEG Deep Learning

- Pre-built EEG models: EEGNet, ShallowFBCSPNet, Deep4Net, EEGConformer, LaBraM, BIOT
- PyTorch-based
- pip install braindecode

### MOABB — EEG Benchmarking

- Mother of All BCI Benchmarks
- Motor imagery, SSVEP, P300 paradigms
- Standardized evaluation, automatic data download
- pip install moabb

### Colab Pro Setup

- A100 40GB: ~$0.10-0.20/hr via 100 monthly compute units
- High-RAM mode: ~52GB system RAM
- A100 80GB available with Pro+
- Mount Google Drive immediately for persistent storage

---

## Novel Project Concepts

### Concept 1: "BrainEvolve" — Evolutionary Search for Brain-Optimal Architectures

LLM agent proposes DNN architecture modifications → evaluate brain-likeness via Brain-Score → commit or revert. Population of model configs (layers, widths, training objectives). Tiered evaluation: RSA/CKA fast screen (seconds) → Brain-Score ridge regression validation (minutes). Start from Net2Brain's 600+ models.

**Demo**: Pareto frontier of ImageNet accuracy vs brain-score evolving in real-time + Nilearn brain maps.

### Concept 2: "BrainMAIA" — Automated Interpretability for Brain Encoding

Adapt MAIA's hypothesis→experiment→update loop. Agent discovers what features brain ROIs have learned. Uses BERG predictions + synthetic image generation + causal intervention.

**Demo**: Agent's evolving understanding shown as annotated brain surface maps.

### Concept 3: "NeuroSurrogate" — Fast Proxy for Brain-Score

Train LightGBM surrogate predicting Brain-Score from architecture descriptors. Net2Brain evaluates 100+ DNNs → extract features → train surrogate. "Brain-Bench" doesn't exist yet — publishable contribution.

**Demo**: Surrogate accuracy + full evolutionary loop with millisecond inner evaluations.

### Concept 4: "TRIBE Explorer" — Autonomous Neuroscience Discovery

TRIBE v2 as evaluation oracle in MAIA-style loop. Agent proposes hypotheses, generates stimuli, runs TRIBE v2, analyzes. Accumulates validated neuroscience findings.

**Demo**: AI doing neuroscience in real-time. Each finding = annotated brain surface map.

### Concept 5: "BrainGPT-Evolve" — LLM-Guided Hypothesis Evolution

BrainGPT-style knowledge + evolutionary search over encoding configs. Hypotheses translated into model configs, evaluated on NSD. Output: better models + validated scientific hypotheses.

### Concept 6: "Brain-Reward Autoresearch" (Our Direction)

Karpathy autoresearch pattern where brain response IS the reward. Agent generates stimuli → TRIBE v2 predicts brain response → ROI extraction → agent interprets → loops. Produces research reports with brain maps.

**Differentiators vs prior art**:

| System | Brain Model | Optimization | Agent Reasoning? | Modality |
|---|---|---|---|---|
| XDream (2020) | Real neurons (invasive) | Genetic algorithm | No | Images |
| NeuroVolve (2025) | fMRI encoding model | Gradient in VLM space | No | Images |
| MindPilot (ICLR 2026) | Real EEG (non-invasive) | Black-box GP surrogate | No | Images |
| **brainana (ours)** | **TRIBE v2 (in silico)** | **LLM autoresearch agent** | **YES** | **Text/Audio/Video** |

### Recommended Hybrid: Concepts 1 + 4

Evolutionary architecture search + TRIBE v2/BERG as fast oracle + Nilearn brain viz:
1. Hour 1-2: Setup Colab, install everything
2. Hour 3-4: Build autoresearch loop
3. Hour 5-6: Run 30-50 iterations
4. Hour 7: Generate brain maps, build dashboard
5. Hour 8: Polish and present

---

## MAIA Deep Dive

### Architecture

Three components:
1. **VLM backbone** (GPT-4V / Claude 3.5 Sonnet / Gemma-3-27B)
2. **System Class**: instruments target neural network, makes subcomponents callable
3. **Tools Class**: Python functions composed into experiments
   - `dataset_exemplars`: finds maximally activating images
   - `text2image`: diffusion model for synthetic test images
   - `edit_images`: causal interventions (add/remove features)

### Agent Loop

1. Receive query
2. Generate initial hypotheses
3. Design experiment (compose tools into Python programs)
4. Execute experiment
5. Observe results (VLM interprets images + activation values)
6. Update hypotheses
7. Iterate until confident
8. Produce final report

Chain-of-thought reasoning between actions. Each step's output feeds next step's planning.

### Key Results

- Neuron descriptions match expert human experimenters (ResNet, CLIP, DINO)
- On Spawrious: identified and removed spurious features, improving robustness approaching fine-tuning on balanced data
- MAIA 2.0 (June 2025): free-form code execution
- OpenMAIA (October 2025): fully open-source models achieve comparable performance

### Adaptation for Brain Encoding ("BrainMAIA")

Replace "artificial neurons" with "brain voxels/ROIs":
1. Query BERG/TRIBE v2 for maximally activating stimuli for target brain region
2. VLM examines stimuli, hypothesizes shared features
3. Generate test images isolating hypothesized feature
4. Run BERG/TRIBE v2 predictions on test images
5. Analyze whether hypothesis predicts activation
6. Iterate and refine

Combine with Microsoft QA encoding: MAIA-style for discovery, QA encoding for rigorous validation.

- Code: https://github.com/multimodal-interpretability/maia

---

## Benchmarks and Objective Functions

### Tiered Evaluation Strategy

**Tier 1 — Ultra-fast (<1 second)**: Linear CKA or RSA between model features and pre-computed brain RDMs. Eliminates bad candidates. Just matrix operations.

**Tier 2 — Fast proxy (<1 minute)**: Ridge regression with fixed λ on small NSD subset (500 images, 1 subject). Or compare to BERG-predicted responses via cosine similarity.

**Tier 3 — Full validation (minutes to hours)**: Noise-ceiling-normalized Pearson R² on full NSD test set, multi-subject, multi-ROI, cross-validated ridge regression. Top candidates only.

### Key Benchmarks

- **Brain-Score**: 100+ benchmarks, V1/V2/V4/IT, scores 0-1 noise-ceiling normalized
- **Algonauts 2025**: CNeuroMod dataset, ~80 hours fMRI, TRIBE won with OOD Pearson ~0.2146
- **Natural Scenes Dataset (NSD)**: 8 subjects, 7T fMRI, 9,000-10,000 scenes/subject, gold standard
- **THINGS**: Aligned fMRI + MEG + EEG for 1,854 objects with 26,107 images
- **EEG-FM-Bench**: 14 datasets, 10 paradigms, benchmarks BENDR/BIOT/CBraMod/EEGPT/LaBraM
- **MOABB**: Motor imagery, SSVEP, P300

### Metrics

- Pearson correlation: fastest, standard for Algonauts/NSD
- Linear CKA: best for quick model ranking, single matrix operation
- Brain-Score PLS regression: with noise-ceiling normalization
- RSA with Spearman: robust but less sensitive

---

## Surrogate Models and Fast Proxies

### What Exists

- BERG: pretrained encoding models, forward pass = instant neural responses
- TRIBE v2: most accurate in-silico brain simulator
- Net2Brain: fast RSA against 600+ models

### The Gap: No "Brain-Bench" Surrogate Exists

NAS-Bench-301 showed surrogate models (LightGBM/XGBoost) trained on ~60,000 evaluations can predict performance in 10^18 search spaces. YOLO-NAS-Bench (March 2026) achieves R²=0.815.

Recipe for brain-encoding surrogate:
1. Evaluate 100-500 model configs on Brain-Score/NSD via Net2Brain
2. Extract architecture descriptors
3. Train LightGBM surrogate
4. Use for millisecond screening in evolutionary loop

**This surrogate itself would be a publishable contribution.**

### Fast Classical Methods as Cheap Proxies

- Ridge regression with fixed λ (skip CV): seconds
- RSA: just compute RDMs and correlate upper triangles
- Linear CKA: single matrix operation
- THINGS EEG data as fast proxy for slower fMRI evaluation

---

## EEG Foundation Models Landscape (April 2026)

| Model | Key Innovation | Scale | Venue |
|---|---|---|---|
| **CBraMod** | Criss-cross spatial-temporal attention | Leads 10 downstream tasks | ICLR 2025 |
| **NeuroLM** | First LLM-based multi-task EEG | 1.7B params, 25,000 hours | ICLR 2025 |
| **BrainGPT/EEGPT** | Autoregressive pretraining | 1.1B params (largest EEG model) | — |
| **LaBraM** | Cross-dataset, VQ neural spectrum prediction | 2,500 hours, 20 datasets | ICLR 2024 Spotlight |
| **LaBraM++** | Improved signal processing foundations | Enhanced from LaBraM | May 2025 |
| **BIOT** | Handles messy cross-dataset biosignals | Multi-dataset | NeurIPS 2023 |
| **BrainOmni** | First unified EEG+MEG FM | Multimodal | NeurIPS 2025 |
| **FEMBA** | Bidirectional state-space (not Transformer) | 7.8M params, linear scaling | 2025 |

All open-source with pretrained weights. LaBraM, BIOT, CBraMod integrated into braindecode.

Key trends: >25,000 hours pre-training data, >15,000 subjects, models reaching 1B+ params, no scaling plateau observed.

---

## Key Repos and Code Links

### Must-Clone for Hackathon

| Repo | Purpose | Stars |
|---|---|---|
| [facebookresearch/tribev2](https://github.com/facebookresearch/tribev2) | Brain simulator | 1,688 |
| [gifale95/BERG](https://github.com/gifale95/BERG) | Fast brain encoding | — |
| [cvai-roig-lab/Net2Brain](https://github.com/cvai-roig-lab/Net2Brain) | 600+ DNN brain comparison | — |
| [brain-score/vision](https://github.com/brain-score/vision) | Brain-Score benchmarks | — |

### Reference Repos

| Repo | Purpose | Stars |
|---|---|---|
| [karpathy/autoresearch](https://github.com/karpathy/autoresearch) | The autoresearch pattern | 69,319 |
| [multimodal-interpretability/maia](https://github.com/multimodal-interpretability/maia) | MAIA agent | 108 |
| [microsoft/automated-explanations](https://github.com/microsoft/automated-explanations) | Brain hypothesis testing | — |
| [ncclab-sustech/MindPilot](https://github.com/ncclab-sustech/MindPilot) | EEG-guided stimulus optimization | 9 |
| [willwx/XDream](https://github.com/willwx/XDream) | Classic closed-loop neuro | — |
| [algorithmicsuperintelligence/openevolve](https://github.com/algorithmicsuperintelligence/openevolve) | Open-source AlphaEvolve | 5,920 |
| [WecoAI/aideml](https://github.com/WecoAI/aideml) | ML agent framework | — |
| [AutoResearch/autora](https://github.com/AutoResearch/autora) | Automated neuroscience research | — |
| [935963004/LaBraM](https://github.com/935963004/labram) | EEG foundation model | — |
| [xw1216/EEG-FM-Bench](https://github.com/xw1216/EEG-FM-Bench) | EEG FM benchmarking | — |
| [NeuroTechX/moabb](https://github.com/NeuroTechX/moabb) | EEG benchmarks | — |
| [braindecode/braindecode](https://github.com/braindecode/braindecode) | EEG deep learning | — |
| [aiming-lab/AutoResearchClaw](https://github.com/aiming-lab/AutoResearchClaw) | Autonomous research system | 10,791 |
| [handsome-rich/Awesome-Auto-Research-Tools](https://github.com/handsome-rich/Awesome-Auto-Research-Tools) | Curated list | — |

---

## Inspirational Reading List

### AI & Research

- [Accelerating Scientific Breakthroughs with an AI Co-Scientist](https://research.google/blog/accelerating-scientific-breakthroughs-with-an-ai-co-scientist/) — Google Research
- [Improving the Academic Workflow: AI Agents for Figures and Peer Review](https://research.google/blog/improving-the-academic-workflow-introducing-two-ai-agents-for-better-figures-and-peer-review/) — Google Research
- [PaperVizAgent](https://arxiv.org/pdf/2601.23265) — Visual agent for researchers
- [ScholarPeer](https://arxiv.org/pdf/2601.22638) — Automated peer review
- [TurboQuant](https://arxiv.org/abs/2504.19874) — Online vector quantization with near-optimal distortion rate (EXCEPTIONAL WORK)

### Attitude & Philosophy

- [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) — Rich Sutton
- [You and Your Research](https://www.cs.virginia.edu/~robins/YouAndYourResearch.html) — Richard Hamming

### Neuroscience + AI

- [neuroaisafety.com](https://neuroaisafety.com)
- [Towards Magnanimous AGI](https://blog.amaranth.foundation/p/towards-magnanimous-agi)
- [Substrate: Information to Atoms](https://substrate.com/information-to-atoms)
- [PAIR with Google](https://pair.withgoogle.com/)
- [AlphaEvolve Blog](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)
- [AlphaEvolve Results Colab](https://colab.research.google.com/github/google-deepmind/alphaevolve_results/blob/master/mathematical_results.ipynb)
- [arxiv:2603.29585v2](https://arxiv.org/html/2603.29585v2)

### Key Papers

- MindPilot: arXiv:2602.10552 (ICLR 2026)
- NeuroVolve: arXiv:2512.00557
- NeuroWeaver: arXiv:2602.13473
- MAIA: arXiv:2404.14394 (ICML 2024)
- TRIBE v2: Meta AI (March 2026)
- Kosmos: arXiv:2511.02824
- AI Scientist (Nature, March 2026)
- BrainGPT (Nature Human Behaviour, November 2024)
- Stanford Digital Twins (Nature, April 2025)

---

## Project Direction Evolution

### Round 1: TRIBE Explorer (Pure Neuroscience Discovery)

Autoresearch agent uses TRIBE v2 to autonomously investigate neuroscience questions. Generate hypotheses → design experiments → simulate brain response → analyze → iterate.

**Pushback**: Cool demo but doesn't directly help the team's EEG foundation model research.

### Round 2: EEG Autoresearch Harness

Autoresearch loop that optimizes EEG foundation model development. Agent modifies architecture/preprocessing/hyperparameters, evaluates on MOABB benchmarks.

**Pushback**: Useful but less novel. More like generic AutoML than a new paradigm.

### Round 3: Brain-Reward Autoresearch (Current Direction)

The key insight: brain response IS the reward function. Instead of optimizing train.py against val_bpb, optimize stimuli/models against brain activation.

Two modes:
1. **Stimulus optimization**: Generate stimuli that maximize target brain region activation
2. **Architecture evolution**: Evolve model architectures toward brain-likeness

This is genuinely novel (no published work combines autoresearch + brain-as-reward) and connects to:
- Team's EEG work (swap TRIBE v2 for EEG model)
- Hackathon theme (autoresearch pattern)
- Broader impact (brain-score, BCI, cognitive training)

### Recommended Final Build: Hybrid BrainEvolve + TRIBE Explorer

1. **Autoresearch loop** with LLM agent
2. **BERG / Brain-Score** as fast brain-likeness evaluation (Tier 1-2)
3. **TRIBE v2** for full in-silico brain simulation (Tier 3 / demo)
4. **Nilearn** brain maps for visualization
5. **Gradio/Streamlit** dashboard for live demo

The agent proposes modifications → evaluates brain alignment → commits or reverts → produces brain maps showing improvement → generates research report.

---

*Last updated: April 9, 2026, during Paradigm Automated Research Hackathon*
