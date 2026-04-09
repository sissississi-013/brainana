# brainana

**Brain-Reward Autoresearch Agent**

An autoresearch agent where brain response IS the reward function. Karpathy's autoresearch pattern meets neuroscience: instead of optimizing `train.py` against `val_bpb`, we optimize **stimuli** against **brain activation**.

Give brainana a neural objective -- "maximize prefrontal cortex activation" -- and it autonomously generates text stimuli, predicts brain response through [TRIBE v2](https://github.com/facebookresearch/tribev2) (Meta's digital brain twin), measures target region activation, reasons about results with neuroscience knowledge, and iterates.

Built at the [Paradigm Automated Research Hackathon](https://paradigm.xyz), April 9, 2026.

## How It Works

```
Research Question → LLM Agent (Claude) → Generate Stimulus
     ↑                                        ↓
  Interpret                            TRIBE v2 (Brain Sim)
     ↑                                        ↓
  Loop ← ← ← ← ← ← ← ← ← ← ROI Extraction (Destrieux Atlas)
     ↓
  Research Report + Brain Maps
```

The agent follows the **autoresearch discipline**:
1. **One hypothesis per iteration** -- generate ONE stimulus to test ONE idea
2. **Evaluate** -- TRIBE v2 predicts fMRI brain response (~20k cortical vertices)
3. **Score** -- extract activation in the target brain region (Destrieux atlas on fsaverage5)
4. **Interpret** -- LLM reasons about WHY this stimulus worked or didn't
5. **Iterate** -- build on findings, try a new direction
6. **Report** -- produce brain maps + research report with scientific findings

## The Autoresearch Analogy

| Karpathy autoresearch | brainana |
|---|---|
| Mutable: `train.py` | Mutable: **stimulus text** |
| Evaluator: `prepare.py` + `val_bpb` | Evaluator: **TRIBE v2** + ROI activation |
| Reward: lower bits-per-byte | Reward: **higher brain region activation** |
| Agent modifies code | Agent generates stimuli + hypotheses |
| Output: better LLM | Output: **brain maps + neuroscience findings** |

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Set your API key
export ANTHROPIC_API_KEY="your-key"

# Run (with mock brain for local testing)
python -m brainana "What activates prefrontal cortex?" prefrontal mock 10

# Run with TRIBE v2 on GPU (Colab)
python -m brainana "What activates prefrontal cortex?" prefrontal tribev2 15
```

## Example Output

In a 10-iteration run targeting the prefrontal cortex, brainana discovered:

> Complex moral dilemmas requiring integration of multiple ethical frameworks
> (utilitarian vs. deontological vs. virtue ethics) with life-or-death stakes
> and perspective-taking of multiple stakeholders produce the strongest
> prefrontal cortex activation -- confirming known neuroscience about
> medial PFC's role in moral reasoning and theory of mind.

The agent generates brain maps showing activation progression and a full research report.

## Why This Is Novel

No prior work combines the autoresearch loop with brain-response-as-reward:

| System | Brain Model | Optimization | Agent Reasoning? |
|---|---|---|---|
| XDream (2020) | Real neurons | Genetic algorithm | No |
| NeuroVolve (2025) | fMRI encoding | Gradient descent | No |
| MindPilot (ICLR 2026) | Real EEG | Black-box GP | No |
| **brainana** | **TRIBE v2 (in silico)** | **LLM autoresearch** | **Yes** |

## Architecture

```
brainana/
  agent.py      # Core autoresearch loop
  brain.py      # Brain simulator (TRIBE v2 / Mock / pluggable)
  roi.py        # Destrieux atlas ROI extraction on fsaverage5
  llm.py        # Claude API wrapper
  prompts.py    # Neuroscience-aware system prompts
  viz.py        # nilearn brain surface visualization
  report.py     # Research report generator
```

## Target Regions

Supported region groups (Destrieux atlas parcellation):
- `prefrontal` -- executive function, decision-making, working memory
- `medial_prefrontal` -- social cognition, theory of mind, moral reasoning
- `temporal` -- language processing, auditory, semantic memory
- `visual` -- visual processing (V1-V4, object recognition)
- `motor` -- movement planning and execution
- `parietal` -- spatial processing, attention, numerical cognition
- `language` -- Broca's + Wernicke's areas, language network

## For Your Own Research

The brain simulator is pluggable -- swap the backend:

```python
from brainana.brain import BrainSimulator

class MyEEGModel(BrainSimulator):
    def simulate(self, text: str) -> np.ndarray:
        # Your EEG foundation model here
        return your_model.predict(text)
```

## References

- [Karpathy autoresearch](https://github.com/karpathy/autoresearch) -- the pattern
- [TRIBE v2](https://github.com/facebookresearch/tribev2) -- Meta's brain simulator
- [MAIA](https://github.com/multimodal-interpretability/maia) -- automated interpretability agent
- [MindPilot](https://github.com/ncclab-sustech/MindPilot) -- EEG-guided stimulus optimization
- [NeuroVolve](https://arxiv.org/abs/2512.00557) -- programmable neural objectives

## License

MIT
