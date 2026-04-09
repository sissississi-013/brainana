# brainana — Learning Log

> Progress, decisions, blockers, and discoveries tracked here throughout the hackathon.

---

## Hour 1: Setup

**Started**: April 9, 2026

### Goal
- Clone tribev2, verify text->brain pipeline on Colab
- Set up project structure and Python environment
- Test ROI extraction with Destrieux atlas on fsaverage5
- HARD DEADLINE: pivot to BERG if TRIBE v2 not working by minute 45

### Project Structure Created
```
brainana/
  doc/
    research.md    # Full research scouting document
    learning.md    # This file -- progress log
  brainana/        # Python package
  notebooks/       # Colab demo notebooks
  outputs/         # Generated reports and brain maps
```

### Steps
- [ ] Clone facebookresearch/tribev2
- [ ] Create Python virtualenv
- [ ] Install tribev2 + nilearn + anthropic
- [ ] Test TRIBE v2 text->brain pipeline
- [ ] Test Destrieux atlas ROI extraction
- [ ] If TRIBE v2 fails: clone BERG as fallback

### Decisions Made
- **Brain oracle**: TRIBE v2 (primary), BERG (fallback)
- **Stimulus type**: Text (agent generates via Claude) + Images (HuggingFace dataset)
- **Direction**: Brain-reward autoresearch -- optimize stimuli against brain activation
- **ROI atlas**: Destrieux on fsaverage5 (~75 named regions)

---
