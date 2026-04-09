"""Brain simulator interface and implementations.

Provides a pluggable brain simulation backend. The agent talks to a
BrainSimulator -- swap between TRIBE v2, BERG, mock, or your own EEG model.
"""

from __future__ import annotations

import logging
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

N_VERTICES_FSAVERAGE5 = 20484
N_VERTICES_LEFT = 10242
N_VERTICES_RIGHT = 10242


class BrainSimulator(ABC):
    """Protocol for brain simulation backends."""

    @abstractmethod
    def simulate(self, text: str) -> np.ndarray:
        """Predict brain response to a text stimulus.

        Returns
        -------
        np.ndarray of shape (n_vertices,) -- mean activation across time
            on the fsaverage5 cortical mesh.
        """
        ...

    @abstractmethod
    def simulate_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Predict brain responses for multiple stimuli."""
        ...


class MockBrainSimulator(BrainSimulator):
    """Deterministic mock for local testing. Produces plausible-looking
    brain maps where language-related regions respond to text complexity
    and emotional content responds to emotional words.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self._base_map = self.rng.randn(N_VERTICES_FSAVERAGE5) * 0.1

    def _text_features(self, text: str) -> dict:
        words = text.split()
        n_words = len(words)
        avg_word_len = np.mean([len(w) for w in words]) if words else 0
        emotional_words = {
            "love", "fear", "anger", "happy", "sad", "joy", "pain",
            "moral", "dilemma", "kill", "save", "suffer", "death", "life",
            "beautiful", "terrible", "amazing", "horrible", "friend", "enemy",
        }
        emotion_ratio = sum(1 for w in words if w.lower() in emotional_words) / max(len(words), 1)
        social_words = {
            "he", "she", "they", "people", "person", "friend", "family",
            "believe", "think", "feel", "want", "know", "said", "told",
            "relationship", "trust", "betray", "help", "society", "community",
        }
        social_ratio = sum(1 for w in words if w.lower() in social_words) / max(len(words), 1)
        return {
            "complexity": min(n_words / 100, 1.0) * avg_word_len / 6.0,
            "emotion": emotion_ratio,
            "social": social_ratio,
            "length": n_words,
        }

    def simulate(self, text: str) -> np.ndarray:
        features = self._text_features(text)
        activation = self._base_map.copy()

        # Prefrontal cortex (vertices ~0-2000 left, ~10242-12242 right) respond to complexity + social
        pfc_boost = features["complexity"] * 0.5 + features["social"] * 0.8
        activation[0:2000] += pfc_boost + self.rng.randn(2000) * 0.05
        activation[N_VERTICES_LEFT:N_VERTICES_LEFT + 2000] += pfc_boost + self.rng.randn(2000) * 0.05

        # Temporal cortex (~3000-5000 left) responds to language complexity
        lang_boost = features["complexity"] * 0.7 + features["length"] / 200
        activation[3000:5000] += lang_boost + self.rng.randn(2000) * 0.05

        # Amygdala-adjacent regions respond to emotion
        emo_boost = features["emotion"] * 1.0
        activation[5000:5500] += emo_boost + self.rng.randn(500) * 0.05
        activation[N_VERTICES_LEFT + 5000:N_VERTICES_LEFT + 5500] += emo_boost + self.rng.randn(500) * 0.05

        # Add global noise
        activation += self.rng.randn(N_VERTICES_FSAVERAGE5) * 0.02
        return activation

    def simulate_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [self.simulate(t) for t in texts]


class TribeV2Simulator(BrainSimulator):
    """TRIBE v2 brain simulator. Requires GPU and tribev2 package."""

    def __init__(self, cache_folder: str = "./cache", device: str = "auto"):
        from tribev2.demo_utils import TribeModel
        self.model = TribeModel.from_pretrained(
            "facebook/tribev2",
            cache_folder=cache_folder,
            device=device,
        )
        self.cache_folder = cache_folder

    def simulate(self, text: str) -> np.ndarray:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, dir=self.cache_folder
        ) as f:
            f.write(text)
            f.flush()
            text_path = f.name

        try:
            events = self.model.get_events_dataframe(text_path=text_path)
            preds, segments = self.model.predict(events=events, verbose=False)
            return preds.mean(axis=0)
        finally:
            Path(text_path).unlink(missing_ok=True)

    def simulate_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [self.simulate(t) for t in texts]


def get_simulator(backend: str = "mock", **kwargs) -> BrainSimulator:
    """Factory function for brain simulators."""
    if backend == "mock":
        return MockBrainSimulator(**kwargs)
    elif backend == "tribev2":
        return TribeV2Simulator(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'mock' or 'tribev2'.")
