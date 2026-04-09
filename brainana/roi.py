"""ROI (Region of Interest) extraction from fsaverage5 brain maps.

Uses the Destrieux atlas to map ~20k vertex predictions to named brain regions.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)

N_VERTICES_LEFT = 10242
N_VERTICES_RIGHT = 10242

REGION_GROUPS = {
    "prefrontal": [
        "G_front_sup", "G_front_middle", "G_front_inf-Orbital",
        "G_front_inf-Triangul", "G_front_inf-Opercular",
        "G_orbital", "G_rectus", "S_front_sup", "S_front_middle",
        "S_front_inf", "S_orbital_lateral", "S_orbital-H_Shaped",
        "G_and_S_cingul-Ant", "G_and_S_cingul-Mid-Ant",
    ],
    "temporal": [
        "G_temp_sup-Lateral", "G_temp_sup-Plan_tempo",
        "G_temp_sup-Plan_polar", "G_temporal_middle", "G_temporal_inf",
        "S_temporal_sup", "S_temporal_transverse", "S_temporal_inf",
    ],
    "visual": [
        "G_cuneus", "G_occipital_sup", "G_occipital_middle",
        "G_oc-temp_lat-fusifor", "S_calcarine", "S_oc_sup_and_transversal",
        "Pole_occipital", "S_occipital_ant", "S_oc_middle_and_Lunatus",
    ],
    "motor": [
        "G_precentral", "G_postcentral", "S_central",
        "G_and_S_paracentral",
    ],
    "parietal": [
        "G_parietal_sup", "G_pariet_inf-Angular",
        "G_pariet_inf-Supramar", "S_intrapariet_and_P_trans",
        "G_precuneus", "S_subparietal",
    ],
    "medial_prefrontal": [
        "G_front_sup", "G_and_S_cingul-Ant", "G_and_S_cingul-Mid-Ant",
        "G_rectus", "G_and_S_subcentral",
    ],
    "language": [
        "G_temp_sup-Lateral", "G_temporal_middle",
        "G_front_inf-Triangul", "G_front_inf-Opercular",
        "S_temporal_sup", "G_and_S_cingul-Mid-Ant",
    ],
}


@lru_cache(maxsize=1)
def _load_destrieux():
    """Load the Destrieux atlas for fsaverage5. Cached after first call."""
    from nilearn.datasets import fetch_atlas_surf_destrieux
    atlas = fetch_atlas_surf_destrieux()
    labels_left = np.array(atlas["map_left"])
    labels_right = np.array(atlas["map_right"])
    label_names = atlas["labels"]
    if isinstance(label_names[0], bytes):
        label_names = [l.decode() for l in label_names]
    return labels_left, labels_right, label_names


def get_roi_activation(
    brain_map: np.ndarray,
    region_name: str,
    hemisphere: str = "both",
) -> float:
    """Extract mean activation for a named Destrieux region.

    Parameters
    ----------
    brain_map : np.ndarray of shape (n_vertices,) where n_vertices = 20484
    region_name : str -- exact Destrieux label name
    hemisphere : 'left', 'right', or 'both'
    """
    labels_left, labels_right, label_names = _load_destrieux()

    matching_idx = [i for i, name in enumerate(label_names) if region_name in name]
    if not matching_idx:
        raise ValueError(f"Region '{region_name}' not found. Available: {label_names}")

    activations = []
    for idx in matching_idx:
        if hemisphere in ("left", "both"):
            mask = labels_left == idx
            if mask.any():
                activations.append(brain_map[:N_VERTICES_LEFT][mask].mean())
        if hemisphere in ("right", "both"):
            mask = labels_right == idx
            if mask.any():
                activations.append(brain_map[N_VERTICES_LEFT:][mask].mean())

    if not activations:
        return 0.0
    return float(np.mean(activations))


def get_region_group_activation(
    brain_map: np.ndarray,
    group_name: str,
) -> float:
    """Extract mean activation for a named region group (e.g., 'prefrontal')."""
    if group_name not in REGION_GROUPS:
        raise ValueError(
            f"Unknown group '{group_name}'. Available: {list(REGION_GROUPS.keys())}"
        )
    scores = []
    for region in REGION_GROUPS[group_name]:
        try:
            scores.append(get_roi_activation(brain_map, region))
        except ValueError:
            continue
    return float(np.mean(scores)) if scores else 0.0


def get_all_region_group_activations(brain_map: np.ndarray) -> dict[str, float]:
    """Get activation for all predefined region groups."""
    return {
        group: get_region_group_activation(brain_map, group)
        for group in REGION_GROUPS
    }


def get_top_regions(brain_map: np.ndarray, n: int = 10) -> list[tuple[str, float]]:
    """Find the top-n most activated Destrieux regions."""
    labels_left, labels_right, label_names = _load_destrieux()
    scores = []
    for i, name in enumerate(label_names):
        if name in ("Unknown", "Medial_wall", "unknown"):
            continue
        vals = []
        mask_l = labels_left == i
        if mask_l.any():
            vals.append(brain_map[:N_VERTICES_LEFT][mask_l].mean())
        mask_r = labels_right == i
        if mask_r.any():
            vals.append(brain_map[N_VERTICES_LEFT:][mask_r].mean())
        if vals:
            scores.append((name, float(np.mean(vals))))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:n]
