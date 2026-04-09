"""Core autoresearch loop -- the brain-reward agent.

This is the heart of brainana: an LLM agent that generates stimuli,
simulates brain response, scores activation in a target region, interprets
results, and iterates. The Karpathy autoresearch pattern applied to neuroscience.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from brainana.brain import BrainSimulator, get_simulator
from brainana.llm import call_llm_json
from brainana.prompts import INTERPRET_PROMPT, REPORT_SYNTHESIS_PROMPT, SYSTEM_PROMPT
from brainana.report import generate_report
from brainana.roi import get_all_region_group_activations, get_region_group_activation, get_top_regions
from brainana.viz import plot_brain_map, plot_score_progression

logger = logging.getLogger(__name__)
console = Console()


def _build_generation_prompt(
    objective: str,
    target_region: str,
    history: list[dict],
    iteration: int,
) -> str:
    """Build the user prompt for stimulus generation."""
    parts = [
        f"## Research Objective\n{objective}\n",
        f"## Target Brain Region\n{target_region}\n",
        f"## Current Iteration\n{iteration}\n",
    ]

    if history:
        parts.append("## Previous Results (most recent first)\n")
        for h in reversed(history[-5:]):
            best_marker = " [BEST SO FAR]" if h.get("is_best") else ""
            parts.append(
                f"- Iteration {h['iteration']}: score={h['score']:.4f}{best_marker}\n"
                f"  Hypothesis: {h['hypothesis']}\n"
                f"  Stimulus snippet: {h['stimulus'][:100]}...\n"
            )
            if isinstance(h.get("interpretation"), dict):
                parts.append(f"  Insight: {h['interpretation'].get('key_insight', '')}\n")

        best_score = max(h["score"] for h in history)
        parts.append(f"\n## Best score so far: {best_score:.4f}\n")
        parts.append("Generate a NEW stimulus that you predict will BEAT the current best. Explain your reasoning.\n")
    else:
        parts.append(
            "This is the FIRST iteration. Generate an initial exploratory stimulus "
            "to establish a baseline. Choose something you'd expect to moderately "
            "activate the target region.\n"
        )

    parts.append("Respond with valid JSON as specified in your system prompt.")
    return "\n".join(parts)


def _build_interpret_prompt(
    stimulus: str,
    target_region: str,
    score: float,
    all_regions: dict[str, float],
    history: list[dict],
) -> str:
    """Build the user prompt for result interpretation."""
    sorted_regions = sorted(all_regions.items(), key=lambda x: x[1], reverse=True)
    region_str = "\n".join(f"  - {name}: {val:.4f}" for name, val in sorted_regions)

    prev_best = max((h["score"] for h in history), default=0)
    comparison = "NEW BEST" if score > prev_best else f"below best ({prev_best:.4f})"

    return (
        f"## Stimulus\n{stimulus}\n\n"
        f"## Target: {target_region}\n"
        f"## Target activation score: {score:.4f} ({comparison})\n\n"
        f"## All region activations:\n{region_str}\n\n"
        f"## Iteration: {len(history) + 1}\n"
        f"Interpret this result. What does the activation pattern tell us?"
    )


def run_brainana(
    objective: str,
    target_region: str = "prefrontal",
    max_iterations: int = 15,
    backend: str = "mock",
    output_dir: str = "./outputs",
    **backend_kwargs,
) -> str:
    """Run the brain-reward autoresearch loop.

    Parameters
    ----------
    objective : Research question (e.g., "What maximally activates prefrontal cortex?")
    target_region : Region group to target (prefrontal, temporal, visual, motor, etc.)
    max_iterations : Number of iterations to run
    backend : Brain simulator backend ('mock' or 'tribev2')
    output_dir : Where to save outputs
    backend_kwargs : Additional args for the simulator

    Returns
    -------
    str : Path to the generated research report
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(Panel(
        f"[bold]brainana[/bold] — Brain-Reward Autoresearch Agent\n\n"
        f"Objective: {objective}\n"
        f"Target: {target_region}\n"
        f"Backend: {backend}\n"
        f"Max iterations: {max_iterations}",
        title="Starting Research Session",
        border_style="blue",
    ))

    brain = get_simulator(backend, **backend_kwargs)
    history: list[dict] = []
    best_score = -float("inf")

    for i in range(max_iterations):
        console.print(f"\n[bold cyan]--- Iteration {i} ---[/bold cyan]")

        # 1. Agent generates stimulus
        gen_prompt = _build_generation_prompt(objective, target_region, history, i)
        try:
            proposal = call_llm_json(SYSTEM_PROMPT, gen_prompt)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            continue

        stimulus = proposal.get("stimulus_text", "")
        hypothesis = proposal.get("hypothesis", "")
        strategy = proposal.get("strategy", "")

        console.print(f"[green]Hypothesis[/green]: {hypothesis}")
        console.print(f"[dim]Stimulus[/dim]: {stimulus[:120]}...")

        # 2. Simulate brain response
        t0 = time.time()
        brain_map = brain.simulate(stimulus)
        sim_time = time.time() - t0
        console.print(f"[dim]Brain simulation: {sim_time:.1f}s[/dim]")

        # 3. Score target ROI
        score = get_region_group_activation(brain_map, target_region)
        all_regions = get_all_region_group_activations(brain_map)
        top = get_top_regions(brain_map, n=5)

        is_best = score > best_score
        if is_best:
            best_score = score
            console.print(f"[bold red]NEW BEST: {score:.4f}[/bold red]")
        else:
            console.print(f"Score: {score:.4f} (best: {best_score:.4f})")

        # 4. Generate brain map
        brain_map_path = plot_brain_map(
            brain_map,
            title=f"Iteration {i} | {target_region}: {score:.4f}",
            output_path=output_dir / f"brain_iter_{i:03d}.png",
        )

        # 5. Agent interprets results
        interp_prompt = _build_interpret_prompt(stimulus, target_region, score, all_regions, history)
        try:
            interpretation = call_llm_json(INTERPRET_PROMPT, interp_prompt)
        except Exception as e:
            logger.error(f"LLM interpretation failed: {e}")
            interpretation = {"interpretation": str(e), "key_insight": "Error"}

        if isinstance(interpretation, dict):
            console.print(f"[yellow]Insight[/yellow]: {interpretation.get('key_insight', '')}")

        # 6. Log
        history.append({
            "iteration": i,
            "stimulus": stimulus,
            "hypothesis": hypothesis,
            "strategy": strategy,
            "score": score,
            "is_best": is_best,
            "interpretation": interpretation,
            "brain_map_path": brain_map_path,
            "all_regions": all_regions,
            "top_regions": top,
            "sim_time": sim_time,
        })

    # Generate score progression plot
    scores = [h["score"] for h in history]
    plot_score_progression(
        scores,
        title=f"brainana: {target_region} Activation Over Iterations",
        output_path=output_dir / "score_progression.png",
        target_region=target_region,
    )

    # Generate synthesis
    console.print("\n[bold]Generating research synthesis...[/bold]")
    try:
        synth_prompt = (
            f"Objective: {objective}\n"
            f"Target: {target_region}\n"
            f"Iterations: {len(history)}\n"
            f"Best score: {best_score:.4f}\n\n"
            f"History:\n" + json.dumps(
                [{k: v for k, v in h.items() if k not in ("brain_map_path", "all_regions", "top_regions")}
                 for h in history],
                indent=2, default=str,
            )
        )
        synthesis = call_llm_json(REPORT_SYNTHESIS_PROMPT, synth_prompt)
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        synthesis = None

    # Generate report
    report_path = generate_report(
        objective=objective,
        target_region=target_region,
        history=history,
        synthesis=synthesis,
        output_dir=output_dir,
    )

    console.print(Panel(
        f"[bold green]Research session complete![/bold green]\n\n"
        f"Iterations: {len(history)}\n"
        f"Best score: {best_score:.4f}\n"
        f"Report: {report_path}",
        title="Session Complete",
        border_style="green",
    ))

    return report_path


if __name__ == "__main__":
    import sys

    objective = sys.argv[1] if len(sys.argv) > 1 else "What maximally activates the prefrontal cortex?"
    target = sys.argv[2] if len(sys.argv) > 2 else "prefrontal"
    backend = sys.argv[3] if len(sys.argv) > 3 else "mock"
    n_iters = int(sys.argv[4]) if len(sys.argv) > 4 else 10

    run_brainana(
        objective=objective,
        target_region=target,
        max_iterations=n_iters,
        backend=backend,
    )
