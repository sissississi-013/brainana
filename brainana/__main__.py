"""CLI entry point: python -m brainana"""

import argparse
import sys

from brainana.agent import run_brainana


def main():
    parser = argparse.ArgumentParser(
        description="brainana: Brain-Reward Autoresearch Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m brainana "What activates prefrontal cortex?" prefrontal mock 5
  python -m brainana "Discover temporal lobe selectivity" temporal tribev2 15
        """,
    )
    parser.add_argument(
        "objective",
        nargs="?",
        default="What maximally activates the prefrontal cortex?",
        help="Research question",
    )
    parser.add_argument(
        "target_region",
        nargs="?",
        default="prefrontal",
        choices=["prefrontal", "temporal", "visual", "motor", "parietal",
                 "medial_prefrontal", "language"],
        help="Target brain region group",
    )
    parser.add_argument(
        "backend",
        nargs="?",
        default="mock",
        choices=["mock", "tribev2"],
        help="Brain simulator backend",
    )
    parser.add_argument(
        "iterations",
        nargs="?",
        type=int,
        default=10,
        help="Number of iterations",
    )
    parser.add_argument("--output-dir", default="./outputs", help="Output directory")

    args = parser.parse_args()

    report = run_brainana(
        objective=args.objective,
        target_region=args.target_region,
        max_iterations=args.iterations,
        backend=args.backend,
        output_dir=args.output_dir,
    )
    print(f"\nReport saved to: {report}")


if __name__ == "__main__":
    main()
