"""
Generate 500 virtual patient interview profiles using the Hopkins
factor-based CFA method, for comparison against the rule-based
ProfileGenerator and Borentain correlation-matrix approaches.

Usage:
    python3 generate_profiles_hopkins_run.py
    python3 generate_profiles_hopkins_run.py --n 500 --seed 42
"""

import argparse
import csv
import json
import os
import random
import sys

import numpy as np

# Setup path so we can import from the vp folder
VP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vp")
sys.path.insert(0, VP_ROOT)

import config
from generate_profiles_hopkins import FactorBasedMADRSGenerator


def generate_profiles(n_profiles: int, seed: int = 42):
    """Generate n profiles using the Hopkins factor-based method."""
    random.seed(seed)
    np.random.seed(seed)

    generator = FactorBasedMADRSGenerator()
    personas = config.PERSONAS
    styles = list(config.COMMUNICATION_STYLE_DESCRIPTIONS.keys())

    profiles = []

    for i in range(n_profiles):
        persona = random.choice(personas)
        style = random.choice(styles)
        target_score = int(round(random.gauss(25, 11)))
        target_score = max(5, min(55, target_score))

        scores_dict = generator.generate(target_score)

        actual_total = int(sum(scores_dict.values()))

        row = {
            "profile_id": i,
            "scale": "madrs",
            "persona_name": persona["Name"],
            "persona_age": int(persona["Age"]),
            "persona_occupation": persona["Occupation"],
            "persona_life_situation": persona["Life Situation"],
            "communication_style": style,
            "target_score": target_score,
            "actual_total_score": actual_total,
        }
        for item_key, score_val in scores_dict.items():
            row[item_key] = int(score_val)

        profiles.append(row)

    print(f"Generated {len(profiles)}/{n_profiles} profiles (hopkins factor method)")
    return profiles


def save_profiles(profiles, output_dir):
    """Save profiles as both CSV and JSON."""
    os.makedirs(output_dir, exist_ok=True)

    # --- CSV ---
    csv_path = os.path.join(output_dir, "profiles_madrs.csv")
    if profiles:
        fieldnames = list(profiles[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(profiles)
        print(f"Saved CSV: {csv_path}")

    # --- JSON ---
    json_path = os.path.join(output_dir, "profiles_madrs.json")
    with open(json_path, "w") as f:
        json.dump(profiles, f, indent=2)
    print(f"Saved JSON: {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate profiles using Hopkins factor-based CFA method."
    )
    parser.add_argument(
        "--n", type=int, default=500,
        help="Number of profiles to generate (default: 500)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (default: ./generated_profiles_hopkins/)"
    )
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "generated_profiles_hopkins"
    )

    profiles = generate_profiles(args.n, seed=args.seed)
    save_profiles(profiles, output_dir)
    print(f"\nDone. Total profiles: {len(profiles)}")


if __name__ == "__main__":
    main()
