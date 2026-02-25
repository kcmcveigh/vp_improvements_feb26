"""
Generate 500 virtual patient interview profiles and save as CSV + JSON
for easy analysis.

Usage:
    python generate_profiles.py
    python generate_profiles.py --n 500 --scale madrs
    python generate_profiles.py --n 500 --scale all
"""

import argparse
import csv
import json
import os
import random
import sys

# Setup path so we can import from the vp folder
VP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vp")
sys.path.insert(0, VP_ROOT)

import config
from core.profile_generator import ProfileGenerator
from core.scale_registry import SCALE_REGISTRY


def generate_profiles(n_profiles: int, scale_name: str, seed: int = 42):
    """Generate n patient profiles for a given scale.

    Returns a list of flat dicts ready for CSV/DataFrame consumption.
    """
    random.seed(seed)
    profile_gen = ProfileGenerator(scale_name)
    max_score = SCALE_REGISTRY.get_max_score(scale_name)

    personas = config.PERSONAS
    styles = list(config.COMMUNICATION_STYLE_DESCRIPTIONS.keys())

    profiles = []
    failed = 0

    for i in range(n_profiles):
        persona = random.choice(personas)
        style = random.choice(styles)
        target_score = int(round(random.gauss(25, 11)))
        target_score = max(5, min(55, target_score))

        scores_dict = profile_gen.generate_profile_for_total_score(target_score)

        if scores_dict is None:
            failed += 1
            continue

        actual_total = sum(scores_dict.values())

        row = {
            "profile_id": i,
            "scale": scale_name,
            "persona_name": persona["Name"],
            "persona_age": persona["Age"],
            "persona_occupation": persona["Occupation"],
            "persona_life_situation": persona["Life Situation"],
            "communication_style": style,
            "target_score": target_score,
            "actual_total_score": actual_total,
        }
        # Add each item score as its own column
        for item_key, score_val in scores_dict.items():
            row[item_key] = score_val

        profiles.append(row)

    print(f"Generated {len(profiles)}/{n_profiles} profiles for {scale_name} "
          f"({failed} failed)")
    return profiles


def save_profiles(profiles, output_dir, scale_name):
    """Save profiles as both CSV and JSON."""
    os.makedirs(output_dir, exist_ok=True)

    # --- CSV ---
    csv_path = os.path.join(output_dir, f"profiles_{scale_name}.csv")
    if profiles:
        fieldnames = list(profiles[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(profiles)
        print(f"Saved CSV: {csv_path}")

    # --- JSON (for programmatic use) ---
    json_path = os.path.join(output_dir, f"profiles_{scale_name}.json")
    with open(json_path, "w") as f:
        json.dump(profiles, f, indent=2)
    print(f"Saved JSON: {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate virtual patient profiles for analysis."
    )
    parser.add_argument(
        "--n", type=int, default=500,
        help="Number of profiles to generate (default: 500)"
    )
    parser.add_argument(
        "--scale", type=str, default="all",
        help="Scale name (e.g. 'madrs', 'sigh_d') or 'all' (default: all)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (default: ./generated_profiles/)"
    )
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "generated_profiles"
    )

    available_scales = SCALE_REGISTRY.get_available_scales()

    if args.scale == "all":
        scales_to_run = available_scales
    else:
        if args.scale not in available_scales:
            print(f"Error: scale '{args.scale}' not found. "
                  f"Available: {available_scales}")
            sys.exit(1)
        scales_to_run = [args.scale]

    all_profiles = []
    for scale in scales_to_run:
        profiles = generate_profiles(args.n, scale, seed=args.seed)
        save_profiles(profiles, output_dir, scale)
        all_profiles.extend(profiles)

    # If multiple scales, also save a combined file
    if len(scales_to_run) > 1 and all_profiles:
        combined_csv = os.path.join(output_dir, "profiles_all.csv")
        # Union of all keys across scales (different scales have different items)
        all_keys = []
        seen = set()
        for p in all_profiles:
            for k in p.keys():
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)

        with open(combined_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_profiles)
        print(f"Saved combined CSV: {combined_csv}")

    print(f"\nDone. Total profiles: {len(all_profiles)}")


if __name__ == "__main__":
    main()
