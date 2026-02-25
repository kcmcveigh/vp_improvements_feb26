"""
Check MADRS profiles against the rule-based ProfileGenerator's
generating constraints and clinical plausibility rules.

Reports per-profile and aggregate violation counts.

Usage:
    python3 check_profile_violations.py --csv generated_profiles_hopkins_latent/profiles_madrs.csv
    python3 check_profile_violations.py --csv generated_profiles_borentain/profiles_madrs.csv --output_dir violation_reports/
"""

import argparse
import csv
import os
import pandas as pd


# ── Generating rules (from _generate_madrs_profile_for_total_score) ─────────
# These are the constrained ranges used when the rule-based generator builds
# profiles. Other generators may not follow them.

def check_generating_rules(row: dict) -> list[dict]:
    """Check a single profile against the rule-based generator's constraints.

    Returns a list of violation dicts.
    """
    violations = []
    s = {k: int(row[k]) for k in ITEM_COLS}

    # 1. Apparent sadness should be within ±1 of reported sadness
    diff = abs(s["APPARENT_SADNESS"] - s["REPORTED_SADNESS"])
    if diff > 1:
        violations.append({
            "rule_type": "generating",
            "rule": "apparent_sadness_within_1_of_reported",
            "detail": f"APPARENT_SADNESS={s['APPARENT_SADNESS']} vs "
                      f"REPORTED_SADNESS={s['REPORTED_SADNESS']} (diff={diff})",
        })

    # 2. Suicidal thoughts bounded by sadness level
    if s["REPORTED_SADNESS"] <= 1:
        max_suicide = 1
    elif s["REPORTED_SADNESS"] <= 3:
        max_suicide = min(3, s["REPORTED_SADNESS"] + 1)
    else:
        max_suicide = 6

    if s["REPORTED_SADNESS"] == 0:
        # When sadness=0, generator doesn't set suicidal thoughts at all (stays 0)
        if s["SUICIDAL_THOUGHTS"] > 0:
            violations.append({
                "rule_type": "generating",
                "rule": "suicidal_zero_when_no_sadness",
                "detail": f"SUICIDAL_THOUGHTS={s['SUICIDAL_THOUGHTS']} "
                          f"but REPORTED_SADNESS=0",
            })
    elif s["SUICIDAL_THOUGHTS"] > max_suicide:
        violations.append({
            "rule_type": "generating",
            "rule": "suicidal_thoughts_bounded_by_sadness",
            "detail": f"SUICIDAL_THOUGHTS={s['SUICIDAL_THOUGHTS']} > "
                      f"max_allowed={max_suicide} (REPORTED_SADNESS={s['REPORTED_SADNESS']})",
        })

    # 3. Pessimistic thoughts bounded: [0, min(6, sadness + 2)]
    if s["REPORTED_SADNESS"] == 0:
        if s["PESSIMISTIC_THOUGHTS"] > 0:
            violations.append({
                "rule_type": "generating",
                "rule": "pessimism_zero_when_no_sadness",
                "detail": f"PESSIMISTIC_THOUGHTS={s['PESSIMISTIC_THOUGHTS']} "
                          f"but REPORTED_SADNESS=0",
            })
    else:
        max_pessimism = min(6, s["REPORTED_SADNESS"] + 2)
        if s["PESSIMISTIC_THOUGHTS"] > max_pessimism:
            violations.append({
                "rule_type": "generating",
                "rule": "pessimistic_thoughts_bounded_by_sadness",
                "detail": f"PESSIMISTIC_THOUGHTS={s['PESSIMISTIC_THOUGHTS']} > "
                          f"max_allowed={max_pessimism} (REPORTED_SADNESS={s['REPORTED_SADNESS']})",
            })

    # 4. Anhedonia floor: inability_to_feel >= reported_sadness // 2
    if s["REPORTED_SADNESS"] > 0:
        min_anhedonia = s["REPORTED_SADNESS"] // 2
        if s["INABILITY_TO_FEEL"] < min_anhedonia:
            violations.append({
                "rule_type": "generating",
                "rule": "anhedonia_floor_by_sadness",
                "detail": f"INABILITY_TO_FEEL={s['INABILITY_TO_FEEL']} < "
                          f"min_allowed={min_anhedonia} (REPORTED_SADNESS={s['REPORTED_SADNESS']})",
            })

    # 5. All items in valid range [0, 6]
    for item in ITEM_COLS:
        if s[item] < 0 or s[item] > 6:
            violations.append({
                "rule_type": "generating",
                "rule": "item_range_0_to_6",
                "detail": f"{item}={s[item]} out of range [0, 6]",
            })

    # 6. Total score matches target
    actual = sum(s.values())
    target = int(row["target_score"])
    if actual != target:
        violations.append({
            "rule_type": "generating",
            "rule": "total_matches_target",
            "detail": f"actual_total={actual} != target_score={target}",
        })

    return violations


# ── Clinical plausibility rules (from _apply_madrs_rules) ──────────────────

def check_clinical_rules(row: dict) -> list[dict]:
    """Check a single profile against MADRS clinical plausibility rules.

    These are the rules in ProfileGenerator._apply_madrs_rules that
    *correct* implausible profiles after generation.
    """
    violations = []
    s = {k: int(row[k]) for k in ITEM_COLS}

    # Rule 1: Core Mood Gate
    # If REPORTED_SADNESS <= 1, SUICIDAL_THOUGHTS should be <= 2
    if s["REPORTED_SADNESS"] <= 1 and s["SUICIDAL_THOUGHTS"] > 2:
        violations.append({
            "rule_type": "clinical",
            "rule": "mood_gate_suicidal",
            "detail": f"REPORTED_SADNESS={s['REPORTED_SADNESS']} <= 1 but "
                      f"SUICIDAL_THOUGHTS={s['SUICIDAL_THOUGHTS']} > 2",
        })

    # Rule 2: Core Mood Gate
    # If REPORTED_SADNESS <= 1, PESSIMISTIC_THOUGHTS should be <= 2
    if s["REPORTED_SADNESS"] <= 1 and s["PESSIMISTIC_THOUGHTS"] > 2:
        violations.append({
            "rule_type": "clinical",
            "rule": "mood_gate_pessimism",
            "detail": f"REPORTED_SADNESS={s['REPORTED_SADNESS']} <= 1 but "
                      f"PESSIMISTIC_THOUGHTS={s['PESSIMISTIC_THOUGHTS']} > 2",
        })

    # Rule 3: Anhedonia Link
    # If INABILITY_TO_FEEL >= 4, REPORTED_SADNESS should be >= 2
    if s["INABILITY_TO_FEEL"] >= 4 and s["REPORTED_SADNESS"] < 2:
        violations.append({
            "rule_type": "clinical",
            "rule": "anhedonia_requires_sadness",
            "detail": f"INABILITY_TO_FEEL={s['INABILITY_TO_FEEL']} >= 4 but "
                      f"REPORTED_SADNESS={s['REPORTED_SADNESS']} < 2",
        })

    # Rule 4: Tension and Sleep Link
    # If INNER_TENSION >= 4, REDUCED_SLEEP should be > 0
    if s["INNER_TENSION"] >= 4 and s["REDUCED_SLEEP"] == 0:
        violations.append({
            "rule_type": "clinical",
            "rule": "tension_requires_sleep_disturbance",
            "detail": f"INNER_TENSION={s['INNER_TENSION']} >= 4 but "
                      f"REDUCED_SLEEP=0",
        })

    return violations


# ── Item columns ────────────────────────────────────────────────────────────

ITEM_COLS = [
    "REPORTED_SADNESS", "APPARENT_SADNESS", "INNER_TENSION",
    "REDUCED_SLEEP", "REDUCED_APPETITE", "CONCENTRATION_DIFFICULTIES",
    "LASSITUDE", "INABILITY_TO_FEEL", "PESSIMISTIC_THOUGHTS",
    "SUICIDAL_THOUGHTS",
]


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Check MADRS profiles for generating and clinical rule violations."
    )
    parser.add_argument(
        "--csv", type=str, required=True,
        help="Path to profiles_madrs.csv"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for violation report CSV (default: next to input CSV)"
    )
    args = parser.parse_args()

    csv_path = os.path.abspath(args.csv)
    output_dir = args.output_dir or os.path.dirname(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} profiles from {csv_path}\n")

    all_violations = []
    profiles_with_gen_violations = 0
    profiles_with_clin_violations = 0

    for _, row in df.iterrows():
        pid = int(row["profile_id"])
        gen_v = check_generating_rules(row)
        clin_v = check_clinical_rules(row)

        if gen_v:
            profiles_with_gen_violations += 1
        if clin_v:
            profiles_with_clin_violations += 1

        for v in gen_v + clin_v:
            v["profile_id"] = pid
            v["target_score"] = int(row["target_score"])
            v["actual_total_score"] = int(row["actual_total_score"])
            all_violations.append(v)

    # ── Summary ─────────────────────────────────────────────────────────────
    n = len(df)
    print("=" * 70)
    print(f"  VIOLATION SUMMARY  ({os.path.basename(csv_path)})")
    print("=" * 70)

    print(f"\n  Total profiles:                     {n}")
    print(f"  Profiles with generating violations: {profiles_with_gen_violations} "
          f"({100 * profiles_with_gen_violations / n:.1f}%)")
    print(f"  Profiles with clinical violations:   {profiles_with_clin_violations} "
          f"({100 * profiles_with_clin_violations / n:.1f}%)")

    # Count by rule
    gen_violations = [v for v in all_violations if v["rule_type"] == "generating"]
    clin_violations = [v for v in all_violations if v["rule_type"] == "clinical"]

    print(f"\n  Total generating rule violations:    {len(gen_violations)}")
    if gen_violations:
        gen_by_rule = {}
        for v in gen_violations:
            gen_by_rule[v["rule"]] = gen_by_rule.get(v["rule"], 0) + 1
        for rule, count in sorted(gen_by_rule.items(), key=lambda x: -x[1]):
            print(f"    {rule:<45} {count:>5}")

    print(f"\n  Total clinical rule violations:      {len(clin_violations)}")
    if clin_violations:
        clin_by_rule = {}
        for v in clin_violations:
            clin_by_rule[v["rule"]] = clin_by_rule.get(v["rule"], 0) + 1
        for rule, count in sorted(clin_by_rule.items(), key=lambda x: -x[1]):
            print(f"    {rule:<45} {count:>5}")

    print()

    # ── Save detailed violations CSV ────────────────────────────────────────
    if all_violations:
        out_path = os.path.join(output_dir, "violations_report.csv")
        fieldnames = ["profile_id", "target_score", "actual_total_score",
                       "rule_type", "rule", "detail"]
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_violations)
        print(f"  Detailed report saved to: {out_path}")
    else:
        print("  No violations found — no report file written.")


if __name__ == "__main__":
    main()
