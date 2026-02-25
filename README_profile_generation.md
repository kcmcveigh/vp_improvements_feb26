# MADRS Profile Generation — Hierarchical Latent Factor Method

## Overview

Given a **target total MADRS score** (0-60), the generator produces a clinically plausible
distribution of scores across the 10 MADRS items.

The core idea: depression is not one thing. A patient scoring 30 might be primarily sad,
primarily anhedonic, or primarily neurovegetative. The generator captures this by sampling
from **4 latent clinical dimensions** that co-vary realistically, then mapping those
latent values down to individual item scores.

```
Target Score (e.g. 30)
        |
        v
┌──────────────────────────────────┐
│  4 Latent Factors (correlated)   │
│  Sadness / Neuroveg / Detach /   │
│  Negative Thought                │
└──────┬───────┬───────┬───────┬───┘
       v       v       v       v
   ┌───────┐ ┌─────┐ ┌─────┐ ┌─────┐
   │Rep Sad│ │Inner│ │Conc │ │Pess │
   │App Sad│ │Sleep│ │Lass │ │Suic │
   │       │ │Appet│ │Feel │ │     │
   └───────┘ └─────┘ └─────┘ └─────┘
   Factor 0  Factor 1 Factor 2 Factor 3
        |
        v
  Adjust to hit target total
        |
        v
  Apply clinical plausibility rules
        |
        v
  Final profile (10 integer scores, each 0-6)
```

---

## Step-by-step detail

### Step 1: Compute factor means from target score

The target score is divided equally across 10 items as a baseline:

```
base_per_item = target_score / 10
```

Each factor's mean is then adjusted so that, after item offsets are added back,
the expected total across items in that factor equals its fair share of points.

```
factor_mean = (base_per_item * n_items_in_factor - sum_of_offsets) / n_items_in_factor
```

**Item offsets** shift individual items up or down relative to their factor. They encode
clinical priors about which symptoms appear first as severity increases:

| Item                     | Offset | Effect                                      |
|--------------------------|--------|---------------------------------------------|
| Reported Sadness         | +2.0   | Appears early, scores high even at low severity |
| Apparent Sadness         |  0.0   | Neutral                                     |
| Inner Tension            |  0.0   | Neutral                                     |
| Reduced Sleep            |  0.0   | Neutral                                     |
| Reduced Appetite         | -0.2   | Slightly suppressed                         |
| Concentration Difficulty |  0.0   | Neutral                                     |
| Lassitude                |  0.0   | Neutral                                     |
| Inability to Feel        | -0.5   | Requires moderate severity to appear         |
| Pessimistic Thoughts     | -0.5   | Requires moderate severity to appear         |
| Suicidal Thoughts        | -2.0   | Strongly suppressed; only emerges at high severity |

The offsets redistribute points *within* a factor but do not change the expected total.

### Step 2: Sample correlated latent factors

The 4 factor scores are drawn from a multivariate normal distribution:

```
factor_scores ~ MVN(factor_means, factor_covariance)
```

The **factor correlation matrix** determines how the dimensions co-vary. A patient
who draws high on Sadness will tend (but not always) to draw high on Detachment.

```
              Sadness  Neuroveg  Detach  NegThought
Sadness         1.00      0.60    0.60        0.60
Neuroveg        0.60      1.00    0.60        0.60
Detach          0.60      0.60    1.00        0.60
NegThought      0.60      0.60    0.60        1.00
```

**Severity scaling:** at low target scores (< 30), the off-diagonal correlations are
attenuated by `min(1.0, target_score / 30)`. This captures the clinical observation
that factor structure is less defined at low severity (floor effects compress variance).

### Step 3: Generate item scores from factor scores

Each item gets a continuous value:

```
item_value = factor_score[item's_factor] + item_offset + noise
```

where `noise ~ N(0, item_residual_sd)`. The residual SD controls how tightly each item
tracks its factor (currently 0.5 for all items).

The continuous value is then rounded and clipped to the integer range [0, 6]:

```
discrete_score = clip(round(item_value), 0, 6)
```

### Step 4: Adjust to hit target total

After discretization, the sum of item scores typically won't exactly match the target.
A greedy adjustment loop corrects this:

- **If under target:** increment the item with the largest positive residual
  (continuous value most above its discrete score — the item that "deserves" more)
- **If over target:** decrement the item with the largest negative residual
  (discrete score most above its continuous value — the item that was rounded up most)

This runs for up to 100 iterations. No clinical rules are applied during this loop
to avoid the rules and adjustments fighting each other.

### Step 5: Apply clinical plausibility rules (final pass)

After the adjustment loop converges, clinical rules are applied once as a final guarantee.
This may shift the total score by 1-2 points but ensures clinical validity.

**Rules:**

1. **Mood Gate:** If Reported Sadness <= 1, cap Suicidal Thoughts at 1 and
   Pessimistic Thoughts at 2. (Can't have high suicidality without meaningful sadness.)

2. **Anhedonia Link:** If Inability to Feel >= 4 but Reported Sadness < 2,
   raise Reported Sadness to 3. (Severe emotional numbness implies depression.)

3. **Tension-Sleep Link:** If Inner Tension >= 4 but Reduced Sleep = 0,
   set Reduced Sleep to 2. (High anxiety without any sleep impact is implausible.)

### Design tradeoffs

| Property                | Behavior                                           |
|-------------------------|----------------------------------------------------|
| Target score accuracy   | Usually exact; may drift 1-2 points after rule pass |
| Clinical plausibility   | Guaranteed by final rule pass                       |
| Inter-item correlations | Emerge from shared factor structure                 |
| Profile diversity       | High — latent sampling + residual noise             |
| Severity sensitivity    | Offsets + scaled covariance adapt to score level    |

---

## Factor-to-item mapping (Hopkins CFA structure)

| Factor              | Items                                                    |
|---------------------|----------------------------------------------------------|
| **Sadness**         | Reported Sadness, Apparent Sadness                       |
| **Neurovegetative** | Inner Tension, Reduced Sleep, Reduced Appetite           |
| **Detachment**      | Concentration Difficulties, Lassitude, Inability to Feel |
| **Negative Thought**| Pessimistic Thoughts, Suicidal Thoughts                  |
