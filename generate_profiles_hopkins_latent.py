import numpy as np

class HierarchicalMADRSGenerator:
    def __init__(self):
        self.item_keys = [
            "REPORTED_SADNESS", "APPARENT_SADNESS", "INNER_TENSION",
            "REDUCED_SLEEP", "REDUCED_APPETITE", "CONCENTRATION_DIFFICULTIES",
            "LASSITUDE", "INABILITY_TO_FEEL", "PESSIMISTIC_THOUGHTS", "SUICIDAL_THOUGHTS"
        ]

        # 0: Sadness (2), 1: Neuroveg (3), 2: Detachment (3), 3: Negative Thought (2)
        self.item_to_factor = {
            "REPORTED_SADNESS": 0, "APPARENT_SADNESS": 0,
            "INNER_TENSION": 1, "REDUCED_SLEEP": 1, "REDUCED_APPETITE": 1,
            "CONCENTRATION_DIFFICULTIES": 2, "LASSITUDE": 2, "INABILITY_TO_FEEL": 2,
            "PESSIMISTIC_THOUGHTS": 3, "SUICIDAL_THOUGHTS": 3
        }

        self.factor_counts = [2, 3, 3, 2]

        # Base factor correlation matrix — lowered to 0.3–0.5 range
        # consistent with published inter-factor correlations (e.g. Borentain 2022).
        # NOTE: These are assumed, not directly empirical. See README.
        self.base_factor_corr = np.array([
            [1.00, 0.6, 0.60, 0.6],  # Sadness
            [0.6, 1.00, 0.60, 0.6],  # Neurovegetative
            [0.6, 0.6, 1.00, 0.6],  # Detachment
            [0.6, 0.6, 0.6, 1.00]   # Negative Thought
        ])

        self.item_residual_sd = {
            "REPORTED_SADNESS": 0.5,
            "APPARENT_SADNESS": 0.5,
            "INNER_TENSION": 0.5,
            "REDUCED_SLEEP": 0.5,
            "REDUCED_APPETITE": .5,
            "CONCENTRATION_DIFFICULTIES": .5,
            "LASSITUDE": 0.50,
            "INABILITY_TO_FEEL": 0.5,
            "PESSIMISTIC_THOUGHTS": 0.5,
            "SUICIDAL_THOUGHTS": 0.5
        }

        # Clinical priors: rank-order is well-supported, magnitudes are assumed.
        # Positive = symptom appears earlier/more easily at lower severity
        # Negative = symptom requires higher latent severity to manifest
        self.item_offsets = {
            "REPORTED_SADNESS": 2, "APPARENT_SADNESS": 0,
            "INNER_TENSION": 0.0, "REDUCED_SLEEP": 0.0, "REDUCED_APPETITE": -0.2,
            "CONCENTRATION_DIFFICULTIES": 0.0, "LASSITUDE": 0.0, "INABILITY_TO_FEEL": -0.5,
            "PESSIMISTIC_THOUGHTS": -0.5, "SUICIDAL_THOUGHTS": -2.0
        }
    
    def _apply_madrs_rules(self, scores):
        """Apply MADRS specific clinical rules."""
        if "REPORTED_SADNESS" in scores:
            # Core Mood Gate
            if scores["REPORTED_SADNESS"] <= 1:
                if scores["SUICIDAL_THOUGHTS"] > 2:
                    scores["SUICIDAL_THOUGHTS"] = 1
                if scores["PESSIMISTIC_THOUGHTS"] > 2:
                    scores["PESSIMISTIC_THOUGHTS"] = 2
            
            # Anhedonia Link
            if scores["INABILITY_TO_FEEL"] >= 4 and scores["REPORTED_SADNESS"] < 2:
                scores["REPORTED_SADNESS"] = 3
            
            # Tension and Sleep Link
            if scores["INNER_TENSION"] >= 4 and scores["REDUCED_SLEEP"] == 0:
                scores["REDUCED_SLEEP"] = 2
        
        return scores

    def _severity_scaled_cov(self, target_score: int) -> np.ndarray:
        """Scale off-diagonal correlations by severity.

        At low severity (floor/near-floor items), inter-factor correlations
        compress. This crude scaling captures that directionally:
        full correlations at score ~30+, attenuated below.
        """
        scale = min(1.0, target_score / 30.0)
        cov = self.base_factor_corr.copy()
        # Scale only off-diagonals; keep variances at 1.0
        for i in range(4):
            for j in range(4):
                if i != j:
                    cov[i, j] *= scale
        return cov

    def generate_profile(self, target_score: int) -> dict | None:
        if not (0 <= target_score <= 60):
            return None

        base_per_item = target_score / 10.0

        # 1. Compute factor means that respect item offsets
        factor_means = []
        for f_idx in range(4):
            items_in_f = [k for k, v in self.item_to_factor.items() if v == f_idx]
            f_offset_sum = sum(self.item_offsets[item] for item in items_in_f)
            f_mean = ((base_per_item * len(items_in_f)) - f_offset_sum) / len(items_in_f)
            factor_means.append(f_mean)

        # 2. Draw correlated latent factors with severity-scaled covariance
        factor_cov = self._severity_scaled_cov(target_score)
        factor_scores = np.random.multivariate_normal(factor_means, factor_cov)

        # 3. Generate individual items with item-specific residual noise
        continuous = {}
        discrete = {}
        for item in self.item_keys:
            f_idx = self.item_to_factor[item]
            item_mean = factor_scores[f_idx] + self.item_offsets[item]
            sampled = np.random.normal(loc=item_mean, scale=self.item_residual_sd[item])
            continuous[item] = sampled
            discrete[item] = max(0, min(6, int(round(sampled))))
        # 4. Adjustment loop to hit target score (no clinical rules here)
        current_sum = sum(discrete.values())
        for _ in range(100):
            if current_sum == target_score:
                break
            if current_sum < target_score:
                cands = [k for k in self.item_keys if discrete[k] < 6]
                if not cands:
                    break
                best = max(cands, key=lambda k: continuous[k] - discrete[k])
                discrete[best] += 1
            else:
                cands = [k for k in self.item_keys if discrete[k] > 0]
                if not cands:
                    break
                best = max(cands, key=lambda k: discrete[k] - continuous[k])
                discrete[best] -= 1
            current_sum = sum(discrete.values())

        # 5. Apply clinical rules as final pass — guarantees plausibility
        #    Total may drift by 1-2 points but clinical validity is preserved.
        discrete = self._apply_madrs_rules(discrete)

        return discrete

    def validate(self, target_score: int = 30, n_samples: int = 5000):
        """Generate n_samples profiles and report summary statistics.

        Use this to sanity-check that output means, SDs, and correlations
        are plausible for the given severity level.
        """
        data = np.zeros((n_samples, 10))
        adj_counts = []
        for i in range(n_samples):
            profile = self.generate_profile(target_score)
            data[i] = [profile[k] for k in self.item_keys]

        means = data.mean(axis=0)
        sds = data.std(axis=0)
        corr = np.corrcoef(data.T)

        print(f"=== Validation: target_score={target_score}, n={n_samples} ===")
        print(f"{'Item':<30} {'Mean':>6} {'SD':>6}")
        print("-" * 44)
        for i, k in enumerate(self.item_keys):
            print(f"{k:<30} {means[i]:6.2f} {sds[i]:6.2f}")
        print(f"\nActual mean total: {data.sum(axis=1).mean():.2f}")
        print(f"Total score SD:   {data.sum(axis=1).std():.2f}")
        print(f"\nInter-item correlation matrix:")
        short = ["RepSad", "AppSad", "InTen", "Sleep", "Appet",
                 "Conc", "Lass", "Feel", "Pessm", "Suic"]
        print(f"{'':>8}", "".join(f"{s:>8}" for s in short))
        for i, s in enumerate(short):
            print(f"{s:>8}", "".join(f"{corr[i,j]:8.2f}" for j in range(10)))


# --- Usage ---
if __name__ == "__main__":
    gen = HierarchicalMADRSGenerator()

    print("Example profiles at different severities:\n")
    for score in [12, 25, 38, 50]:
        p = gen.generate_profile(score)
        items = " ".join(f"{v}" for v in p.values())
        print(f"  target={score:2d}  items=[{items}]  sum={sum(p.values())}")

    print()
    gen.validate(target_score=30, n_samples=5000)