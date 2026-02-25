import numpy as np
from scipy.stats import multivariate_normal

class FactorBasedMADRSGenerator:
    def __init__(self):
        # 1. Define Factor-to-Item Loadings (Blue values from your image)
        self.loadings = {
            "SADNESS": {"APPARENT_SADNESS": 0.79, "REPORTED_SADNESS": 0.74},
            "NEUROVEG": {"INNER_TENSION": 0.37, "REDUCED_SLEEP": 0.36, "REDUCED_APPETITE": 0.42},
            "DETACH": {"CONCENTRATION_DIFFICULTIES": 0.52, "LASSITUDE": 0.61, "INABILITY_TO_FEEL": 0.64},
            "NEG_THOUGHT": {"PESSIMISTIC_THOUGHTS": 0.79, "SUICIDAL_THOUGHTS": 0.29}
        }
        
        # 2. Define Latent Factor Correlations (Curved lines on the left of your image)
        self.factor_names = ["SADNESS", "NEUROVEG", "DETACH", "NEG_THOUGHT"]
        self.factor_corr = np.array([
            [1.00, 0.83, 0.69, 0.38],  # Sadness
            [0.83, 1.00, 0.88, 0.57],  # Neuroveg
            [0.69, 0.88, 1.00, 0.62],  # Detach
            [0.38, 0.57, 0.62, 1.00]   # Neg Thought
        ])

    def generate(self, target_score: int):
        # Step A: Sample the Latent Patient Profile
        # This determines if they are 'mostly sad' or 'mostly neurovegetative'
        latent_values = multivariate_normal.rvs(mean=[0,0,0,0], cov=self.factor_corr)
        factor_map = dict(zip(self.factor_names, latent_values))

        # Step B: Calculate 'Propensity' for each item
        propensities = {}
        for factor, items in self.loadings.items():
            for item, loading in items.items():
                # Propensity = (Factor Score * Loading) + noise
                # We use exp() to ensure propensities are positive for sampling
                val = (factor_map[factor] * loading) 
                propensities[item] = np.exp(val) 

        # Step C: Distribute points based on propensities
        # This ensures the sum is exactly target_score
        items = list(propensities.keys())
        probs = np.array(list(propensities.values()))
        probs /= probs.sum()

        scores = {item: 0 for item in items}
        points_to_assign = target_score
        
        while points_to_assign > 0:
            # Respect MADRS ceiling of 6
            valid_indices = [i for i, item in enumerate(items) if scores[item] < 6]
            if not valid_indices: break
            
            p_sub = probs[valid_indices] / probs[valid_indices].sum()
            idx = np.random.choice(valid_indices, p=p_sub)
            scores[items[idx]] += 1
            points_to_assign -= 1
            
        return scores