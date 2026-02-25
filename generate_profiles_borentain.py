import pandas as pd
import numpy as np

def generate_from_matrix(csv_path, target_score):
    # Load and prep matrix
    df = pd.read_csv(csv_path, index_col=0)
    corr_matrix = df.values
    n_items = len(df)
    
    # Generate a raw sample using the correlation matrix
    # We use a mean of 0 and let the correlation dictate the variance structure
    raw_sample = np.random.multivariate_normal(np.zeros(n_items), corr_matrix)
    
    # 1. Transform raw sample to probabilities using Softmax
    # A higher 'beta' makes the profile more 'stereotypical' to the correlations
    beta = 1.5
    exp_s = np.exp(raw_sample * beta)
    probs = exp_s / np.sum(exp_s)
    
    # 2. Distribute target_score points based on these probabilities
    final_scores = np.zeros(n_items, dtype=int)
    remaining_points = target_score
    
    # Iteratively assign points while respecting the 0-6 item limit
    while remaining_points > 0:
        # Only pick items that haven't hit the ceiling of 6
        valid_mask = final_scores < 6
        if not np.any(valid_mask): break # All items at 6
        
        # Re-normalize probabilities for valid items
        current_probs = probs[valid_mask] / probs[valid_mask].sum()
        valid_indices = np.where(valid_mask)[0]
        
        idx = np.random.choice(valid_indices, p=current_probs)
        final_scores[idx] += 1
        remaining_points -= 1
        
    return dict(zip(df.index, final_scores))