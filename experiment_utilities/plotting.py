import numpy as np
import scipy.stats


def bootstrapped_mean_and_ci(data, num_samples=1000, percentiles=[2.5, 97.5]):
    scores = []
    for _ in range(num_samples):
        idxs = np.random.randint(0, data.shape[1], size=(data.shape[1],))
        sampled_data = data[:, idxs]
        mean = np.mean(sampled_data, axis=1)
        scores.append(mean)
    scores = np.vstack(scores)
    mean = np.mean(scores, axis=0)
    ci_lower, ci_higher = np.percentile(scores, percentiles, axis=0)
    return mean, ci_lower, ci_higher