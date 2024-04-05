from experiment_utilities.plotting import bootstrapped_mean_and_ci
import numpy as np
import matplotlib.pyplot as plt
class Test:
    def test_bootstrapped_mean_and_ci(self):
        data = np.random.randn(10, 15) *   np.random.rand(10, 1) + np.random.randn(10, 1)
        mean, ci_lower, ci_higher = bootstrapped_mean_and_ci(data)
        plt.plot(mean)
        plt.plot(ci_lower)
        plt.plot(ci_higher)
        plt.show()
        pass
