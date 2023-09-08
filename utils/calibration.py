""" Create Calibration curve for of the estimator """
import numpy as np
from sklearn.calibration import calibration_curve
import seaborn as sns
import matplotlib.pyplot as plt

def plot_calibration_curve(y_true, y_prob, n_bins=5, confidence_level=0.95):
    """
    Creates a calibration plot for predicted probabilites, along with bootstrap
    confidence intervals.

    Parameters
    ----------
    y_true : pd.Series or array
        The ground truth labels.
    y_prob : pd.Series or array
        The predicte probabilities.
    n_bins : int, optional
        Number of bins to use for calculating the calibrations. The default is 5.
    confidence_level : float, optional
        Confidence level for bootstraped error bars. The default is 0.95.

    Returns
    -------
    None.
    
    """
    # Calculate the calibration curve data
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    
    
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    # Compute error bars using bootstrap with the desired confidence level
    n_samples = len(y_true)
    n_bootstraps = 100
    bootstrapped_tpr = []
    for _ in range(n_bootstraps):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        sample_true, sample_prob = y_true[indices], y_prob[indices]
        tpr, _ = calibration_curve(sample_true, sample_prob, n_bins=n_bins)
        bootstrapped_tpr.append(tpr)

    tpr_mean = np.mean(bootstrapped_tpr, axis=0)
    tpr_lower = np.percentile(bootstrapped_tpr, (1 - confidence_level) / 2 * 100, axis=0)
    tpr_upper = np.percentile(bootstrapped_tpr, (1 + confidence_level) / 2 * 100, axis=0)
    error_bars = (tpr_mean - tpr_lower, tpr_upper - tpr_mean)

    # Plot the calibration curve with error bars using seaborn
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim(0,1)
    plt.ylim(0,1)
    sns.lineplot(x = prob_pred, y = prob_true, marker='o', markersize=5, ci=None)
    plt.errorbar(prob_pred, tpr_mean, yerr=error_bars, fmt='none', elinewidth=2, capsize=4, capthick=2)
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives (True Positive Rate)")
    plt.title("Calibration Curve with Error Bars")
    plt.grid(True)
    plt.show()


