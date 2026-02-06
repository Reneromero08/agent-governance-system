"""
Test: PROPER Bayesian Connections to R Formula (Reconstructed)
PROPER implementation using exact Bayesian inference on Gaussian data.
Reconstructed from the successful test run that generated Q15_PROPER_TEST_RESULTS.md.

Hypothesis:
1. R relates to Likelihood Precision (1/sigma), not Posterior Precision (n/sigma^2).
2. R relates to Free Energy per sample, not total Evidence.
"""

import numpy as np

def compute_R(data):
    """
    Compute R = E / grad_S for a 1D cluster of data.
    E evaluated at the mean (max compatibility) -> E(0) = 1 (assuming Gaussian kernel).
    grad_S = sample standard deviation.
    """
    if len(data) < 2:
        return 0.0
    
    mu = np.mean(data)
    std = np.std(data, ddof=1) # Sample std deviation
    
    if std < 1e-9:
        return 1000.0 # Cap for singular clusters
        
    E = 1.0 
    R = E / std
    return R

def bayesian_inference_gaussian(data, prior_mu=0, prior_sigma=10.0, true_sigma_obs=None):
    """
    Exact Bayesian Inference for Gaussian Mean.
    """
    n = len(data)
    sample_mean = np.mean(data)
    
    # If true_sigma is unknown, we use sample std as estimator (approximate inference)
    if true_sigma_obs is None:
        sigma_obs = np.std(data, ddof=1)
        if sigma_obs < 1e-9: sigma_obs = 1e-3
    else:
        sigma_obs = true_sigma_obs
        
    # Precision
    tau_0 = 1.0 / (prior_sigma**2)
    tau_obs = 1.0 / (sigma_obs**2)
    
    # Posterior Precision = Prior Prec + n * Obs Prec
    tau_n = tau_0 + n * tau_obs
    sigma_n = np.sqrt(1.0 / tau_n)
    
    # Posterior Mean
    mu_n = (tau_0 * prior_mu + n * tau_obs * sample_mean) / tau_n
    
    return {
        'posterior_mean': mu_n,
        'posterior_std': sigma_n,
        'posterior_precision': tau_n,
        'likelihood_precision': tau_obs,
        'posterior_concentration': 1.0/sigma_n
    }

def run_test():
    np.random.seed(42)  # Same seed as report
    print("Running PROPER Bayesian R Correlation Test...", flush=True)
    print("-" * 60, flush=True)
    
    results = []
    
    # Experiment 1: Varying Sigma
    print("Experiment 1: Varying Spread (Sigma), Fixed N=20", flush=True)
    for true_sigma in np.linspace(0.1, 5.0, 50):
        data = np.random.normal(0, true_sigma, 20)
        
        R = compute_R(data)
        # Use simple sample estimation for Bayes to demonstrate mechanical identity
        bayes = bayesian_inference_gaussian(data)
        
        results.append({
            'type': 'vary_sigma',
            'sigma': true_sigma,
            'R': R,
            'post_prec': bayes['posterior_precision'],
            'lik_prec': bayes['likelihood_precision']
        })

    # Experiment 2: Varying N
    print("Experiment 2: Varying N, Fixed Sigma=1.0", flush=True)
    for n in range(5, 200, 5):
        data = np.random.normal(0, 1.0, n)
        
        R = compute_R(data)
        bayes = bayesian_inference_gaussian(data)
        
        results.append({
            'type': 'vary_n',
            'n': n,
            'R': R,
            'post_prec': bayes['posterior_precision'],
            'lik_prec': bayes['likelihood_precision']
        })
        
    # Analyze Exp 1
    res_sigma = [r for r in results if r['type'] == 'vary_sigma']
    Rs = np.array([r['R'] for r in res_sigma])
    SqrtLikPrec = np.sqrt(np.array([r['lik_prec'] for r in res_sigma]))
    
    corr_lik_sqrt = np.corrcoef(Rs, SqrtLikPrec)[0,1]
    # This should be exactly 1.0000 because R is mechanically 1/sample_std and lik_prec is 1/ sample_std^2
    print(f"Correlation R vs Sqrt(Likelihood Precision): {corr_lik_sqrt:.4f}", flush=True)
    
    # Analyze Exp 2
    res_n = [r for r in results if r['type'] == 'vary_n']
    Rs_n = np.array([r['R'] for r in res_n])
    PostPrecs_n = np.array([r['post_prec'] for r in res_n])
    
    corr_r_post = np.corrcoef(Rs_n, PostPrecs_n)[0,1]
    print(f"Correlation R vs Posterior Precision: {corr_r_post:.4f}", flush=True)

if __name__ == "__main__":
    run_test()
