"""
Test: VALIDATED Bayesian Connections to R Formula

Q15: Is R formally connected to posterior concentration and evidence accumulation?

This is a FIXED version with proper Bayesian methods:
1. Laplace approximation for posterior computation
2. Analytic KL divergence between Gaussian posteriors
3. Fisher information for information gain
4. Multiple trials with statistical validation

"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')


def softmax(x):
    """Stable softmax."""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy_loss(probs, y):
    """Cross-entropy loss."""
    m = y.shape[0]
    return -np.mean(np.log(probs[np.arange(m), y] + 1e-10))


def prediction_entropy(probs):
    """Shannon entropy of predictions."""
    probs_safe = np.clip(probs, 1e-10, 1.0)
    return np.mean(-np.sum(probs_safe * np.log(probs_safe), axis=1))


class TinyNet:
    """Simple network with Hessian computation."""
    def __init__(self, input_dim, hidden_dim, output_dim, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(output_dim)

    def forward(self, X):
        """Forward pass with cache."""
        self.X = X
        z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0, z1)
        z2 = self.a1 @ self.W2 + self.b2
        self.probs = softmax(z2)
        return self.probs

    def backward(self, y):
        """Backward pass with gradients."""
        m = y.shape[0]
        dz2 = self.probs.copy()
        dz2[np.arange(m), y] -= 1
        dz2 /= m
        dW2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self.a1 > 0)
        dW1 = self.X.T @ dz1
        db1 = np.sum(dz1, axis=0)
        return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}

    def update(self, grads, lr):
        """Update parameters."""
        self.W1 -= lr * grads['W1']
        self.b1 -= lr * grads['b1']
        self.W2 -= lr * grads['W2']
        self.b2 -= lr * grads['b2']

    def get_params(self):
        """Get flattened parameters."""
        return np.concatenate([self.W1.flatten(), self.b1, self.W2.flatten(), self.b2])

    def set_params(self, params):
        """Set parameters from flattened vector."""
        idx = 0
        self.W1 = params[idx:idx+self.W1.size].reshape(self.W1.shape)
        idx += self.W1.size
        self.b1 = params[idx:idx+self.b1.size]
        idx += self.b1.size
        self.W2 = params[idx:idx+self.W2.size].reshape(self.W2.shape)
        idx += self.W2.size
        self.b2 = params[idx:]


def compute_hessian_diagonal(net, X, y, epsilon=1e-5):
    """
    Compute diagonal of Hessian using finite differences.

    This is the curvature of the loss landscape.
    High diagonal = sharp posterior (high concentration)
    Low diagonal = flat posterior (low concentration)
    """
    params = net.get_params()
    n_params = len(params)

    hessian_diag = np.zeros(n_params)

    # Sample subset to save computation
    sample_indices = np.random.choice(n_params, size=min(200, n_params), replace=False)

    for i in sample_indices:
        # Forward difference
        params_plus = params.copy()
        params_plus[i] += epsilon
        net.set_params(params_plus)
        loss_plus = cross_entropy_loss(net.forward(X), y)

        # Backward difference
        params_minus = params.copy()
        params_minus[i] -= epsilon
        net.set_params(params_minus)
        loss_minus = cross_entropy_loss(net.forward(X), y)

        # Central difference for second derivative
        # f''(x) ≈ (f(x+h) - 2f(x) + f(x-h)) / h^2
        net.set_params(params)
        loss_center = cross_entropy_loss(net.forward(X), y)

        hessian_diag[i] = (loss_plus - 2 * loss_center + loss_minus) / (epsilon ** 2)

    return hessian_diag


def gaussian_kl(mu1, sigma1_sq, mu2, sigma2_sq):
    """
    Analytic KL divergence between two Gaussians.

    KL(N1 || N2) = 0.5 * [log(det2/det1) - d + tr(inv2 @ Sigma1) + (mu2-mu1)^T @ inv2 @ (mu2-mu1)]

    For diagonal Gaussians, simplifies to:
    KL = 0.5 * Σ [log(σ2²/σ1²) + σ1²/σ2² + (μ1-μ2)²/σ2² - 1]
    """
    eps = 1e-10

    # Ensure positive variances
    sigma1_sq = np.maximum(sigma1_sq, eps)
    sigma2_sq = np.maximum(sigma2_sq, eps)

    # Analytic KL for diagonal Gaussians
    term1 = np.log(sigma2_sq / sigma1_sq)
    term2 = sigma1_sq / sigma2_sq
    term3 = (mu1 - mu2) ** 2 / sigma2_sq

    kl = 0.5 * np.sum(term1 + term2 + term3 - 1)

    return kl


def laplace_posterior(net, X, y, regularization=1.0):
    """
    Compute Laplace approximation of posterior.

    Posterior ≈ N(θ_ML, (H + λI)^{-1})

    Where:
    - θ_ML: Maximum likelihood parameters (current weights)
    - H: Hessian of loss at θ_ML
    - λ: L2 regularization prior (improves conditioning)

    Returns: (mean, variance) of posterior
    """
    # Mean = current parameters (ML estimate)
    mean = net.get_params()

    # Variance = (Hessian + λI)^{-1}
    hessian_diag = compute_hessian_diagonal(net, X, y)

    # Add regularization for numerical stability
    posterior_var = 1.0 / (hessian_diag + regularization)

    return mean, posterior_var


def compute_fisher_information(net, X, y, n_samples=10):
    """
    Compute Fisher information matrix.

    In Bayesian inference, Fisher information I(θ) = -E[∂² log p(D|θ)/∂θ²]
    This measures how much information the data provides about parameters.

    For classification, can approximate using gradients of log-likelihood.
    """
    probs = net.forward(X)
    grads = net.backward(y)

    # Flatten all gradients
    grad_flat = np.concatenate([g.flatten() for g in grads.values()])

    # Fisher ≈ E[grad²] (for classification)
    # Use empirical average over data
    fisher_diag = grad_flat ** 2

    return np.mean(fisher_diag)


def compute_R(loss_improvement, grad_magnitude, pred_entropy):
    """Compute R formula."""
    E = max(loss_improvement, 0.001)
    nabla_S = max(grad_magnitude, 0.001)
    H = max(min(pred_entropy, 4.9), 0.1)
    Df = 5 - H
    sigma = np.e
    R = (E / nabla_S) * (sigma ** Df)
    return R


def log_marginal_likelihood_approx(net, X, y, regularization=1.0):
    """
    Approximate log marginal likelihood using Laplace approximation.

    log p(D) ≈ log p(D|θ_ML) - (k/2)*log(n) + log p(θ_ML) - 0.5*log|H|

    Where:
    - p(D|θ_ML): Likelihood at MLE
    - k: Number of parameters
    - p(θ_ML): Prior at MLE (assume N(0, σ²))
    - |H|: Hessian determinant

    For Laplace approximation with diagonal Hessian:
    log |H| ≈ Σ log(H_ii)
    """
    n_params = len(net.get_params())
    n_data = X.shape[0]

    # Log-likelihood at MLE
    probs = net.forward(X)
    log_likelihood = np.sum(np.log(probs[np.arange(len(y)), y] + 1e-10))

    # Hessian diagonal
    hessian_diag = compute_hessian_diagonal(net, X, y)
    log_det_hessian = np.sum(np.log(hessian_diag + regularization))

    # Prior at MLE (assuming N(0, I) prior)
    params = net.get_params()
    log_prior = -0.5 * np.sum(params ** 2)

    # Laplace approximation
    log_marginal = log_likelihood - 0.5 * n_params * np.log(n_data) + log_prior - 0.5 * log_det_hessian

    return log_marginal


def run_validated_bayesian_test(
    n_trials=5, n_samples=400, n_features=10, n_classes=3,
    n_epochs=40, lr=0.1
):
    """
    Run validated Bayesian test with multiple trials.

    Returns aggregate results with statistics.
    """
    print("=" * 80)
    print("VALIDATED BAYESIAN CONNECTION TEST (FIXED)")
    print("=" * 80)
    print()
    print(f"Running {n_trials} trials with different seeds...")
    print()

    all_results = {
        'corr_hessian': [],
        'corr_kl': [],
        'corr_fisher': [],
        'corr_loss': []
    }

    for trial in range(n_trials):
        seed = trial * 1000 + 42
        print(f"Trial {trial + 1}/{n_trials} (seed={seed})")
        print("-" * 80)

        # Generate data
        np.random.seed(seed)
        X_train = np.random.randn(n_samples, n_features)
        centers = np.random.randn(n_classes, n_features) * 2
        distances = np.array([np.sum((X_train - c) ** 2, axis=1) for c in centers])
        y_train = np.argmin(distances, axis=0)
        X_train += np.random.randn(*X_train.shape) * 0.2

        split = int(0.8 * n_samples)
        X_test = X_train[split:]
        y_test = y_train[split:]
        X_train = X_train[:split]
        y_train = y_train[:split]

        # Initialize network
        input_dim = X_train.shape[1]
        output_dim = len(np.unique(y_train))
        net = TinyNet(input_dim, 32, output_dim, seed=seed)

        # Track metrics
        history = {
            'loss': [], 'acc': [], 'R': [], 'hessian_diag_mean': [],
            'posterior_var_mean': [], 'kl_divergence': [], 'fisher': [],
            'pred_entropy': [], 'grad_mag': []
        }

        prev_posterior_mean = None
        prev_posterior_var = None
        prev_loss = None

        for epoch in range(n_epochs):
            # Forward pass
            probs = net.forward(X_train)
            loss = cross_entropy_loss(probs, y_train)
            acc = np.mean(np.argmax(probs, axis=1) == y_train)

            # Backward pass
            grads = net.backward(y_train)
            grad_mag = np.sqrt(sum(np.sum(g ** 2) for g in grads.values()))

            # Prediction entropy
            pred_H = prediction_entropy(probs)

            # === Bayesian Metrics ===

            # 1. Hessian diagonal (curvature)
            hessian_diag = compute_hessian_diagonal(net, X_train, y_train)
            hessian_mean = np.mean(np.abs(hessian_diag))

            # 2. Posterior using Laplace approximation
            posterior_mean, posterior_var = laplace_posterior(net, X_train, y_train)
            posterior_var_mean = np.mean(posterior_var)

            # 3. KL divergence between posteriors
            if prev_posterior_mean is not None and prev_posterior_var is not None:
                kl = gaussian_kl(posterior_mean, posterior_var,
                                prev_posterior_mean, prev_posterior_var)
            else:
                kl = 0.0
            prev_posterior_mean = posterior_mean.copy()
            prev_posterior_var = posterior_var.copy()

            # 4. Fisher information
            fisher = compute_fisher_information(net, X_train, y_train)

            # Compute R
            if prev_loss is not None:
                loss_improvement = prev_loss - loss
                R = compute_R(loss_improvement, grad_mag, pred_H)
            else:
                loss_improvement = 0
                R = 1.0

            # Store
            history['loss'].append(loss)
            history['acc'].append(acc)
            history['R'].append(R)
            history['hessian_diag_mean'].append(hessian_mean)
            history['posterior_var_mean'].append(posterior_var_mean)
            history['kl_divergence'].append(kl)
            history['fisher'].append(fisher)
            history['pred_entropy'].append(pred_H)
            history['grad_mag'].append(grad_mag)

            # Update
            net.update(grads, lr)
            prev_loss = loss

        # Compute correlations for this trial
        R = np.array(history['R'])
        hessian = np.array(history['hessian_diag_mean'])
        kl = np.array(history['kl_divergence'])
        fisher = np.array(history['fisher'])

        # Prediction 1: R vs Hessian (concentration)
        corr_hessian = np.corrcoef(R, hessian)[0, 1]

        # Prediction 2: R predicts KL (information gain)
        if len(R) > 1 and len(kl) > 1:
            corr_kl = np.corrcoef(R[:-1], kl[1:])[0, 1]
        else:
            corr_kl = 0.0

        # Prediction 2b: R vs Fisher (alternative information measure)
        corr_fisher = np.corrcoef(R, fisher)[0, 1]

        # R vs loss (sanity check)
        corr_loss = np.corrcoef(R, history['loss'])[0, 1]

        # Store
        all_results['corr_hessian'].append(corr_hessian)
        all_results['corr_kl'].append(corr_kl)
        all_results['corr_fisher'].append(corr_fisher)
        all_results['corr_loss'].append(corr_loss)

        print(f"  R vs Hessian: {corr_hessian:.4f}")
        print(f"  R vs KL: {corr_kl:.4f}")
        print(f"  R vs Fisher: {corr_fisher:.4f}")
        print()

    # Aggregate results
    print("=" * 80)
    print("AGGREGATE RESULTS (mean ± std)")
    print("=" * 80)
    print()

    for key in ['corr_hessian', 'corr_kl', 'corr_fisher', 'corr_loss']:
        values = np.array(all_results[key])
        mean = np.mean(values)
        std = np.std(values)

        print(f"{key:25s}: {mean:+.4f} ± {std:.4f}")

    print()

    # Statistical significance testing
    print("=" * 80)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 80)
    print()

    hessian_values = np.array(all_results['corr_hessian'])
    kl_values = np.array(all_results['corr_kl'])
    fisher_values = np.array(all_results['corr_fisher'])

    # t-test for significance
    from scipy import stats

    # Test if correlation is significantly different from 0
    def test_significance(values):
        # Compute t-statistic
        n = len(values)
        if n < 2:
            return 0.0, 1.0

        # Fisher z-transform
        z = np.arctanh(values)  # Fisher's r-to-z transform
        mean_z = np.mean(z)

        # Standard error
        se = 1 / np.sqrt(n - 3)

        # Back-transform
        mean_r = np.tanh(mean_z)
        ci_low = np.tanh(mean_z - 1.96 * se)
        ci_high = np.tanh(mean_z + 1.96 * se)

        return mean_r, (ci_low, ci_high)

    hessian_r, hessian_ci = test_significance(hessian_values)
    kl_r, kl_ci = test_significance(kl_values)
    fisher_r, fisher_ci = test_significance(fisher_values)

    print("Prediction 1: R tracks posterior concentration (Hessian)")
    print(f"  Mean correlation: {hessian_r:.4f}")
    print(f"  95% CI: [{hessian_ci[0]:.4f}, {hessian_ci[1]:.4f}]")
    if hessian_ci[0] > 0.3 or hessian_ci[1] < -0.3:
        print(f"  [+] VALIDATED: Significant correlation (CI excludes 0)")
    else:
        print(f"  [-] FALSIFIED: No significant correlation")
    print()

    print("Prediction 2: R predicts information gain (KL divergence)")
    print(f"  Mean correlation: {kl_r:.4f}")
    print(f"  95% CI: [{kl_ci[0]:.4f}, {kl_ci[1]:.4f}]")
    if kl_ci[0] > 0.5:
        print(f"  [+] VALIDATED: Strong predictive correlation (CI > 0.5)")
    elif kl_ci[0] > 0.3:
        print(f"  [*] PARTIAL: Weak predictive correlation (CI > 0.3)")
    else:
        print(f"  [-] FALSIFIED: No predictive correlation")
    print()

    print("Prediction 2b: R vs Fisher information")
    print(f"  Mean correlation: {fisher_r:.4f}")
    print(f"  95% CI: [{fisher_ci[0]:.4f}, {fisher_ci[1]:.4f}]")
    if fisher_ci[0] > 0.5:
        print(f"  [+] VALIDATED: Strong correlation with Fisher info")
    elif fisher_ci[0] > 0.3:
        print(f"  [*] PARTIAL: Weak correlation with Fisher info")
    else:
        print(f"  [-] FALSIFIED: No correlation with Fisher info")
    print()

    # Final verdict
    print("=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    print()

    passes = 0

    # Count passes
    if hessian_ci[0] > 0.3:
        passes += 1
    if kl_ci[0] > 0.5:
        passes += 1
    if fisher_ci[0] > 0.5:
        passes += 1

    print(f"Predictions validated: {passes}/3")
    print()

    if passes == 0:
        print("CONCLUSION: NO Bayesian connections found")
        print("  R does not track posterior concentration or information gain")
    elif passes == 1:
        print("CONCLUSION: WEAK Bayesian connections")
        print("  R has one validated connection, but overall link is weak")
    elif passes == 2:
        print("CONCLUSION: STRONG Bayesian connections")
        print("  R has strong connections to 2/3 Bayesian quantities")
    else:
        print("CONCLUSION: VALIDATED Bayesian connections")
        print("  R tracks all tested Bayesian quantities")

    return all_results


if __name__ == "__main__":
    results = run_validated_bayesian_test(
        n_trials=5,
        n_samples=400,
        n_features=10,
        n_classes=3,
        n_epochs=40,
        lr=0.1
    )
