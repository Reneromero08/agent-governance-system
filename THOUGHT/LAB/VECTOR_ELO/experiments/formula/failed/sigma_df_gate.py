"""
sigma^Df Gate - Just the fractal scaling factor

User said: "it's not direction it's yes or no gradients"
User pointed to: sigma^Df = Fractal scaling factor

Maybe the gate is just sigma^Df = e^(5-H)

Low entropy (H small) -> high Df -> large sigma^Df -> YES
High entropy (H large) -> low Df -> small sigma^Df -> NO

The gate is about CERTAINTY, not improvement prediction.
"""

import numpy as np
from typing import Dict
import warnings
warnings.filterwarnings('ignore')


class TinyNet:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, seed: int = None):
        if seed is not None:
            np.random.seed(seed)
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(output_dim)

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ self.W2 + self.b2
        exp_z2 = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
        probs = exp_z2 / np.sum(exp_z2, axis=1, keepdims=True)
        return probs, {'X': X, 'z1': z1, 'a1': a1, 'probs': probs}

    def backward(self, y, cache):
        m = y.shape[0]
        probs = cache['probs']
        dz2 = probs.copy()
        dz2[np.arange(m), y] -= 1
        dz2 /= m
        dW2 = cache['a1'].T @ dz2
        db2 = np.sum(dz2, axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (cache['z1'] > 0)
        dW1 = cache['X'].T @ dz1
        db1 = np.sum(dz1, axis=0)
        return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}

    def update(self, grads, lr):
        self.W1 -= lr * grads['W1']
        self.b1 -= lr * grads['b1']
        self.W2 -= lr * grads['W2']
        self.b2 -= lr * grads['b2']


def cross_entropy_loss(probs, y):
    m = y.shape[0]
    return -np.mean(np.log(probs[np.arange(m), y] + 1e-10))


def prediction_entropy(probs):
    """H from model predictions."""
    probs_safe = np.clip(probs, 1e-10, 1.0)
    return np.mean(-np.sum(probs_safe * np.log(probs_safe), axis=1))


def sigma_Df(H, sigma=np.e):
    """
    Just the fractal scaling factor.
    sigma^Df where Df = 5 - H

    Low H (certain) -> high Df -> large scale -> YES
    High H (uncertain) -> low Df -> small scale -> NO
    """
    H = np.clip(H, 0.1, 4.9)
    Df = 5 - H
    return sigma ** Df


def generate_data(n_samples=500, n_features=10, n_classes=3, noise=0.3, seed=42):
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features)
    centers = np.random.randn(n_classes, n_features) * 2
    distances = np.array([np.sum((X - c) ** 2, axis=1) for c in centers])
    y = np.argmin(distances, axis=0)
    X += np.random.randn(*X.shape) * noise
    split = int(0.8 * n_samples)
    return X[:split], y[:split], X[split:], y[split:]


def train_standard(X_train, y_train, X_test, y_test, lr=0.1, n_epochs=100, seed=42):
    """Standard SGD."""
    net = TinyNet(X_train.shape[1], 32, len(np.unique(y_train)), seed=seed)
    history = {'acc': []}

    for epoch in range(n_epochs):
        probs, cache = net.forward(X_train)
        grads = net.backward(y_train, cache)
        net.update(grads, lr)

        test_probs, _ = net.forward(X_test)
        acc = np.mean(np.argmax(test_probs, axis=1) == y_test)
        history['acc'].append(acc)

    return history['acc'][-1]


def train_sigma_weighted(X_train, y_train, X_test, y_test, lr=0.1, n_epochs=100, seed=42):
    """
    Weight gradient by sigma^Df.

    Low entropy (certain) -> big step
    High entropy (uncertain) -> small step
    """
    net = TinyNet(X_train.shape[1], 32, len(np.unique(y_train)), seed=seed)
    history = {'acc': [], 'H': [], 'sigma_Df': []}

    for epoch in range(n_epochs):
        probs, cache = net.forward(X_train)
        H = prediction_entropy(probs)
        grads = net.backward(y_train, cache)

        # Scale by sigma^Df (normalized)
        scale = sigma_Df(H)
        # Normalize to keep LR reasonable
        if len(history['sigma_Df']) > 5:
            mean_scale = np.mean(history['sigma_Df'][-10:])
            effective_lr = lr * (scale / mean_scale)
            effective_lr = np.clip(effective_lr, lr * 0.1, lr * 2.0)
        else:
            effective_lr = lr

        net.update(grads, effective_lr)

        test_probs, _ = net.forward(X_test)
        acc = np.mean(np.argmax(test_probs, axis=1) == y_test)

        history['acc'].append(acc)
        history['H'].append(H)
        history['sigma_Df'].append(scale)

    return history['acc'][-1], history


def train_sigma_gated(X_train, y_train, X_test, y_test, lr=0.1, n_epochs=100, seed=42, threshold=0.5):
    """
    Gate gradients by sigma^Df relative to threshold.

    sigma^Df > threshold -> YES, take step
    sigma^Df < threshold -> NO, skip step
    """
    net = TinyNet(X_train.shape[1], 32, len(np.unique(y_train)), seed=seed)
    history = {'acc': [], 'gates': []}

    sigma_history = []

    for epoch in range(n_epochs):
        probs, cache = net.forward(X_train)
        H = prediction_entropy(probs)
        grads = net.backward(y_train, cache)

        scale = sigma_Df(H)
        sigma_history.append(scale)

        # Gate relative to median
        if len(sigma_history) > 5:
            median_scale = np.median(sigma_history[-20:])
            ratio = scale / median_scale
            gate = 1 if ratio >= threshold else 0
        else:
            gate = 1

        if gate:
            net.update(grads, lr)

        test_probs, _ = net.forward(X_test)
        acc = np.mean(np.argmax(test_probs, axis=1) == y_test)

        history['acc'].append(acc)
        history['gates'].append(gate)

    return history['acc'][-1], np.mean(history['gates'])


def analyze_sigma_Df():
    """Analyze sigma^Df during training."""
    X_train, y_train, X_test, y_test = generate_data()

    print("sigma^Df during training:")
    print(f"{'Epoch':>6} | {'H':>8} | {'Df':>8} | {'sigma^Df':>12}")
    print("-" * 45)

    net = TinyNet(X_train.shape[1], 32, len(np.unique(y_train)), seed=42)

    for epoch in range(100):
        probs, cache = net.forward(X_train)
        H = prediction_entropy(probs)
        Df = 5 - H
        scale = sigma_Df(H)

        if epoch in [0, 5, 10, 20, 30, 50, 70, 99]:
            print(f"{epoch:>6} | {H:>8.3f} | {Df:>8.3f} | {scale:>12.2f}")

        grads = net.backward(y_train, cache)
        net.update(grads, 0.1)


if __name__ == "__main__":
    print("=" * 70)
    print("sigma^Df GATE - The fractal scaling factor")
    print("=" * 70)
    print()
    print("sigma^Df = e^(5-H)")
    print("Low H (certain) -> large scale -> YES (trust this gradient)")
    print("High H (uncertain) -> small scale -> NO (don't trust)")
    print()

    # Analyze sigma^Df evolution
    print("-" * 70)
    print("sigma^Df during standard training")
    print("-" * 70)
    analyze_sigma_Df()

    # Compare methods
    print("\n" + "-" * 70)
    print("Comparison: Standard vs sigma-weighted vs sigma-gated")
    print("-" * 70)

    X_train, y_train, X_test, y_test = generate_data()

    standard = train_standard(X_train, y_train, X_test, y_test)
    weighted_acc, weighted_hist = train_sigma_weighted(X_train, y_train, X_test, y_test)
    gated_acc, gate_rate = train_sigma_gated(X_train, y_train, X_test, y_test, threshold=0.8)

    print(f"\n  Standard SGD:     {standard:.2%}")
    print(f"  Sigma-weighted:   {weighted_acc:.2%}")
    print(f"  Sigma-gated (0.8): {gated_acc:.2%} (gate rate: {gate_rate:.1%})")

    # Test on noisy data
    print("\n" + "-" * 70)
    print("Noisy data test")
    print("-" * 70)

    X_noisy, y_noisy, X_test_n, y_test_n = generate_data(noise=0.8, seed=123)

    standard_n = train_standard(X_noisy, y_noisy, X_test_n, y_test_n)
    weighted_n, _ = train_sigma_weighted(X_noisy, y_noisy, X_test_n, y_test_n)
    gated_n, gate_rate_n = train_sigma_gated(X_noisy, y_noisy, X_test_n, y_test_n, threshold=0.8)

    print(f"\n  Standard SGD:     {standard_n:.2%}")
    print(f"  Sigma-weighted:   {weighted_n:.2%}")
    print(f"  Sigma-gated (0.8): {gated_n:.2%} (gate rate: {gate_rate_n:.1%})")

    # The insight
    print("\n" + "=" * 70)
    print("INSIGHT")
    print("=" * 70)

    print("""
sigma^Df encodes CERTAINTY:
- High certainty (low H) -> large sigma^Df -> confident update
- Low certainty (high H) -> small sigma^Df -> tentative update

In training, H starts LOW (random predictions, max entropy ~log(3)=1.1)
and stays relatively low as model becomes confident.

The formula says:
- When model is uncertain, TRUST the gradient more (high Df)
- When model is overconfident, be MORE CAUTIOUS (low Df)

Wait - that's backwards from what I expected!
""")

    # Check correlation
    print("\n" + "-" * 70)
    print("Correlation: sigma^Df vs actual improvement")
    print("-" * 70)

    net = TinyNet(X_train.shape[1], 32, len(np.unique(y_train)), seed=42)
    scales = []
    improvements = []

    for epoch in range(100):
        probs, cache = net.forward(X_train)
        H = prediction_entropy(probs)
        loss = cross_entropy_loss(probs, y_train)

        grads = net.backward(y_train, cache)
        net.update(grads, 0.1)

        new_probs, _ = net.forward(X_train)
        new_loss = cross_entropy_loss(new_probs, y_train)

        scales.append(sigma_Df(H))
        improvements.append(loss - new_loss)

    corr = np.corrcoef(scales, improvements)[0, 1]
    print(f"\n  Correlation(sigma^Df, improvement): {corr:.3f}")

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    if corr > 0.3:
        print(f"\n** sigma^Df: VALIDATES as certainty-weighted gate")
    else:
        print(f"\nX  sigma^Df alone doesn't predict improvement (corr={corr:.3f})")
