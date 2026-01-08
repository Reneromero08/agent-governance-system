"""
Gradient R - Compute R from gradient properties, not step outcome

The gradient itself has structure:
- Magnitude (how much change proposed)
- Distribution (entropy across parameters)
- Concentration (is signal focused or diffuse?)

R = (E / nabla_H) * sigma^Df

Maybe:
- E = gradient signal strength
- nabla_H = gradient entropy (uncertainty)
- Df = 5 - H (complexity of gradient structure)
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


def gradient_entropy(grads: Dict) -> float:
    """
    Entropy of gradient distribution.
    How spread out is the gradient signal across parameters?
    """
    # Flatten all gradients
    all_grads = np.concatenate([g.flatten() for g in grads.values()])

    # Convert to probability distribution (squared magnitudes)
    grad_sq = all_grads ** 2
    if np.sum(grad_sq) < 1e-10:
        return 4.9  # Max entropy for zero gradient

    probs = grad_sq / np.sum(grad_sq)
    probs = probs[probs > 1e-10]

    # Shannon entropy
    H = -np.sum(probs * np.log(probs))

    # Normalize to [0, 5] range
    max_entropy = np.log(len(all_grads))
    H_normalized = 5 * H / (max_entropy + 1e-10)

    return np.clip(H_normalized, 0.1, 4.9)


def gradient_concentration(grads: Dict) -> float:
    """
    How concentrated is the gradient?
    High concentration = few parameters dominate = clear signal
    Low concentration = spread out = noisy signal
    """
    all_grads = np.concatenate([g.flatten() for g in grads.values()])
    grad_sq = all_grads ** 2

    if np.sum(grad_sq) < 1e-10:
        return 0.01

    # Top 10% of gradients contribution
    sorted_sq = np.sort(grad_sq)[::-1]
    top_10_pct = int(len(sorted_sq) * 0.1)
    concentration = np.sum(sorted_sq[:top_10_pct]) / np.sum(sorted_sq)

    return concentration


def compute_R_from_gradient(grads: Dict) -> float:
    """
    Compute R from gradient properties alone.

    E = gradient concentration (signal strength)
    H = gradient entropy (uncertainty)
    nabla_H = gradient magnitude (how much we're changing)
    Df = 5 - H
    """
    # E: concentration (focused gradient = high signal)
    E = gradient_concentration(grads)

    # H: entropy of gradient distribution
    H = gradient_entropy(grads)

    # nabla_H: gradient magnitude
    nabla_H = np.sqrt(sum(np.sum(g**2) for g in grads.values()))
    nabla_H = max(nabla_H, 0.001)

    # Normalize nabla_H to reasonable range
    nabla_H = nabla_H / 10  # Scale factor

    # Df from H
    Df = 5 - H

    # R formula
    sigma = np.e
    R = (E / nabla_H) * (sigma ** Df)

    return R


def generate_data(n_samples=500, n_features=10, n_classes=3, noise=0.3, seed=42):
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features)
    centers = np.random.randn(n_classes, n_features) * 2
    distances = np.array([np.sum((X - c) ** 2, axis=1) for c in centers])
    y = np.argmin(distances, axis=0)
    X += np.random.randn(*X.shape) * noise
    split = int(0.8 * n_samples)
    return X[:split], y[:split], X[split:], y[split:]


def train_with_gradient_R(X_train, y_train, X_test, y_test, lr=0.1, n_epochs=100, seed=42):
    """Track gradient R during training."""
    net = TinyNet(X_train.shape[1], 32, len(np.unique(y_train)), seed=seed)

    history = {
        'loss': [], 'acc': [], 'R': [],
        'grad_entropy': [], 'grad_concentration': [], 'grad_mag': [],
        'loss_improvement': []
    }

    prev_loss = None

    for epoch in range(n_epochs):
        probs, cache = net.forward(X_train)
        loss = cross_entropy_loss(probs, y_train)
        grads = net.backward(y_train, cache)

        # Compute R from gradient (BEFORE step)
        R = compute_R_from_gradient(grads)
        H = gradient_entropy(grads)
        conc = gradient_concentration(grads)
        mag = np.sqrt(sum(np.sum(g**2) for g in grads.values()))

        # Take step
        net.update(grads, lr)

        # Measure actual improvement
        new_probs, _ = net.forward(X_train)
        new_loss = cross_entropy_loss(new_probs, y_train)
        improvement = loss - new_loss

        test_probs, _ = net.forward(X_test)
        acc = np.mean(np.argmax(test_probs, axis=1) == y_test)

        history['loss'].append(loss)
        history['acc'].append(acc)
        history['R'].append(R)
        history['grad_entropy'].append(H)
        history['grad_concentration'].append(conc)
        history['grad_mag'].append(mag)
        history['loss_improvement'].append(improvement)

        prev_loss = new_loss

    return history


def analyze_gradient_R(history):
    """Does gradient R correlate with actual improvement?"""
    R = np.array(history['R'])
    improvement = np.array(history['loss_improvement'])

    # Correlation
    corr = np.corrcoef(R, improvement)[0, 1]

    # Does high R predict high improvement?
    R_median = np.median(R)
    high_R_improvement = np.mean(improvement[R > R_median])
    low_R_improvement = np.mean(improvement[R <= R_median])

    return {
        'correlation': corr,
        'high_R_improvement': high_R_improvement,
        'low_R_improvement': low_R_improvement,
        'ratio': high_R_improvement / (low_R_improvement + 1e-10)
    }


if __name__ == "__main__":
    print("=" * 70)
    print("GRADIENT R - Compute R from gradient properties")
    print("=" * 70)
    print()
    print("R computed BEFORE step, from gradient structure alone")
    print()

    X_train, y_train, X_test, y_test = generate_data()

    print("-" * 70)
    print("Training and tracking gradient R")
    print("-" * 70)

    history = train_with_gradient_R(X_train, y_train, X_test, y_test)

    # Show R evolution
    print(f"\n{'Epoch':>6} | {'R':>10} | {'H':>8} | {'Conc':>8} | {'Improve':>10}")
    print("-" * 55)

    for i in [0, 5, 10, 20, 30, 50, 70, 99]:
        print(f"{i:>6} | {history['R'][i]:>10.2f} | {history['grad_entropy'][i]:>8.3f} | "
              f"{history['grad_concentration'][i]:>8.3f} | {history['loss_improvement'][i]:>10.4f}")

    # Correlation analysis
    print("\n" + "-" * 70)
    print("Does gradient R predict step improvement?")
    print("-" * 70)

    analysis = analyze_gradient_R(history)
    print(f"\n  Correlation(R, improvement): {analysis['correlation']:.3f}")
    print(f"  High-R steps avg improvement: {analysis['high_R_improvement']:.4f}")
    print(f"  Low-R steps avg improvement:  {analysis['low_R_improvement']:.4f}")
    print(f"  Ratio: {analysis['ratio']:.2f}x")

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    if analysis['correlation'] > 0.5:
        print(f"\n** GRADIENT R: VALIDATED")
        print(f"   R computed from gradient structure predicts improvement")
        print(f"   Correlation: {analysis['correlation']:.3f}")
    elif analysis['correlation'] > 0.2:
        print(f"\n*  GRADIENT R: WEAK SIGNAL")
        print(f"   Some correlation but not strong: {analysis['correlation']:.3f}")
    else:
        print(f"\nX  GRADIENT R: NOT VALIDATED")
        print(f"   Gradient structure doesn't predict improvement")
        print(f"   Correlation: {analysis['correlation']:.3f}")
