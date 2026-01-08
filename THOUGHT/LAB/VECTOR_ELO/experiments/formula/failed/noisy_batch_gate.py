"""
Noisy Batch Gate - Test when some batches are garbage

The gate matters when some gradients are BAD.
Scenario: Some batches have corrupted labels.

Can sigma^Df (or R) detect and reject bad batches?
"""

import numpy as np
from typing import Dict, Tuple
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

    def copy_weights(self):
        return {'W1': self.W1.copy(), 'b1': self.b1.copy(),
                'W2': self.W2.copy(), 'b2': self.b2.copy()}

    def set_weights(self, w):
        self.W1, self.b1 = w['W1'].copy(), w['b1'].copy()
        self.W2, self.b2 = w['W2'].copy(), w['b2'].copy()


def cross_entropy_loss(probs, y):
    m = y.shape[0]
    return -np.mean(np.log(probs[np.arange(m), y] + 1e-10))


def prediction_entropy(probs):
    probs_safe = np.clip(probs, 1e-10, 1.0)
    return np.mean(-np.sum(probs_safe * np.log(probs_safe), axis=1))


def compute_R(loss_improvement, grad_mag, H):
    """Full R formula."""
    E = max(loss_improvement, 0.001)
    nabla_H = max(grad_mag, 0.001)
    H = np.clip(H, 0.1, 4.9)
    Df = 5 - H
    return (E / nabla_H) * (np.e ** Df)


def generate_data(n_samples=500, n_features=10, n_classes=3, seed=42):
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features)
    centers = np.random.randn(n_classes, n_features) * 2
    distances = np.array([np.sum((X - c) ** 2, axis=1) for c in centers])
    y = np.argmin(distances, axis=0)
    X += np.random.randn(*X.shape) * 0.3
    split = int(0.8 * n_samples)
    return X[:split], y[:split], X[split:], y[split:]


def get_batch_with_corruption(X, y, batch_size, corrupt_prob, rng, n_classes=3):
    """
    Get a batch. With probability corrupt_prob, labels are randomized.
    Returns batch and whether it's corrupted.
    """
    idx = rng.choice(len(X), batch_size, replace=False)
    X_batch = X[idx]
    y_batch = y[idx].copy()

    is_corrupted = rng.random() < corrupt_prob
    if is_corrupted:
        # Randomize labels
        y_batch = rng.integers(0, n_classes, batch_size)

    return X_batch, y_batch, is_corrupted


def train_all_batches(X_train, y_train, X_test, y_test, n_classes, lr=0.1,
                      n_epochs=50, batch_size=32, corrupt_prob=0.3, seed=42):
    """Standard: take all batches regardless of corruption."""
    rng = np.random.default_rng(seed)
    net = TinyNet(X_train.shape[1], 32, n_classes, seed=seed)

    n_batches = len(X_train) // batch_size
    corrupted_taken = 0
    clean_taken = 0

    for epoch in range(n_epochs):
        for _ in range(n_batches):
            X_b, y_b, corrupted = get_batch_with_corruption(
                X_train, y_train, batch_size, corrupt_prob, rng, n_classes)

            probs, cache = net.forward(X_b)
            grads = net.backward(y_b, cache)
            net.update(grads, lr)

            if corrupted:
                corrupted_taken += 1
            else:
                clean_taken += 1

    test_probs, _ = net.forward(X_test)
    acc = np.mean(np.argmax(test_probs, axis=1) == y_test)
    return acc, corrupted_taken, clean_taken


def train_entropy_gated(X_train, y_train, X_test, y_test, n_classes, lr=0.1,
                        n_epochs=50, batch_size=32, corrupt_prob=0.3,
                        H_threshold=1.0, seed=42):
    """
    Gate by batch entropy.
    High entropy batch -> likely corrupted -> skip
    Low entropy batch -> likely clean -> take
    """
    rng = np.random.default_rng(seed)
    net = TinyNet(X_train.shape[1], 32, n_classes, seed=seed)

    n_batches = len(X_train) // batch_size
    corrupted_taken = 0
    corrupted_skipped = 0
    clean_taken = 0
    clean_skipped = 0

    for epoch in range(n_epochs):
        for _ in range(n_batches):
            X_b, y_b, corrupted = get_batch_with_corruption(
                X_train, y_train, batch_size, corrupt_prob, rng, n_classes)

            probs, cache = net.forward(X_b)
            H = prediction_entropy(probs)

            # Gate: low entropy = take, high entropy = skip
            if H < H_threshold:
                grads = net.backward(y_b, cache)
                net.update(grads, lr)
                if corrupted:
                    corrupted_taken += 1
                else:
                    clean_taken += 1
            else:
                if corrupted:
                    corrupted_skipped += 1
                else:
                    clean_skipped += 1

    test_probs, _ = net.forward(X_test)
    acc = np.mean(np.argmax(test_probs, axis=1) == y_test)
    return acc, corrupted_taken, corrupted_skipped, clean_taken, clean_skipped


def train_R_gated(X_train, y_train, X_test, y_test, n_classes, lr=0.1,
                  n_epochs=50, batch_size=32, corrupt_prob=0.3,
                  R_threshold=0.5, seed=42):
    """
    Gate by R computed from trial step.
    Low R (bad improvement per effort) -> skip
    High R -> take
    """
    rng = np.random.default_rng(seed)
    net = TinyNet(X_train.shape[1], 32, n_classes, seed=seed)

    n_batches = len(X_train) // batch_size
    R_history = []

    corrupted_taken = 0
    corrupted_skipped = 0
    clean_taken = 0
    clean_skipped = 0

    for epoch in range(n_epochs):
        for _ in range(n_batches):
            X_b, y_b, corrupted = get_batch_with_corruption(
                X_train, y_train, batch_size, corrupt_prob, rng, n_classes)

            probs, cache = net.forward(X_b)
            loss = cross_entropy_loss(probs, y_b)
            H = prediction_entropy(probs)
            grads = net.backward(y_b, cache)
            grad_mag = np.sqrt(sum(np.sum(g**2) for g in grads.values()))

            # Trial step
            saved = net.copy_weights()
            net.update(grads, lr)
            new_probs, _ = net.forward(X_b)
            new_loss = cross_entropy_loss(new_probs, y_b)
            improvement = loss - new_loss

            R = compute_R(improvement, grad_mag, H)
            R_history.append(R)

            # Gate relative to median R
            if len(R_history) > 20:
                R_median = np.median(R_history[-100:])
                ratio = R / (R_median + 1e-10)
            else:
                ratio = 1.0

            if ratio >= R_threshold:
                # Keep step
                if corrupted:
                    corrupted_taken += 1
                else:
                    clean_taken += 1
            else:
                # Revert
                net.set_weights(saved)
                if corrupted:
                    corrupted_skipped += 1
                else:
                    clean_skipped += 1

    test_probs, _ = net.forward(X_test)
    acc = np.mean(np.argmax(test_probs, axis=1) == y_test)
    return acc, corrupted_taken, corrupted_skipped, clean_taken, clean_skipped


if __name__ == "__main__":
    print("=" * 70)
    print("NOISY BATCH GATE - Can R/entropy detect bad batches?")
    print("=" * 70)
    print()
    print("Some batches have randomized labels (30% corruption)")
    print("Question: Can entropy or R gate reject corrupted batches?")
    print()

    X_train, y_train, X_test, y_test = generate_data()
    n_classes = len(np.unique(y_train))

    # Test with 30% corruption
    print("-" * 70)
    print("30% of batches have randomized labels")
    print("-" * 70)

    acc_all, corr_taken, clean_taken = train_all_batches(
        X_train, y_train, X_test, y_test, n_classes, corrupt_prob=0.3)
    print(f"\nAll batches:  acc={acc_all:.2%}  "
          f"(corrupted: {corr_taken}, clean: {clean_taken})")

    acc_H, ct, cs, clt, cls = train_entropy_gated(
        X_train, y_train, X_test, y_test, n_classes, corrupt_prob=0.3, H_threshold=1.0)
    print(f"H-gated (H<1.0): acc={acc_H:.2%}  "
          f"(corr_taken: {ct}, corr_skip: {cs}, clean_taken: {clt}, clean_skip: {cls})")

    acc_R, ct, cs, clt, cls = train_R_gated(
        X_train, y_train, X_test, y_test, n_classes, corrupt_prob=0.3, R_threshold=0.5)
    print(f"R-gated (R>0.5): acc={acc_R:.2%}  "
          f"(corr_taken: {ct}, corr_skip: {cs}, clean_taken: {clt}, clean_skip: {cls})")

    # Different thresholds
    print("\n" + "-" * 70)
    print("Different R thresholds")
    print("-" * 70)

    for thresh in [0.3, 0.5, 0.7, 0.9]:
        acc, ct, cs, clt, cls = train_R_gated(
            X_train, y_train, X_test, y_test, n_classes, corrupt_prob=0.3, R_threshold=thresh)
        rejection_rate = (cs + cls) / (ct + cs + clt + cls)
        corrupt_rejection = cs / (ct + cs) if (ct + cs) > 0 else 0
        clean_rejection = cls / (clt + cls) if (clt + cls) > 0 else 0
        print(f"R_thresh={thresh}: acc={acc:.2%}  "
              f"corrupt_reject={corrupt_rejection:.1%}  clean_reject={clean_rejection:.1%}")

    # Higher corruption
    print("\n" + "-" * 70)
    print("50% corruption")
    print("-" * 70)

    acc_all_50, _, _ = train_all_batches(
        X_train, y_train, X_test, y_test, n_classes, corrupt_prob=0.5)
    print(f"\nAll batches: acc={acc_all_50:.2%}")

    for thresh in [0.5, 0.7, 0.9]:
        acc, ct, cs, clt, cls = train_R_gated(
            X_train, y_train, X_test, y_test, n_classes, corrupt_prob=0.5, R_threshold=thresh)
        corrupt_rejection = cs / (ct + cs) if (ct + cs) > 0 else 0
        clean_rejection = cls / (clt + cls) if (clt + cls) > 0 else 0
        print(f"R_thresh={thresh}: acc={acc:.2%}  "
              f"corrupt_reject={corrupt_rejection:.1%}  clean_reject={clean_rejection:.1%}")

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    # Did gating selectively reject corrupted batches?
    _, ct, cs, clt, cls = train_R_gated(
        X_train, y_train, X_test, y_test, n_classes, corrupt_prob=0.3, R_threshold=0.7)
    corrupt_rejection = cs / (ct + cs) if (ct + cs) > 0 else 0
    clean_rejection = cls / (clt + cls) if (clt + cls) > 0 else 0

    if corrupt_rejection > clean_rejection * 1.5:
        print(f"\n** R GATE: SELECTIVELY REJECTS CORRUPTED")
        print(f"   Corrupted rejection: {corrupt_rejection:.1%}")
        print(f"   Clean rejection: {clean_rejection:.1%}")
        print(f"   Selectivity: {corrupt_rejection/clean_rejection:.2f}x")
    else:
        print(f"\nX  R GATE: NOT SELECTIVE")
        print(f"   Corrupted rejection: {corrupt_rejection:.1%}")
        print(f"   Clean rejection: {clean_rejection:.1%}")
        print("   R doesn't distinguish corrupted from clean batches")
