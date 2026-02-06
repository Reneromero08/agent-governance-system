"""
R Dynamics Analysis - What does R actually track during training?

R measures step quality. Let's see:
1. How R evolves over training
2. What phases R detects (rapid learning vs plateau)
3. Whether R peaks correlate with breakthrough moments
"""

import numpy as np
from typing import Dict, List, Tuple
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
    probs_safe = np.clip(probs, 1e-10, 1.0)
    return np.mean(-np.sum(probs_safe * np.log(probs_safe), axis=1))


def compute_R(loss_improvement, grad_magnitude, pred_entropy):
    """R = (E / nabla_H) * sigma^Df"""
    E = max(loss_improvement, 0.001)
    nabla_H = max(grad_magnitude, 0.001)
    H = max(min(pred_entropy, 4.9), 0.1)
    Df = 5 - H

    sigma = np.e
    R = (E / nabla_H) * (sigma ** Df)
    return R


def generate_data(n_samples=500, n_features=10, n_classes=3, seed=42):
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features)
    centers = np.random.randn(n_classes, n_features) * 2
    distances = np.array([np.sum((X - c) ** 2, axis=1) for c in centers])
    y = np.argmin(distances, axis=0)
    X += np.random.randn(*X.shape) * 0.3
    split = int(0.8 * n_samples)
    return X[:split], y[:split], X[split:], y[split:]


def analyze_R_dynamics(X_train, y_train, X_test, y_test, lr=0.1, n_epochs=100, seed=42):
    """Track R and related metrics across training."""
    net = TinyNet(X_train.shape[1], 32, len(np.unique(y_train)), seed=seed)

    history = {
        'loss': [],
        'acc': [],
        'R': [],
        'H': [],
        'grad_mag': [],
        'loss_delta': [],
        'acc_delta': []
    }

    prev_loss = None
    prev_acc = None

    for epoch in range(n_epochs):
        probs, cache = net.forward(X_train)
        loss = cross_entropy_loss(probs, y_train)
        grads = net.backward(y_train, cache)
        H = prediction_entropy(probs)
        grad_mag = np.sqrt(sum(np.sum(g**2) for g in grads.values()))

        test_probs, _ = net.forward(X_test)
        acc = np.mean(np.argmax(test_probs, axis=1) == y_test)

        # Compute metrics
        if prev_loss is not None:
            loss_delta = prev_loss - loss
            acc_delta = acc - prev_acc
            R = compute_R(loss_delta, grad_mag, H)
        else:
            loss_delta = 0
            acc_delta = 0
            R = 1.0

        history['loss'].append(loss)
        history['acc'].append(acc)
        history['R'].append(R)
        history['H'].append(H)
        history['grad_mag'].append(grad_mag)
        history['loss_delta'].append(loss_delta)
        history['acc_delta'].append(acc_delta)

        net.update(grads, lr)

        prev_loss = loss
        prev_acc = acc

    return history


def find_phases(history):
    """Identify training phases based on R behavior."""
    R = np.array(history['R'])
    loss = np.array(history['loss'])
    acc = np.array(history['acc'])

    # Phase detection based on R
    R_rolling_mean = np.convolve(R, np.ones(5)/5, mode='valid')

    phases = []

    # Phase 1: Initial high R (rapid learning)
    if len(R_rolling_mean) > 10:
        initial_R = np.mean(R_rolling_mean[:10])
        phases.append(('Initial', 0, 10, initial_R))

    # Find R peak
    R_peak_idx = np.argmax(R)
    R_peak_val = R[R_peak_idx]
    phases.append(('R Peak', R_peak_idx, R_peak_idx+1, R_peak_val))

    # Find when R drops below 10% of peak
    R_threshold = R_peak_val * 0.1
    for i in range(R_peak_idx, len(R)):
        if R[i] < R_threshold:
            phases.append(('Diminishing Returns', i, len(R), np.mean(R[i:])))
            break

    return phases


def main():
    print("=" * 70)
    print("R DYNAMICS ANALYSIS - What does R track during training?")
    print("=" * 70)

    X_train, y_train, X_test, y_test = generate_data()
    history = analyze_R_dynamics(X_train, y_train, X_test, y_test, n_epochs=100)

    # Print epoch-by-epoch
    print("\n" + "-" * 70)
    print("Epoch-by-epoch dynamics (every 10 epochs)")
    print("-" * 70)
    print(f"{'Epoch':>6} | {'Loss':>8} | {'Acc':>6} | {'R':>12} | {'H':>6} | {'d_loss':>8} | {'d_acc':>6}")
    print("-" * 70)

    for i in [0, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 99]:
        if i < len(history['loss']):
            print(f"{i:>6} | {history['loss'][i]:>8.4f} | {history['acc'][i]:>6.2%} | {history['R'][i]:>12.2f} | {history['H'][i]:>6.3f} | {history['loss_delta'][i]:>8.4f} | {history['acc_delta'][i]:>6.2%}")

    # Phase analysis
    print("\n" + "-" * 70)
    print("Training Phases (detected by R)")
    print("-" * 70)

    phases = find_phases(history)
    for name, start, end, r_val in phases:
        if start < len(history['loss']):
            end_idx = min(end, len(history['loss'])) - 1
            print(f"  {name:20} epochs {start:>3}-{end_idx:>3}, R={r_val:>10.2f}, acc={history['acc'][end_idx]:.2%}")

    # Key insights
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)

    R = np.array(history['R'])
    loss = np.array(history['loss'])
    acc = np.array(history['acc'])

    # R peak location
    R_peak_idx = np.argmax(R)
    print(f"\n1. R peaks at epoch {R_peak_idx}")
    print(f"   At R peak: loss={loss[R_peak_idx]:.4f}, acc={acc[R_peak_idx]:.2%}")

    # When does acc plateau?
    acc_plateau_idx = None
    for i in range(len(acc)-10):
        if np.std(acc[i:i+10]) < 0.01:  # Stable for 10 epochs
            acc_plateau_idx = i
            break

    if acc_plateau_idx:
        print(f"\n2. Accuracy plateaus at epoch {acc_plateau_idx}")
        print(f"   At plateau: loss={loss[acc_plateau_idx]:.4f}, acc={acc[acc_plateau_idx]:.2%}")

    # Correlation between R and future improvement
    print("\n3. R as predictor of NEXT step quality:")
    future_improvements = []
    R_at_prediction = []
    for i in range(len(R)-1):
        future_improvements.append(history['loss_delta'][i+1])
        R_at_prediction.append(R[i])

    corr = np.corrcoef(R_at_prediction[:-1], future_improvements[1:])[0,1]
    print(f"   Correlation(R[t], loss_improvement[t+1]) = {corr:.3f}")

    # When R says "stop"
    R_peak = np.max(R)
    threshold_ratio = 0.1
    stop_epoch = None
    for i in range(20, len(R)):
        if R[i] < R_peak * threshold_ratio:
            stop_epoch = i
            break

    if stop_epoch:
        print(f"\n4. R-based early stop signal at epoch {stop_epoch}")
        print(f"   At stop: acc={acc[stop_epoch]:.2%}")
        print(f"   Final acc (epoch 99): {acc[-1]:.2%}")
        print(f"   Accuracy loss from early stop: {acc[-1] - acc[stop_epoch]:.2%}")
        print(f"   Epochs saved: {100 - stop_epoch}")

    # The efficiency trade-off
    print("\n" + "=" * 70)
    print("R'S ROLE: Efficiency vs Accuracy Trade-off")
    print("=" * 70)

    print("""
R tells you:
- When you're extracting signal efficiently (high R)
- When diminishing returns set in (R drops)
- NOT which direction to go (that's the gradient's job)

The formula is a THERMOMETER, not a COMPASS.
It measures extraction efficiency, not optimal direction.
""")

    # What R IS good for
    print("=" * 70)
    print("PRACTICAL APPLICATIONS")
    print("=" * 70)

    print("""
1. EARLY STOPPING: Stop when R drops to X% of peak
   - Saves compute while maintaining most accuracy

2. LEARNING RATE SCHEDULING: Reduce LR when R drops
   - High R = trust the gradient more
   - Low R = smaller steps as signal weakens

3. ANOMALY DETECTION: Sudden R spikes/drops indicate:
   - Data distribution shift
   - Entering/leaving loss plateaus
   - Training instability

4. CURRICULUM LEARNING: Order samples by their R contribution
   - Train on high-R samples first
""")


if __name__ == "__main__":
    main()
