"""
Follow the Weights v2: Correct interpretation

R measures "gradient step quality" - how good is this step?
NOT "position quality" - how good is this spot?

Use R to:
1. Scale step size by R (high R = good step, trust it more)
2. Accumulate momentum weighted by R
3. Early stop when R drops (diminishing returns)
"""

import numpy as np
from typing import Dict, Tuple, List
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
    """
    R = (E / nabla_H) * sigma^Df

    E = loss improvement (signal extracted this step)
    nabla_H = gradient magnitude (how much we're changing)
    Df = 5 - H (calibrated scaling)
    """
    E = max(loss_improvement, 0.001)
    nabla_H = max(grad_magnitude, 0.001)
    H = max(min(pred_entropy, 4.9), 0.1)
    Df = 5 - H

    sigma = np.e
    R = (E / nabla_H) * (sigma ** Df)
    return R


def train_standard(X_train, y_train, X_test, y_test, lr=0.1, n_epochs=100, seed=42):
    net = TinyNet(X_train.shape[1], 32, len(np.unique(y_train)), seed=seed)
    history = {'loss': [], 'acc': []}

    for epoch in range(n_epochs):
        probs, cache = net.forward(X_train)
        loss = cross_entropy_loss(probs, y_train)
        grads = net.backward(y_train, cache)
        net.update(grads, lr)

        test_probs, _ = net.forward(X_test)
        acc = np.mean(np.argmax(test_probs, axis=1) == y_test)
        history['loss'].append(loss)
        history['acc'].append(acc)

    return {'final_loss': history['loss'][-1], 'final_acc': history['acc'][-1], 'history': history}


def train_R_scaled_steps(X_train, y_train, X_test, y_test, base_lr=0.1, n_epochs=100, seed=42):
    """
    Scale each step by R.
    High R = this step is extracting good signal, trust it more.
    Low R = this step is noisy, trust it less.
    """
    net = TinyNet(X_train.shape[1], 32, len(np.unique(y_train)), seed=seed)
    history = {'loss': [], 'acc': [], 'R': [], 'effective_lr': []}
    prev_loss = None

    for epoch in range(n_epochs):
        probs, cache = net.forward(X_train)
        loss = cross_entropy_loss(probs, y_train)
        grads = net.backward(y_train, cache)
        H = prediction_entropy(probs)
        grad_mag = np.sqrt(sum(np.sum(g**2) for g in grads.values()))

        # Compute R for THIS step
        if prev_loss is not None:
            improvement = prev_loss - loss
            R = compute_R(improvement, grad_mag, H)

            # Scale LR by R (normalized)
            # Use ratio to recent average for stability
            if len(history['R']) > 5:
                R_avg = np.mean(history['R'][-10:])
                R_ratio = R / (R_avg + 1e-10)
                effective_lr = base_lr * np.clip(R_ratio, 0.1, 2.0)
            else:
                effective_lr = base_lr
        else:
            R = 1.0
            effective_lr = base_lr

        net.update(grads, effective_lr)

        test_probs, _ = net.forward(X_test)
        acc = np.mean(np.argmax(test_probs, axis=1) == y_test)

        history['loss'].append(loss)
        history['acc'].append(acc)
        history['R'].append(R)
        history['effective_lr'].append(effective_lr)
        prev_loss = loss

    return {'final_loss': history['loss'][-1], 'final_acc': history['acc'][-1], 'history': history}


def train_R_momentum(X_train, y_train, X_test, y_test, base_lr=0.1, n_epochs=100, seed=42):
    """
    R-weighted momentum: accumulate gradients weighted by their R.
    High R gradients contribute more to momentum.
    """
    net = TinyNet(X_train.shape[1], 32, len(np.unique(y_train)), seed=seed)
    history = {'loss': [], 'acc': [], 'R': []}
    prev_loss = None

    # Momentum accumulators
    momentum = {'W1': 0, 'b1': 0, 'W2': 0, 'b2': 0}
    beta = 0.9

    for epoch in range(n_epochs):
        probs, cache = net.forward(X_train)
        loss = cross_entropy_loss(probs, y_train)
        grads = net.backward(y_train, cache)
        H = prediction_entropy(probs)
        grad_mag = np.sqrt(sum(np.sum(g**2) for g in grads.values()))

        # Compute R
        if prev_loss is not None:
            improvement = prev_loss - loss
            R = compute_R(improvement, grad_mag, H)
        else:
            R = 1.0

        # R-weighted momentum update
        R_weight = np.clip(R / (np.mean(history['R'][-10:]) + 1e-10) if len(history['R']) > 5 else 1.0, 0.1, 2.0)

        for key in momentum:
            # Weight current gradient by R
            momentum[key] = beta * momentum[key] + (1 - beta) * R_weight * grads[key]

        # Apply momentum
        net.W1 -= base_lr * momentum['W1']
        net.b1 -= base_lr * momentum['b1']
        net.W2 -= base_lr * momentum['W2']
        net.b2 -= base_lr * momentum['b2']

        test_probs, _ = net.forward(X_test)
        acc = np.mean(np.argmax(test_probs, axis=1) == y_test)

        history['loss'].append(loss)
        history['acc'].append(acc)
        history['R'].append(R)
        prev_loss = loss

    return {'final_loss': history['loss'][-1], 'final_acc': history['acc'][-1], 'history': history}


def train_R_early_stop(X_train, y_train, X_test, y_test, base_lr=0.1, max_epochs=200, seed=42):
    """
    Early stopping based on R.
    When R drops to X% of peak (ratio threshold), stop.
    """
    net = TinyNet(X_train.shape[1], 32, len(np.unique(y_train)), seed=seed)
    history = {'loss': [], 'acc': [], 'R': []}
    prev_loss = None
    R_peak = 0
    below_threshold_count = 0
    R_threshold_ratio = 0.1
    patience = 10

    for epoch in range(max_epochs):
        probs, cache = net.forward(X_train)
        loss = cross_entropy_loss(probs, y_train)
        grads = net.backward(y_train, cache)
        H = prediction_entropy(probs)
        grad_mag = np.sqrt(sum(np.sum(g**2) for g in grads.values()))

        # Compute R
        if prev_loss is not None:
            improvement = prev_loss - loss
            R = compute_R(improvement, grad_mag, H)
        else:
            R = 1.0

        # Track peak
        if R > R_peak:
            R_peak = R

        # Check early stop condition
        R_ratio = R / (R_peak + 1e-10)
        if epoch > 20 and R_ratio < R_threshold_ratio:
            below_threshold_count += 1
        else:
            below_threshold_count = 0

        net.update(grads, base_lr)

        test_probs, _ = net.forward(X_test)
        acc = np.mean(np.argmax(test_probs, axis=1) == y_test)

        history['loss'].append(loss)
        history['acc'].append(acc)
        history['R'].append(R)
        prev_loss = loss

        # Early stop
        if below_threshold_count >= patience:
            break

    return {
        'final_loss': history['loss'][-1],
        'final_acc': history['acc'][-1],
        'stopped_epoch': len(history['loss']),
        'history': history
    }


def generate_data(n_samples=500, n_features=10, n_classes=3, seed=42):
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features)
    centers = np.random.randn(n_classes, n_features) * 2
    distances = np.array([np.sum((X - c) ** 2, axis=1) for c in centers])
    y = np.argmin(distances, axis=0)
    X += np.random.randn(*X.shape) * 0.3
    split = int(0.8 * n_samples)
    return X[:split], y[:split], X[split:], y[split:]


if __name__ == "__main__":
    print("=" * 60)
    print("FOLLOW THE WEIGHTS v2: Correct R interpretation")
    print("=" * 60)
    print()
    print("R measures step quality, not position quality.")
    print("Use R to weight steps, not choose directions.")
    print()

    X_train, y_train, X_test, y_test = generate_data()

    print("-" * 60)
    print("Training (100 epochs)...")
    print("-" * 60)

    standard = train_standard(X_train, y_train, X_test, y_test)
    print(f"Standard SGD:     loss={standard['final_loss']:.4f}, acc={standard['final_acc']:.4f}")

    scaled = train_R_scaled_steps(X_train, y_train, X_test, y_test)
    print(f"R-scaled steps:   loss={scaled['final_loss']:.4f}, acc={scaled['final_acc']:.4f}")

    momentum = train_R_momentum(X_train, y_train, X_test, y_test)
    print(f"R-weighted momentum: loss={momentum['final_loss']:.4f}, acc={momentum['final_acc']:.4f}")

    early = train_R_early_stop(X_train, y_train, X_test, y_test, max_epochs=200)
    print(f"R-early stop:     loss={early['final_loss']:.4f}, acc={early['final_acc']:.4f} (stopped@{early['stopped_epoch']})")

    # Compare convergence speed
    print("\n" + "-" * 60)
    print("Convergence speed (epochs to 85% accuracy)")
    print("-" * 60)

    def epochs_to_threshold(history, threshold=0.85):
        for i, acc in enumerate(history['acc']):
            if acc >= threshold:
                return i + 1
        return len(history['acc'])

    std_epochs = epochs_to_threshold(standard['history'])
    scaled_epochs = epochs_to_threshold(scaled['history'])
    mom_epochs = epochs_to_threshold(momentum['history'])

    print(f"Standard:   {std_epochs} epochs")
    print(f"R-scaled:   {scaled_epochs} epochs")
    print(f"R-momentum: {mom_epochs} epochs")

    # Verdict
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    improvements = []
    if scaled['final_acc'] > standard['final_acc']:
        improvements.append("R-scaled steps")
    if momentum['final_acc'] > standard['final_acc']:
        improvements.append("R-momentum")
    if early['final_acc'] >= standard['final_acc'] and early['stopped_epoch'] < 100:
        improvements.append(f"R-early stop ({early['stopped_epoch']} vs 100 epochs)")

    if improvements:
        print("\n** R-GUIDED METHODS HELP:")
        for imp in improvements:
            print(f"   - {imp}")
    else:
        print("\nX  NO IMPROVEMENT over standard SGD")
        print("   R measures step quality but doesn't improve optimization")
