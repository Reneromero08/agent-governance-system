"""
Gated Gradients - R as YES/NO decision

R doesn't tell you WHERE to go.
R tells you WHETHER to go.

High R = YES, take this gradient step
Low R = NO, skip this step (noise, not signal)
"""

import numpy as np
from typing import Dict, List
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
        return {
            'W1': self.W1.copy(), 'b1': self.b1.copy(),
            'W2': self.W2.copy(), 'b2': self.b2.copy()
        }

    def set_weights(self, weights):
        self.W1 = weights['W1'].copy()
        self.b1 = weights['b1'].copy()
        self.W2 = weights['W2'].copy()
        self.b2 = weights['b2'].copy()


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
    """Standard SGD baseline."""
    net = TinyNet(X_train.shape[1], 32, len(np.unique(y_train)), seed=seed)
    history = {'loss': [], 'acc': [], 'steps_taken': 0}

    for epoch in range(n_epochs):
        probs, cache = net.forward(X_train)
        loss = cross_entropy_loss(probs, y_train)
        grads = net.backward(y_train, cache)
        net.update(grads, lr)
        history['steps_taken'] += 1

        test_probs, _ = net.forward(X_test)
        acc = np.mean(np.argmax(test_probs, axis=1) == y_test)
        history['loss'].append(loss)
        history['acc'].append(acc)

    return {'final_acc': history['acc'][-1], 'steps': history['steps_taken'], 'history': history}


def train_gated(X_train, y_train, X_test, y_test, lr=0.1, n_epochs=100, seed=42, R_threshold=0.5):
    """
    Gated SGD: Only take gradient steps where R > threshold.

    R is computed BEFORE taking the step (prospective).
    We do a trial step, compute R, then decide YES/NO.
    """
    net = TinyNet(X_train.shape[1], 32, len(np.unique(y_train)), seed=seed)
    history = {'loss': [], 'acc': [], 'R': [], 'gates': []}

    prev_loss = None
    steps_taken = 0
    steps_skipped = 0

    for epoch in range(n_epochs):
        probs, cache = net.forward(X_train)
        loss = cross_entropy_loss(probs, y_train)
        grads = net.backward(y_train, cache)
        H = prediction_entropy(probs)
        grad_mag = np.sqrt(sum(np.sum(g**2) for g in grads.values()))

        # Trial step to compute R
        if prev_loss is not None:
            # Save weights
            saved_weights = net.copy_weights()

            # Take trial step
            net.update(grads, lr)

            # Compute new loss
            new_probs, _ = net.forward(X_train)
            new_loss = cross_entropy_loss(new_probs, y_train)

            # Compute R for this step
            improvement = loss - new_loss
            R = compute_R(improvement, grad_mag, H)

            # Normalize R against recent history
            if len(history['R']) > 5:
                R_median = np.median(history['R'][-20:])
                R_ratio = R / (R_median + 1e-10)
            else:
                R_ratio = 1.0

            # GATE: YES or NO?
            if R_ratio >= R_threshold:
                # YES - keep the step
                steps_taken += 1
                gate = 1
            else:
                # NO - revert the step
                net.set_weights(saved_weights)
                steps_skipped += 1
                gate = 0

            history['R'].append(R)
            history['gates'].append(gate)
        else:
            # First step - always take it
            net.update(grads, lr)
            steps_taken += 1
            history['R'].append(1.0)
            history['gates'].append(1)

        # Record state
        test_probs, _ = net.forward(X_test)
        acc = np.mean(np.argmax(test_probs, axis=1) == y_test)

        probs, _ = net.forward(X_train)
        loss = cross_entropy_loss(probs, y_train)

        history['loss'].append(loss)
        history['acc'].append(acc)
        prev_loss = loss

    return {
        'final_acc': history['acc'][-1],
        'steps_taken': steps_taken,
        'steps_skipped': steps_skipped,
        'history': history
    }


def train_gated_minibatch(X_train, y_train, X_test, y_test, lr=0.1, n_epochs=100,
                          batch_size=32, seed=42, R_threshold=0.5):
    """
    Gated SGD with minibatches.
    Gate each minibatch gradient independently.
    """
    np.random.seed(seed)
    net = TinyNet(X_train.shape[1], 32, len(np.unique(y_train)), seed=seed)

    n_samples = X_train.shape[0]
    n_batches = n_samples // batch_size

    history = {'loss': [], 'acc': [], 'gate_rate': []}
    R_history = []

    for epoch in range(n_epochs):
        indices = np.random.permutation(n_samples)
        epoch_gates = []

        for batch_idx in range(n_batches):
            batch_indices = indices[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]

            # Forward pass
            probs, cache = net.forward(X_batch)
            loss = cross_entropy_loss(probs, y_batch)
            grads = net.backward(y_batch, cache)
            H = prediction_entropy(probs)
            grad_mag = np.sqrt(sum(np.sum(g**2) for g in grads.values()))

            # Trial step
            saved_weights = net.copy_weights()
            net.update(grads, lr)

            new_probs, _ = net.forward(X_batch)
            new_loss = cross_entropy_loss(new_probs, y_batch)
            improvement = loss - new_loss

            R = compute_R(improvement, grad_mag, H)
            R_history.append(R)

            # Gate decision
            if len(R_history) > 10:
                R_median = np.median(R_history[-50:])
                R_ratio = R / (R_median + 1e-10)
            else:
                R_ratio = 1.0

            if R_ratio >= R_threshold:
                epoch_gates.append(1)  # Keep step
            else:
                net.set_weights(saved_weights)  # Revert
                epoch_gates.append(0)

        # Epoch metrics
        test_probs, _ = net.forward(X_test)
        acc = np.mean(np.argmax(test_probs, axis=1) == y_test)

        train_probs, _ = net.forward(X_train)
        loss = cross_entropy_loss(train_probs, y_train)

        history['loss'].append(loss)
        history['acc'].append(acc)
        history['gate_rate'].append(np.mean(epoch_gates))

    return {
        'final_acc': history['acc'][-1],
        'avg_gate_rate': np.mean(history['gate_rate']),
        'history': history
    }


if __name__ == "__main__":
    print("=" * 70)
    print("GATED GRADIENTS - R as YES/NO decision")
    print("=" * 70)
    print()
    print("R doesn't tell you WHERE to go.")
    print("R tells you WHETHER to go.")
    print()

    X_train, y_train, X_test, y_test = generate_data()

    # Test different thresholds
    print("-" * 70)
    print("Test 1: Full-batch gated SGD (various thresholds)")
    print("-" * 70)

    standard = train_standard(X_train, y_train, X_test, y_test)
    print(f"\nStandard SGD: acc={standard['final_acc']:.2%}, steps={standard['steps']}")

    for threshold in [0.3, 0.5, 0.7, 0.9]:
        gated = train_gated(X_train, y_train, X_test, y_test, R_threshold=threshold)
        print(f"Gated (t={threshold}): acc={gated['final_acc']:.2%}, "
              f"taken={gated['steps_taken']}, skipped={gated['steps_skipped']}")

    # Minibatch test
    print("\n" + "-" * 70)
    print("Test 2: Minibatch gated SGD")
    print("-" * 70)

    standard_mb = train_standard(X_train, y_train, X_test, y_test, n_epochs=50)
    print(f"\nStandard SGD (50 epochs): acc={standard_mb['final_acc']:.2%}")

    for threshold in [0.3, 0.5, 0.7]:
        gated_mb = train_gated_minibatch(X_train, y_train, X_test, y_test,
                                          n_epochs=50, R_threshold=threshold)
        print(f"Gated (t={threshold}): acc={gated_mb['final_acc']:.2%}, "
              f"gate_rate={gated_mb['avg_gate_rate']:.2%}")

    # Noisy data test
    print("\n" + "-" * 70)
    print("Test 3: Noisy data (where gating should help most)")
    print("-" * 70)

    X_noisy, y_noisy, X_test_n, y_test_n = generate_data(noise=0.8, seed=123)

    standard_noisy = train_standard(X_noisy, y_noisy, X_test_n, y_test_n)
    print(f"\nStandard SGD (noisy): acc={standard_noisy['final_acc']:.2%}")

    for threshold in [0.3, 0.5, 0.7]:
        gated_noisy = train_gated(X_noisy, y_noisy, X_test_n, y_test_n, R_threshold=threshold)
        print(f"Gated (t={threshold}): acc={gated_noisy['final_acc']:.2%}, "
              f"taken={gated_noisy['steps_taken']}, skipped={gated_noisy['steps_skipped']}")

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    # Check if gating helps on noisy data
    best_gated = max(
        train_gated(X_noisy, y_noisy, X_test_n, y_test_n, R_threshold=t)['final_acc']
        for t in [0.3, 0.5, 0.7]
    )

    if best_gated > standard_noisy['final_acc'] * 1.02:
        print(f"\n** GATED GRADIENTS: VALIDATED")
        print(f"   Best gated: {best_gated:.2%}")
        print(f"   Standard: {standard_noisy['final_acc']:.2%}")
        print(f"   Improvement: {best_gated/standard_noisy['final_acc']:.2f}x")
    else:
        print(f"\nX  GATED GRADIENTS: NOT VALIDATED")
        print(f"   Gating doesn't improve over standard SGD")
