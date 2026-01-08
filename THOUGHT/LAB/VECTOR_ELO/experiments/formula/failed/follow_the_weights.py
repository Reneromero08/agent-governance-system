"""
Follow the Weights: Use R to NAVIGATE weight space

The formula works for gradient descent (r = 0.96).
Now: use R to actually GUIDE the optimization, not just measure it.

R = (E / nabla_S) * sigma^Df

Use R to:
1. Decide which direction to step
2. Modulate step size
3. Navigate toward high-R regions (high signal, low noise)
"""

import numpy as np
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')


class TinyNet:
    """Simple 2-layer network."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, seed: int = None):
        if seed is not None:
            np.random.seed(seed)
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(output_dim)

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, dict]:
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ self.W2 + self.b2
        exp_z2 = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
        probs = exp_z2 / np.sum(exp_z2, axis=1, keepdims=True)
        return probs, {'X': X, 'z1': z1, 'a1': a1, 'probs': probs}

    def backward(self, y: np.ndarray, cache: dict) -> dict:
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

    def get_params(self) -> List[np.ndarray]:
        return [self.W1, self.b1, self.W2, self.b2]

    def set_params(self, params: List[np.ndarray]):
        self.W1, self.b1, self.W2, self.b2 = params

    def update(self, grads: dict, lr: float):
        self.W1 -= lr * grads['W1']
        self.b1 -= lr * grads['b1']
        self.W2 -= lr * grads['W2']
        self.b2 -= lr * grads['b2']


def cross_entropy_loss(probs: np.ndarray, y: np.ndarray) -> float:
    m = y.shape[0]
    return -np.mean(np.log(probs[np.arange(m), y] + 1e-10))


def prediction_entropy(probs: np.ndarray) -> float:
    probs_safe = np.clip(probs, 1e-10, 1.0)
    return np.mean(-np.sum(probs_safe * np.log(probs_safe), axis=1))


def compute_R_for_weights(
    net: TinyNet,
    X: np.ndarray,
    y: np.ndarray,
    prev_loss: float = None
) -> Tuple[float, dict]:
    """
    Compute R for current weight configuration.

    E = signal (loss improvement or gradient coherence)
    nabla_H = entropy gradient (how fast entropy is changing)
    Df = 5 - H (calibrated)
    """
    probs, cache = net.forward(X)
    loss = cross_entropy_loss(probs, y)
    grads = net.backward(y, cache)

    # H = prediction entropy
    H = prediction_entropy(probs)
    H = max(H, 0.1)
    H = min(H, 4.9)

    # E = signal strength
    if prev_loss is not None:
        E = max(prev_loss - loss, 0.001)  # Loss improvement
    else:
        E = 0.1

    # nabla_H = gradient magnitude (proxy for entropy gradient)
    grad_mag = np.sqrt(sum(np.sum(g**2) for g in grads.values()))
    nabla_H = max(grad_mag, 0.001)

    # Df from calibration
    Df = max(5 - H, 0.1)

    # R
    sigma = np.e
    R = (E / nabla_H) * (sigma ** Df)

    return R, {
        'loss': loss,
        'H': H,
        'E': E,
        'nabla_H': nabla_H,
        'Df': Df,
        'grads': grads,
        'probs': probs
    }


def probe_directions(
    net: TinyNet,
    X: np.ndarray,
    y: np.ndarray,
    grads: dict,
    prev_loss: float,
    n_directions: int = 8,
    step_size: float = 0.01
) -> Tuple[int, List[float]]:
    """
    Probe multiple directions in weight space, find highest R.

    Directions:
    - Negative gradient (standard SGD)
    - Positive gradient (anti-gradient)
    - Random perturbations
    - Gradient + random noise
    """
    original_params = [p.copy() for p in net.get_params()]
    R_scores = []
    directions = []

    # Direction 0: negative gradient (standard)
    directions.append({k: -v for k, v in grads.items()})

    # Direction 1: positive gradient (anti)
    directions.append({k: v for k, v in grads.items()})

    # Directions 2-7: random + noise combinations
    np.random.seed(42)
    for i in range(n_directions - 2):
        noise = {k: np.random.randn(*v.shape) * 0.1 for k, v in grads.items()}
        if i % 2 == 0:
            # Gradient + noise
            directions.append({k: -grads[k] + noise[k] for k in grads})
        else:
            # Pure random
            directions.append(noise)

    # Probe each direction
    for d in directions:
        # Take small step in direction d
        net.W1 = original_params[0] + step_size * d['W1']
        net.b1 = original_params[1] + step_size * d['b1']
        net.W2 = original_params[2] + step_size * d['W2']
        net.b2 = original_params[3] + step_size * d['b2']

        R, _ = compute_R_for_weights(net, X, y, prev_loss)
        R_scores.append(R)

    # Restore
    net.set_params(original_params)

    best_dir = np.argmax(R_scores)
    return best_dir, R_scores, directions


def train_standard_sgd(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    lr: float = 0.1, n_epochs: int = 100, seed: int = 42
) -> Dict:
    """Standard SGD baseline."""
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


def train_R_guided(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    lr: float = 0.1, n_epochs: int = 100, seed: int = 42
) -> Dict:
    """
    R-guided training: Use R to choose direction.
    Follow the weights toward high-R regions.
    """
    net = TinyNet(X_train.shape[1], 32, len(np.unique(y_train)), seed=seed)
    history = {'loss': [], 'acc': [], 'R': [], 'best_dir': []}
    prev_loss = None

    for epoch in range(n_epochs):
        R, info = compute_R_for_weights(net, X_train, y_train, prev_loss)
        grads = info['grads']
        loss = info['loss']

        # Probe directions, find highest R
        best_dir, R_scores, directions = probe_directions(
            net, X_train, y_train, grads, loss
        )

        # Step in best direction
        d = directions[best_dir]
        net.W1 -= lr * d['W1']
        net.b1 -= lr * d['b1']
        net.W2 -= lr * d['W2']
        net.b2 -= lr * d['b2']

        test_probs, _ = net.forward(X_test)
        acc = np.mean(np.argmax(test_probs, axis=1) == y_test)

        history['loss'].append(loss)
        history['acc'].append(acc)
        history['R'].append(R)
        history['best_dir'].append(best_dir)
        prev_loss = loss

    return {'final_loss': history['loss'][-1], 'final_acc': history['acc'][-1], 'history': history}


def train_R_adaptive_lr(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    base_lr: float = 0.1, n_epochs: int = 100, seed: int = 42
) -> Dict:
    """
    R-adaptive LR: Use R to modulate step size.
    High R = confident, bigger step
    Low R = uncertain, smaller step
    """
    net = TinyNet(X_train.shape[1], 32, len(np.unique(y_train)), seed=seed)
    history = {'loss': [], 'acc': [], 'R': [], 'lr': []}
    prev_loss = None

    # Track R for normalization
    R_history = []

    for epoch in range(n_epochs):
        R, info = compute_R_for_weights(net, X_train, y_train, prev_loss)
        grads = info['grads']
        loss = info['loss']

        R_history.append(R)

        # Adaptive LR based on R
        if len(R_history) > 5:
            R_mean = np.mean(R_history[-10:])
            R_ratio = R / (R_mean + 1e-10)
            adaptive_lr = base_lr * np.clip(R_ratio, 0.1, 3.0)
        else:
            adaptive_lr = base_lr

        net.update(grads, adaptive_lr)

        test_probs, _ = net.forward(X_test)
        acc = np.mean(np.argmax(test_probs, axis=1) == y_test)

        history['loss'].append(loss)
        history['acc'].append(acc)
        history['R'].append(R)
        history['lr'].append(adaptive_lr)
        prev_loss = loss

    return {'final_loss': history['loss'][-1], 'final_acc': history['acc'][-1], 'history': history}


def generate_data(n_samples: int = 500, n_features: int = 10, n_classes: int = 3, seed: int = 42):
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
    print("FOLLOW THE WEIGHTS: R-guided optimization")
    print("=" * 60)
    print()

    X_train, y_train, X_test, y_test = generate_data()
    print(f"Data: {len(X_train)} train, {len(X_test)} test")
    print()

    # Run all methods
    print("-" * 60)
    print("Training...")
    print("-" * 60)

    sgd = train_standard_sgd(X_train, y_train, X_test, y_test, lr=0.1, n_epochs=100)
    print(f"Standard SGD:   loss={sgd['final_loss']:.4f}, acc={sgd['final_acc']:.4f}")

    r_guided = train_R_guided(X_train, y_train, X_test, y_test, lr=0.1, n_epochs=100)
    print(f"R-guided dir:   loss={r_guided['final_loss']:.4f}, acc={r_guided['final_acc']:.4f}")

    r_adaptive = train_R_adaptive_lr(X_train, y_train, X_test, y_test, base_lr=0.1, n_epochs=100)
    print(f"R-adaptive LR:  loss={r_adaptive['final_loss']:.4f}, acc={r_adaptive['final_acc']:.4f}")

    # Analysis
    print("\n" + "-" * 60)
    print("Analysis: Direction choices in R-guided")
    print("-" * 60)

    dir_counts = {}
    for d in r_guided['history']['best_dir']:
        dir_counts[d] = dir_counts.get(d, 0) + 1

    print("\nDirection usage:")
    print("  0 = negative gradient (standard SGD)")
    print("  1 = positive gradient (anti)")
    print("  2+ = random/noise")
    for d, count in sorted(dir_counts.items()):
        pct = count / len(r_guided['history']['best_dir']) * 100
        print(f"  Direction {d}: {count} times ({pct:.1f}%)")

    # Verdict
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    sgd_acc = sgd['final_acc']
    guided_acc = r_guided['final_acc']
    adaptive_acc = r_adaptive['final_acc']

    print(f"\nStandard SGD:  {sgd_acc:.4f}")
    print(f"R-guided:      {guided_acc:.4f} ({'+' if guided_acc > sgd_acc else ''}{(guided_acc - sgd_acc)*100:.2f}%)")
    print(f"R-adaptive LR: {adaptive_acc:.4f} ({'+' if adaptive_acc > sgd_acc else ''}{(adaptive_acc - sgd_acc)*100:.2f}%)")

    if guided_acc > sgd_acc or adaptive_acc > sgd_acc:
        print("\n** R-GUIDED OPTIMIZATION HELPS")
        if guided_acc > sgd_acc:
            print("   Direction selection improves over standard SGD")
        if adaptive_acc > sgd_acc:
            print("   Adaptive LR improves over standard SGD")
    else:
        print("\nX  R-GUIDED OPTIMIZATION DOESN'T BEAT SGD")
        print("   Standard gradient descent is already optimal here")
