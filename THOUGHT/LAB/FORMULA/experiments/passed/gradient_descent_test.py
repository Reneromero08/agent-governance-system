"""
Test: Does the Living Formula predict gradient descent behavior?

Hypothesis: R = (E / nabla_S) * sigma^Df can serve as a quality metric
for gradient steps, predicting when to trust the gradient vs when
we're in chaotic regions.

Mapping:
- E = signal strength (loss reduction per step, or gradient coherence)
- nabla_S = entropy gradient (gradient magnitude, or prediction entropy change)
- Df = landscape "sharpness" (estimated from Hessian trace or loss variance)
- sigma = base (use e or 2)
"""

import numpy as np
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

# Simple neural network for testing
class TinyNet:
    """2-layer neural network for gradient descent testing."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        # Xavier initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(output_dim)

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Forward pass with cached activations."""
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(0, z1)  # ReLU
        z2 = a1 @ self.W2 + self.b2
        # Softmax
        exp_z2 = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
        probs = exp_z2 / np.sum(exp_z2, axis=1, keepdims=True)

        cache = {'X': X, 'z1': z1, 'a1': a1, 'z2': z2, 'probs': probs}
        return probs, cache

    def backward(self, y: np.ndarray, cache: dict) -> dict:
        """Backward pass returning gradients."""
        m = y.shape[0]
        probs = cache['probs']

        # Output layer gradient
        dz2 = probs.copy()
        dz2[np.arange(m), y] -= 1
        dz2 /= m

        dW2 = cache['a1'].T @ dz2
        db2 = np.sum(dz2, axis=0)

        # Hidden layer gradient
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (cache['z1'] > 0)  # ReLU derivative

        dW1 = cache['X'].T @ dz1
        db1 = np.sum(dz1, axis=0)

        return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}

    def get_params(self) -> np.ndarray:
        """Flatten all parameters into a single vector."""
        return np.concatenate([self.W1.flatten(), self.b1, self.W2.flatten(), self.b2])

    def set_params(self, params: np.ndarray):
        """Set parameters from flattened vector."""
        idx = 0
        W1_size = self.W1.size
        self.W1 = params[idx:idx+W1_size].reshape(self.W1.shape)
        idx += W1_size

        b1_size = self.b1.size
        self.b1 = params[idx:idx+b1_size]
        idx += b1_size

        W2_size = self.W2.size
        self.W2 = params[idx:idx+W2_size].reshape(self.W2.shape)
        idx += W2_size

        self.b2 = params[idx:]

    def update(self, grads: dict, lr: float):
        """Apply gradient update."""
        self.W1 -= lr * grads['W1']
        self.b1 -= lr * grads['b1']
        self.W2 -= lr * grads['W2']
        self.b2 -= lr * grads['b2']


def cross_entropy_loss(probs: np.ndarray, y: np.ndarray) -> float:
    """Cross-entropy loss."""
    m = y.shape[0]
    log_probs = -np.log(probs[np.arange(m), y] + 1e-10)
    return np.mean(log_probs)


def prediction_entropy(probs: np.ndarray) -> float:
    """Shannon entropy of predictions (uncertainty)."""
    # Avoid log(0)
    probs_safe = np.clip(probs, 1e-10, 1.0)
    entropy = -np.sum(probs_safe * np.log(probs_safe), axis=1)
    return np.mean(entropy)


def gradient_magnitude(grads: dict) -> float:
    """L2 norm of all gradients."""
    total = 0.0
    for key, grad in grads.items():
        total += np.sum(grad ** 2)
    return np.sqrt(total)


def gradient_coherence(grads: dict, prev_grads: dict) -> float:
    """Cosine similarity between consecutive gradients."""
    if prev_grads is None:
        return 1.0

    curr = np.concatenate([g.flatten() for g in grads.values()])
    prev = np.concatenate([g.flatten() for g in prev_grads.values()])

    norm_curr = np.linalg.norm(curr)
    norm_prev = np.linalg.norm(prev)

    if norm_curr < 1e-10 or norm_prev < 1e-10:
        return 0.0

    return np.dot(curr, prev) / (norm_curr * norm_prev)


def estimate_sharpness(net: TinyNet, X: np.ndarray, y: np.ndarray,
                       epsilon: float = 0.01, n_samples: int = 5) -> float:
    """
    Estimate loss landscape sharpness by perturbing parameters.
    Higher values = sharper minima (worse generalization).
    """
    original_params = net.get_params().copy()
    original_probs, _ = net.forward(X)
    original_loss = cross_entropy_loss(original_probs, y)

    losses = []
    for _ in range(n_samples):
        # Random perturbation
        perturbation = np.random.randn(len(original_params)) * epsilon
        net.set_params(original_params + perturbation)

        probs, _ = net.forward(X)
        loss = cross_entropy_loss(probs, y)
        losses.append(loss)

    # Restore original
    net.set_params(original_params)

    # Sharpness = how much loss increases with perturbation
    return np.std(losses) / (original_loss + 1e-10)


def compute_R(E: float, nabla_S: float, Df: float, sigma: float = np.e) -> float:
    """
    Compute the Living Formula: R = (E / nabla_S) * sigma^Df

    With calibration from text domain:
    - E mapped to signal strength
    - nabla_S mapped to gradient magnitude
    - Df mapped to sharpness (will test if 5-H relationship holds)
    """
    if nabla_S < 1e-10:
        nabla_S = 1e-10  # Avoid division by zero

    R = (E / nabla_S) * (sigma ** Df)
    return R


def run_gradient_descent_with_R(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    hidden_dim: int = 32,
    lr: float = 0.1,
    n_epochs: int = 100,
    sigma: float = np.e
) -> Dict:
    """
    Run gradient descent while tracking R at each step.
    """
    input_dim = X_train.shape[1]
    output_dim = len(np.unique(y_train))

    net = TinyNet(input_dim, hidden_dim, output_dim)

    history = {
        'loss': [],
        'test_loss': [],
        'accuracy': [],
        'R': [],
        'E': [],
        'nabla_S': [],
        'Df': [],
        'pred_entropy': [],
        'grad_coherence': [],
        'loss_delta': []
    }

    prev_grads = None
    prev_loss = None

    for epoch in range(n_epochs):
        # Forward pass
        probs, cache = net.forward(X_train)
        loss = cross_entropy_loss(probs, y_train)

        # Test metrics
        test_probs, _ = net.forward(X_test)
        test_loss = cross_entropy_loss(test_probs, y_test)
        test_preds = np.argmax(test_probs, axis=1)
        accuracy = np.mean(test_preds == y_test)

        # Backward pass
        grads = net.backward(y_train, cache)

        # === Compute R components ===

        # E (signal strength): loss reduction or gradient coherence
        if prev_loss is not None:
            loss_delta = prev_loss - loss
            E = max(loss_delta, 0.001)  # Signal = improvement
        else:
            loss_delta = 0
            E = 0.1  # Initial estimate

        # nabla_S (entropy gradient): gradient magnitude
        nabla_S = gradient_magnitude(grads)

        # Prediction entropy
        pred_H = prediction_entropy(probs)

        # Df (sharpness): estimated from loss landscape
        # Using calibration: Df = 5 - H (from text domain)
        # But also estimate directly from sharpness
        Df_entropy = max(5 - pred_H, 0.1)  # From entropy calibration
        Df_sharp = estimate_sharpness(net, X_train, y_train) * 5  # Scale to similar range

        # Use entropy-based Df (from our calibration)
        Df = Df_entropy

        # Gradient coherence
        coherence = gradient_coherence(grads, prev_grads)

        # === Compute R ===
        R = compute_R(E, nabla_S, Df, sigma)

        # Store history
        history['loss'].append(loss)
        history['test_loss'].append(test_loss)
        history['accuracy'].append(accuracy)
        history['R'].append(R)
        history['E'].append(E)
        history['nabla_S'].append(nabla_S)
        history['Df'].append(Df)
        history['pred_entropy'].append(pred_H)
        history['grad_coherence'].append(coherence)
        history['loss_delta'].append(loss_delta)

        # Update
        net.update(grads, lr)
        prev_grads = {k: v.copy() for k, v in grads.items()}
        prev_loss = loss

    return history


def generate_toy_data(n_samples: int = 500, n_features: int = 10,
                      n_classes: int = 3, noise: float = 0.1) -> Tuple:
    """Generate a toy classification dataset."""
    np.random.seed(42)

    X = np.random.randn(n_samples, n_features)

    # Create class centers
    centers = np.random.randn(n_classes, n_features) * 2

    # Assign labels based on distance to centers
    distances = np.array([np.sum((X - c) ** 2, axis=1) for c in centers])
    y = np.argmin(distances, axis=0)

    # Add noise
    X += np.random.randn(*X.shape) * noise

    # Split
    split = int(0.8 * n_samples)
    return X[:split], y[:split], X[split:], y[split:]


def analyze_R_predictions(history: Dict) -> Dict:
    """Analyze how well R predicts training behavior."""

    results = {}

    # 1. Does R correlate with loss improvement?
    R = np.array(history['R'])
    loss_delta = np.array(history['loss_delta'])

    # Skip first point (no delta)
    R_valid = R[1:]
    delta_valid = loss_delta[1:]

    if len(R_valid) > 2:
        corr_loss = np.corrcoef(R_valid, delta_valid)[0, 1]
        results['R_vs_loss_improvement'] = corr_loss
    else:
        results['R_vs_loss_improvement'] = 0

    # 2. Does high R predict good generalization?
    accuracy = np.array(history['accuracy'])
    corr_acc = np.corrcoef(R, accuracy)[0, 1]
    results['R_vs_accuracy'] = corr_acc

    # 3. Does R stabilize at convergence?
    # Compare R variance in first half vs second half
    mid = len(R) // 2
    R_early_var = np.var(R[:mid])
    R_late_var = np.var(R[mid:])
    results['R_stabilization'] = R_early_var / (R_late_var + 1e-10)

    # 4. Does nabla_S decrease as we converge?
    nabla_S = np.array(history['nabla_S'])
    gradient_decay = nabla_S[0] / (nabla_S[-1] + 1e-10)
    results['gradient_decay'] = gradient_decay

    # 5. Does Df change predictably?
    Df = np.array(history['Df'])
    Df_trend = np.polyfit(np.arange(len(Df)), Df, 1)[0]
    results['Df_trend'] = Df_trend

    return results


def test_adaptive_lr_with_R(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    hidden_dim: int = 32,
    base_lr: float = 0.1,
    n_epochs: int = 100
) -> Tuple[Dict, Dict]:
    """
    Test if using R to adapt learning rate improves training.

    Hypothesis: lr(t) = base_lr * f(R) should outperform constant lr
    """
    input_dim = X_train.shape[1]
    output_dim = len(np.unique(y_train))

    # Fixed LR baseline
    np.random.seed(123)
    net_fixed = TinyNet(input_dim, hidden_dim, output_dim)
    history_fixed = {'loss': [], 'accuracy': []}

    for epoch in range(n_epochs):
        probs, cache = net_fixed.forward(X_train)
        loss = cross_entropy_loss(probs, y_train)
        grads = net_fixed.backward(y_train, cache)
        net_fixed.update(grads, base_lr)

        test_probs, _ = net_fixed.forward(X_test)
        test_preds = np.argmax(test_probs, axis=1)
        accuracy = np.mean(test_preds == y_test)

        history_fixed['loss'].append(loss)
        history_fixed['accuracy'].append(accuracy)

    # Adaptive LR using R
    np.random.seed(123)  # Same init
    net_adaptive = TinyNet(input_dim, hidden_dim, output_dim)
    history_adaptive = {'loss': [], 'accuracy': [], 'lr': [], 'R': []}

    prev_loss = None

    for epoch in range(n_epochs):
        probs, cache = net_adaptive.forward(X_train)
        loss = cross_entropy_loss(probs, y_train)
        grads = net_adaptive.backward(y_train, cache)

        # Compute R
        if prev_loss is not None:
            E = max(prev_loss - loss, 0.001)
        else:
            E = 0.1

        nabla_S = gradient_magnitude(grads)
        pred_H = prediction_entropy(probs)
        Df = max(5 - pred_H, 0.1)

        R = compute_R(E, nabla_S, Df)

        # Adaptive learning rate: scale by normalized R
        # High R = stable, can use higher lr
        # Low R = chaotic, use lower lr
        R_normalized = np.clip(R / (R + 1), 0.1, 2.0)  # Sigmoid-like normalization
        adaptive_lr = base_lr * R_normalized

        net_adaptive.update(grads, adaptive_lr)

        test_probs, _ = net_adaptive.forward(X_test)
        test_preds = np.argmax(test_probs, axis=1)
        accuracy = np.mean(test_preds == y_test)

        history_adaptive['loss'].append(loss)
        history_adaptive['accuracy'].append(accuracy)
        history_adaptive['lr'].append(adaptive_lr)
        history_adaptive['R'].append(R)

        prev_loss = loss

    return history_fixed, history_adaptive


if __name__ == "__main__":
    print("=" * 60)
    print("GRADIENT DESCENT + LIVING FORMULA TEST")
    print("=" * 60)
    print()
    print("Hypothesis: R = (E / nabla_S) * sigma^Df predicts training quality")
    print()

    # Generate data
    X_train, y_train, X_test, y_test = generate_toy_data(
        n_samples=500, n_features=10, n_classes=3, noise=0.2
    )

    print(f"Data: {X_train.shape[0]} train, {X_test.shape[0]} test, {X_train.shape[1]} features")
    print()

    # Run standard training with R tracking
    print("-" * 60)
    print("Phase 1: Track R during standard gradient descent")
    print("-" * 60)

    history = run_gradient_descent_with_R(
        X_train, y_train, X_test, y_test,
        hidden_dim=32, lr=0.1, n_epochs=100
    )

    print(f"\nFinal loss: {history['loss'][-1]:.4f}")
    print(f"Final accuracy: {history['accuracy'][-1]:.4f}")
    print(f"Final R: {history['R'][-1]:.4f}")

    # Analyze R predictions
    print("\n" + "-" * 60)
    print("Phase 2: Analyze R predictive power")
    print("-" * 60)

    analysis = analyze_R_predictions(history)

    print(f"\nR vs loss improvement correlation: {analysis['R_vs_loss_improvement']:.4f}")
    print(f"R vs accuracy correlation: {analysis['R_vs_accuracy']:.4f}")
    print(f"R stabilization ratio (early/late var): {analysis['R_stabilization']:.4f}")
    print(f"Gradient decay (start/end): {analysis['gradient_decay']:.4f}")
    print(f"Df trend (slope): {analysis['Df_trend']:.6f}")

    # Interpret results
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    if analysis['R_vs_loss_improvement'] > 0.3:
        print("\n** R PREDICTS loss improvement (r > 0.3)")
    elif analysis['R_vs_loss_improvement'] > 0:
        print("\n*  R weakly predicts loss improvement")
    else:
        print("\nX  R does NOT predict loss improvement")

    if analysis['R_vs_accuracy'] > 0.5:
        print("** R PREDICTS generalization (r > 0.5)")
    elif analysis['R_vs_accuracy'] > 0:
        print("*  R weakly predicts generalization")
    else:
        print("X  R does NOT predict generalization")

    if analysis['R_stabilization'] > 2:
        print("** R STABILIZES at convergence (ratio > 2)")
    else:
        print("*  R variance constant through training")

    # Test adaptive learning rate
    print("\n" + "-" * 60)
    print("Phase 3: Test R-adaptive learning rate")
    print("-" * 60)

    hist_fixed, hist_adaptive = test_adaptive_lr_with_R(
        X_train, y_train, X_test, y_test,
        hidden_dim=32, base_lr=0.1, n_epochs=100
    )

    print(f"\nFixed LR final accuracy: {hist_fixed['accuracy'][-1]:.4f}")
    print(f"R-adaptive LR final accuracy: {hist_adaptive['accuracy'][-1]:.4f}")
    print(f"\nFixed LR final loss: {hist_fixed['loss'][-1]:.4f}")
    print(f"R-adaptive LR final loss: {hist_adaptive['loss'][-1]:.4f}")

    # Compare convergence speed (epochs to 90% of final accuracy)
    final_acc_fixed = hist_fixed['accuracy'][-1]
    final_acc_adaptive = hist_adaptive['accuracy'][-1]

    target_fixed = 0.9 * final_acc_fixed
    target_adaptive = 0.9 * final_acc_adaptive

    epochs_to_90_fixed = next((i for i, a in enumerate(hist_fixed['accuracy']) if a >= target_fixed), 100)
    epochs_to_90_adaptive = next((i for i, a in enumerate(hist_adaptive['accuracy']) if a >= target_adaptive), 100)

    print(f"\nEpochs to 90% final accuracy:")
    print(f"  Fixed LR: {epochs_to_90_fixed}")
    print(f"  R-adaptive LR: {epochs_to_90_adaptive}")

    # Final verdict
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    improvements = 0

    if analysis['R_vs_loss_improvement'] > 0.2:
        improvements += 1
    if analysis['R_vs_accuracy'] > 0.3:
        improvements += 1
    if hist_adaptive['accuracy'][-1] > hist_fixed['accuracy'][-1]:
        improvements += 1
    if epochs_to_90_adaptive < epochs_to_90_fixed:
        improvements += 1

    print(f"\nScore: {improvements}/4 metrics show R is useful")

    if improvements >= 3:
        print("\n** VALIDATED: R formula helps gradient descent!")
        print("   The intuition was RIGHT - this has practical use.")
    elif improvements >= 2:
        print("\n*  PARTIAL: R shows promise but needs refinement")
    else:
        print("\nX  NOT VALIDATED: R doesn't help gradient descent")
        print("   The formula may be descriptive only.")

    # Show R trajectory
    print("\n" + "-" * 60)
    print("R trajectory (first 10 epochs):")
    print("-" * 60)
    for i in range(min(10, len(history['R']))):
        print(f"  Epoch {i:2d}: R={history['R'][i]:8.4f}, loss={history['loss'][i]:.4f}, acc={history['accuracy'][i]:.4f}")
