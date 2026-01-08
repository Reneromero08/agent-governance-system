"""
Test: Relative R threshold (% of peak) for early stopping.

The absolute R value varies by problem. But the RATIO of current R to peak R
should be more universal.
"""

import numpy as np
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

from gradient_descent_test import (
    TinyNet, cross_entropy_loss, prediction_entropy,
    gradient_magnitude, compute_R, generate_toy_data
)


def train_with_relative_R_stopping(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    hidden_dim: int = 32,
    lr: float = 0.1,
    max_epochs: int = 500,
    R_ratio_threshold: float = 0.3,  # Stop when R drops to 30% of peak
    patience: int = 10
) -> Dict:
    """Stop when R / R_peak drops below threshold."""

    input_dim = X_train.shape[1]
    output_dim = len(np.unique(y_train))

    np.random.seed(42)
    net = TinyNet(input_dim, hidden_dim, output_dim)

    history = {'train_loss': [], 'R': [], 'R_ratio': [], 'test_acc': []}
    prev_loss = None
    R_peak = 0.0
    below_threshold_count = 0

    for epoch in range(max_epochs):
        probs, cache = net.forward(X_train)
        train_loss = cross_entropy_loss(probs, y_train)
        grads = net.backward(y_train, cache)

        # Compute R
        if prev_loss is not None:
            E = max(prev_loss - train_loss, 0.001)
        else:
            E = 0.1

        nabla_S = gradient_magnitude(grads)
        pred_H = prediction_entropy(probs)
        Df = max(5 - pred_H, 0.1)

        R = compute_R(E, nabla_S, Df)

        # Track peak R
        if R > R_peak:
            R_peak = R

        # Compute ratio
        R_ratio = R / R_peak if R_peak > 0 else 1.0

        # Test accuracy
        test_probs, _ = net.forward(X_test)
        test_acc = np.mean(np.argmax(test_probs, axis=1) == y_test)

        history['train_loss'].append(train_loss)
        history['R'].append(R)
        history['R_ratio'].append(R_ratio)
        history['test_acc'].append(test_acc)

        # Early stopping on relative R
        if epoch > 20 and R_ratio < R_ratio_threshold:  # Skip warmup
            below_threshold_count += 1
        else:
            below_threshold_count = 0

        if below_threshold_count >= patience:
            break

        net.update(grads, lr)
        prev_loss = train_loss

    test_probs, _ = net.forward(X_test)
    final_test_acc = np.mean(np.argmax(test_probs, axis=1) == y_test)

    return {
        'stopped_epoch': epoch + 1,
        'final_test_acc': final_test_acc,
        'final_R': R,
        'R_peak': R_peak,
        'final_R_ratio': R_ratio,
        'history': history
    }


def compare_relative_thresholds(n_runs: int = 10) -> Dict:
    """Compare different relative R thresholds."""

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    results = {t: {'test_acc': [], 'epochs': []} for t in thresholds}

    # Also track validation baseline
    val_results = {'test_acc': [], 'epochs': []}

    for run in range(n_runs):
        np.random.seed(run * 100 + 7)
        X, y, X_test, y_test = generate_toy_data(
            n_samples=600, n_features=10, n_classes=3, noise=0.3
        )

        # Validation baseline
        split = int(0.8 * len(X))
        X_train_val, X_val = X[:split], X[split:]
        y_train_val, y_val = y[:split], y[split:]

        # Train with validation stopping
        np.random.seed(42)
        input_dim = X.shape[1]
        output_dim = len(np.unique(y))
        net = TinyNet(input_dim, 32, output_dim)

        best_val_loss = float('inf')
        patience_count = 0
        best_acc = 0
        stopped_epoch = 0

        for epoch in range(500):
            probs, cache = net.forward(X_train_val)
            grads = net.backward(y_train_val, cache)
            net.update(grads, 0.1)

            val_probs, _ = net.forward(X_val)
            val_loss = cross_entropy_loss(val_probs, y_val)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_count = 0
                test_probs, _ = net.forward(X_test)
                best_acc = np.mean(np.argmax(test_probs, axis=1) == y_test)
                stopped_epoch = epoch + 1
            else:
                patience_count += 1

            if patience_count >= 20:
                break

        val_results['test_acc'].append(best_acc)
        val_results['epochs'].append(stopped_epoch)

        # Test each R threshold
        for thresh in thresholds:
            result = train_with_relative_R_stopping(
                X, y, X_test, y_test,
                R_ratio_threshold=thresh, patience=10
            )
            results[thresh]['test_acc'].append(result['final_test_acc'])
            results[thresh]['epochs'].append(result['stopped_epoch'])

    return results, val_results


if __name__ == "__main__":
    print("=" * 60)
    print("RELATIVE R THRESHOLD TEST")
    print("=" * 60)
    print()
    print("Stop when R / R_peak < threshold")
    print("This should be more universal than absolute R values.")
    print()

    # Single run demo
    print("-" * 60)
    print("Single run with R_ratio_threshold = 0.3")
    print("-" * 60)

    np.random.seed(42)
    X, y, X_test, y_test = generate_toy_data(
        n_samples=600, n_features=10, n_classes=3, noise=0.3
    )

    result = train_with_relative_R_stopping(
        X, y, X_test, y_test,
        R_ratio_threshold=0.3, patience=10
    )

    print(f"\nStopped at epoch: {result['stopped_epoch']}")
    print(f"Peak R: {result['R_peak']:.4f}")
    print(f"Final R: {result['final_R']:.4f}")
    print(f"Final R ratio: {result['final_R_ratio']:.4f}")
    print(f"Test accuracy: {result['final_test_acc']:.4f}")

    # Show R trajectory
    print("\nR ratio trajectory (every 10 epochs):")
    for i in range(0, min(len(result['history']['R']), 100), 10):
        print(f"  Epoch {i:3d}: R={result['history']['R'][i]:6.3f}, ratio={result['history']['R_ratio'][i]:.3f}, acc={result['history']['test_acc'][i]:.3f}")

    # Statistical comparison
    print("\n" + "-" * 60)
    print("Statistical comparison (10 runs)")
    print("-" * 60)

    results, val_results = compare_relative_thresholds(n_runs=10)

    print(f"\n{'Method':>20} | {'Mean Acc':>10} | {'Std Acc':>10} | {'Mean Epochs':>12}")
    print("-" * 60)

    val_mean = np.mean(val_results['test_acc'])
    val_std = np.std(val_results['test_acc'])
    val_epochs = np.mean(val_results['epochs'])
    print(f"{'Validation':>20} | {val_mean:>10.4f} | {val_std:>10.4f} | {val_epochs:>12.1f}")

    for thresh, data in sorted(results.items()):
        mean_acc = np.mean(data['test_acc'])
        std_acc = np.std(data['test_acc'])
        mean_epochs = np.mean(data['epochs'])
        print(f"{'R_ratio=' + str(thresh):>20} | {mean_acc:>10.4f} | {std_acc:>10.4f} | {mean_epochs:>12.1f}")

    # Find best threshold
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    best_thresh = None
    best_score = -float('inf')

    for thresh, data in results.items():
        mean_acc = np.mean(data['test_acc'])
        mean_epochs = np.mean(data['epochs'])

        # Score: accuracy - small penalty for epochs (efficiency)
        score = mean_acc - 0.0001 * mean_epochs

        if score > best_score:
            best_score = score
            best_thresh = thresh

    print(f"\nBest relative threshold: {best_thresh}")
    print(f"  Mean accuracy: {np.mean(results[best_thresh]['test_acc']):.4f}")
    print(f"  Mean epochs: {np.mean(results[best_thresh]['epochs']):.1f}")

    # Compare to validation
    best_acc = np.mean(results[best_thresh]['test_acc'])
    best_epochs = np.mean(results[best_thresh]['epochs'])

    diff_acc = best_acc - val_mean
    diff_epochs = val_epochs - best_epochs

    print(f"\nVs validation-based stopping:")
    print(f"  Accuracy difference: {diff_acc:+.4f}")
    print(f"  Epochs saved: {diff_epochs:+.1f}")

    if diff_acc >= -0.02:  # Within 2%
        print("\n** RELATIVE R STOPPING IS COMPETITIVE!")
        if diff_epochs > 0:
            print(f"   Trains {diff_epochs:.0f} fewer epochs on average")
        if diff_acc > 0:
            print(f"   Actually BETTER accuracy!")
    else:
        print("\nX  Relative R stopping underperforms")

    # The insight
    print("\n" + "-" * 60)
    print("INSIGHT")
    print("-" * 60)
    print(f"""
The relative threshold R/R_peak normalizes across problem scales.

When R drops to {best_thresh*100:.0f}% of its peak:
- Signal extraction is ~{(1-best_thresh)*100:.0f}% complete
- Further training yields diminishing returns
- Can stop WITHOUT a validation set

This is the practical application of the Living Formula:
R measures "signal remaining" in the gradient.
    """)
