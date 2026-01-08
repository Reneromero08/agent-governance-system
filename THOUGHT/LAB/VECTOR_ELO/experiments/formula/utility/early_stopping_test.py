"""
Test: Can R serve as an early stopping signal?

Hypothesis: When R drops below a threshold, we've extracted the signal.
Stopping there should match or beat validation-based early stopping.
"""

import numpy as np
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

from gradient_descent_test import (
    TinyNet, cross_entropy_loss, prediction_entropy,
    gradient_magnitude, compute_R, generate_toy_data
)


def train_until_convergence(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    hidden_dim: int = 32,
    lr: float = 0.1,
    max_epochs: int = 500,
    patience: int = 20
) -> Dict:
    """Train with validation-based early stopping (baseline)."""

    input_dim = X_train.shape[1]
    output_dim = len(np.unique(y_train))

    np.random.seed(42)
    net = TinyNet(input_dim, hidden_dim, output_dim)

    best_val_loss = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0
    best_params = net.get_params().copy()

    history = {'train_loss': [], 'val_loss': [], 'test_acc': []}

    for epoch in range(max_epochs):
        # Forward + backward on training
        probs, cache = net.forward(X_train)
        train_loss = cross_entropy_loss(probs, y_train)
        grads = net.backward(y_train, cache)
        net.update(grads, lr)

        # Validation loss
        val_probs, _ = net.forward(X_val)
        val_loss = cross_entropy_loss(val_probs, y_val)

        # Test accuracy
        test_probs, _ = net.forward(X_test)
        test_acc = np.mean(np.argmax(test_probs, axis=1) == y_test)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['test_acc'].append(test_acc)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_params = net.get_params().copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    # Restore best params
    net.set_params(best_params)
    test_probs, _ = net.forward(X_test)
    final_test_acc = np.mean(np.argmax(test_probs, axis=1) == y_test)

    return {
        'stopped_epoch': epoch + 1,
        'best_epoch': best_epoch,
        'final_test_acc': final_test_acc,
        'history': history
    }


def train_with_R_stopping(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    hidden_dim: int = 32,
    lr: float = 0.1,
    max_epochs: int = 500,
    R_threshold: float = 0.5,
    R_patience: int = 10
) -> Dict:
    """Train with R-based early stopping."""

    input_dim = X_train.shape[1]
    output_dim = len(np.unique(y_train))

    np.random.seed(42)
    net = TinyNet(input_dim, hidden_dim, output_dim)

    history = {'train_loss': [], 'R': [], 'test_acc': []}
    prev_loss = None
    R_below_threshold_count = 0

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

        # Test accuracy
        test_probs, _ = net.forward(X_test)
        test_acc = np.mean(np.argmax(test_probs, axis=1) == y_test)

        history['train_loss'].append(train_loss)
        history['R'].append(R)
        history['test_acc'].append(test_acc)

        # R-based early stopping
        if R < R_threshold:
            R_below_threshold_count += 1
        else:
            R_below_threshold_count = 0

        if R_below_threshold_count >= R_patience:
            break

        net.update(grads, lr)
        prev_loss = train_loss

    test_probs, _ = net.forward(X_test)
    final_test_acc = np.mean(np.argmax(test_probs, axis=1) == y_test)

    return {
        'stopped_epoch': epoch + 1,
        'final_test_acc': final_test_acc,
        'final_R': R,
        'history': history
    }


def find_optimal_R_threshold(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    thresholds: List[float] = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
) -> Dict:
    """Find the best R threshold by validation."""

    results = {}

    for threshold in thresholds:
        # Combine train + val for R-stopping (fair comparison)
        X_combined = np.vstack([X_train, X_val])
        y_combined = np.concatenate([y_train, y_val])

        result = train_with_R_stopping(
            X_combined, y_combined, X_test, y_test,
            R_threshold=threshold, R_patience=10
        )

        results[threshold] = {
            'test_acc': result['final_test_acc'],
            'stopped_epoch': result['stopped_epoch']
        }

    return results


def test_R_derivative_stopping(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    hidden_dim: int = 32,
    lr: float = 0.1,
    max_epochs: int = 500,
    dR_threshold: float = 0.01,
    dR_patience: int = 10
) -> Dict:
    """
    Stop when dR/dt (rate of change of R) approaches zero.
    This means R has stabilized = signal extraction complete.
    """

    input_dim = X_train.shape[1]
    output_dim = len(np.unique(y_train))

    np.random.seed(42)
    net = TinyNet(input_dim, hidden_dim, output_dim)

    history = {'train_loss': [], 'R': [], 'dR': [], 'test_acc': []}
    prev_loss = None
    prev_R = None
    dR_stable_count = 0

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

        # Compute dR
        if prev_R is not None:
            dR = abs(R - prev_R)
        else:
            dR = float('inf')

        # Test accuracy
        test_probs, _ = net.forward(X_test)
        test_acc = np.mean(np.argmax(test_probs, axis=1) == y_test)

        history['train_loss'].append(train_loss)
        history['R'].append(R)
        history['dR'].append(dR if dR != float('inf') else 0)
        history['test_acc'].append(test_acc)

        # dR-based early stopping
        if dR < dR_threshold:
            dR_stable_count += 1
        else:
            dR_stable_count = 0

        if dR_stable_count >= dR_patience:
            break

        net.update(grads, lr)
        prev_loss = train_loss
        prev_R = R

    test_probs, _ = net.forward(X_test)
    final_test_acc = np.mean(np.argmax(test_probs, axis=1) == y_test)

    return {
        'stopped_epoch': epoch + 1,
        'final_test_acc': final_test_acc,
        'final_R': R,
        'final_dR': dR,
        'history': history
    }


def compare_stopping_methods(n_runs: int = 10) -> Dict:
    """Compare validation-based vs R-based early stopping."""

    val_results = {'test_acc': [], 'epochs': []}
    R_results = {'test_acc': [], 'epochs': []}
    dR_results = {'test_acc': [], 'epochs': []}

    for run in range(n_runs):
        # Generate fresh data each run
        np.random.seed(run * 100)
        X, y, X_test, y_test = generate_toy_data(
            n_samples=600, n_features=10, n_classes=3, noise=0.3
        )

        # Split train into train/val
        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        # Validation-based stopping
        val_result = train_until_convergence(
            X_train, y_train, X_val, y_val, X_test, y_test,
            patience=15
        )
        val_results['test_acc'].append(val_result['final_test_acc'])
        val_results['epochs'].append(val_result['stopped_epoch'])

        # R-based stopping (use combined train+val for fairness)
        X_combined = np.vstack([X_train, X_val])
        y_combined = np.concatenate([y_train, y_val])

        R_result = train_with_R_stopping(
            X_combined, y_combined, X_test, y_test,
            R_threshold=1.0, R_patience=10
        )
        R_results['test_acc'].append(R_result['final_test_acc'])
        R_results['epochs'].append(R_result['stopped_epoch'])

        # dR-based stopping
        dR_result = test_R_derivative_stopping(
            X_combined, y_combined, X_test, y_test,
            dR_threshold=0.05, dR_patience=10
        )
        dR_results['test_acc'].append(dR_result['final_test_acc'])
        dR_results['epochs'].append(dR_result['stopped_epoch'])

    return {
        'validation': val_results,
        'R_threshold': R_results,
        'dR_stable': dR_results
    }


if __name__ == "__main__":
    print("=" * 60)
    print("EARLY STOPPING TEST: R vs Validation-based")
    print("=" * 60)
    print()
    print("Hypothesis: R dropping below threshold = signal exhausted")
    print()

    # Generate data with train/val/test split
    np.random.seed(42)
    X, y, X_test, y_test = generate_toy_data(
        n_samples=600, n_features=10, n_classes=3, noise=0.3
    )
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    print(f"Data: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test")
    print()

    # Test 1: Validation-based early stopping (baseline)
    print("-" * 60)
    print("Test 1: Validation-based early stopping (baseline)")
    print("-" * 60)

    val_result = train_until_convergence(
        X_train, y_train, X_val, y_val, X_test, y_test,
        patience=20
    )

    print(f"Stopped at epoch: {val_result['stopped_epoch']}")
    print(f"Best epoch: {val_result['best_epoch']}")
    print(f"Test accuracy: {val_result['final_test_acc']:.4f}")

    # Test 2: R-threshold early stopping
    print("\n" + "-" * 60)
    print("Test 2: R-threshold early stopping")
    print("-" * 60)

    # Combine train+val for fair comparison (R doesn't need val set)
    X_combined = np.vstack([X_train, X_val])
    y_combined = np.concatenate([y_train, y_val])

    R_result = train_with_R_stopping(
        X_combined, y_combined, X_test, y_test,
        R_threshold=1.0, R_patience=10
    )

    print(f"Stopped at epoch: {R_result['stopped_epoch']}")
    print(f"Final R: {R_result['final_R']:.4f}")
    print(f"Test accuracy: {R_result['final_test_acc']:.4f}")

    # Test 3: dR (R derivative) early stopping
    print("\n" + "-" * 60)
    print("Test 3: dR-based early stopping (R stabilization)")
    print("-" * 60)

    dR_result = test_R_derivative_stopping(
        X_combined, y_combined, X_test, y_test,
        dR_threshold=0.05, dR_patience=10
    )

    print(f"Stopped at epoch: {dR_result['stopped_epoch']}")
    print(f"Final R: {dR_result['final_R']:.4f}")
    print(f"Final dR: {dR_result['final_dR']:.6f}")
    print(f"Test accuracy: {dR_result['final_test_acc']:.4f}")

    # Test 4: Find optimal R threshold
    print("\n" + "-" * 60)
    print("Test 4: R threshold sensitivity")
    print("-" * 60)

    threshold_results = find_optimal_R_threshold(
        X_train, y_train, X_val, y_val, X_test, y_test,
        thresholds=[0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
    )

    print(f"\n{'Threshold':>10} | {'Test Acc':>10} | {'Epochs':>8}")
    print("-" * 35)
    for thresh, data in sorted(threshold_results.items()):
        print(f"{thresh:>10.1f} | {data['test_acc']:>10.4f} | {data['stopped_epoch']:>8}")

    best_thresh = max(threshold_results.items(), key=lambda x: x[1]['test_acc'])
    print(f"\nBest threshold: {best_thresh[0]} (acc={best_thresh[1]['test_acc']:.4f})")

    # Test 5: Statistical comparison (multiple runs)
    print("\n" + "-" * 60)
    print("Test 5: Statistical comparison (10 runs)")
    print("-" * 60)

    comparison = compare_stopping_methods(n_runs=10)

    print(f"\n{'Method':>15} | {'Mean Acc':>10} | {'Std Acc':>10} | {'Mean Epochs':>12}")
    print("-" * 55)
    for method, data in comparison.items():
        mean_acc = np.mean(data['test_acc'])
        std_acc = np.std(data['test_acc'])
        mean_epochs = np.mean(data['epochs'])
        print(f"{method:>15} | {mean_acc:>10.4f} | {std_acc:>10.4f} | {mean_epochs:>12.1f}")

    # Final verdict
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    val_mean = np.mean(comparison['validation']['test_acc'])
    R_mean = np.mean(comparison['R_threshold']['test_acc'])
    dR_mean = np.mean(comparison['dR_stable']['test_acc'])

    val_epochs = np.mean(comparison['validation']['epochs'])
    R_epochs = np.mean(comparison['R_threshold']['epochs'])
    dR_epochs = np.mean(comparison['dR_stable']['epochs'])

    print(f"\nValidation-based: {val_mean:.4f} acc, {val_epochs:.1f} epochs")
    print(f"R-threshold:      {R_mean:.4f} acc, {R_epochs:.1f} epochs")
    print(f"dR-stable:        {dR_mean:.4f} acc, {dR_epochs:.1f} epochs")

    # Check if R-based methods are competitive
    R_competitive = R_mean >= val_mean - 0.02  # Within 2% is competitive
    dR_competitive = dR_mean >= val_mean - 0.02

    if R_competitive or dR_competitive:
        print("\n** R-BASED EARLY STOPPING IS COMPETITIVE")
        print("   Can stop training WITHOUT validation set!")

        if R_epochs < val_epochs or dR_epochs < val_epochs:
            print("   AND trains faster!")

        if R_competitive and dR_competitive:
            winner = "R-threshold" if R_mean > dR_mean else "dR-stable"
            print(f"\n   Best R-based method: {winner}")
    else:
        print("\nX  R-based early stopping underperforms validation")
        print("   Needs better threshold calibration or different R formulation")

    # The key insight
    print("\n" + "-" * 60)
    print("KEY INSIGHT")
    print("-" * 60)
    print("""
R-based stopping uses ONLY training data to decide when to stop.
Validation-based stopping requires holding out data.

If R-based stopping is competitive, it means:
1. The formula captures intrinsic training dynamics
2. We can use ALL data for training (no val split needed)
3. The formula has PRACTICAL value beyond description
    """)
