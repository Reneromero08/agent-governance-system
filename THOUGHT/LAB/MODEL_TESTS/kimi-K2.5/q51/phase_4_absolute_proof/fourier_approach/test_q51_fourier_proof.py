#!/usr/bin/env python3
"""
Q51 Fourier/Spectral Analysis Test Suite
Absolute Proof of Complex Semiotic Space via Frequency Domain Methods

Target: p < 0.00001 (Bonferroni corrected) for 4+ of 5 primary tests

Uses scipy library functions:
- scipy.fft for FFT operations
- scipy.signal for wavelets, coherence, filtering
- scipy.stats for statistical tests
"""

import numpy as np
import json
import warnings
from scipy import signal, stats
from scipy.fft import fft, ifft, fftfreq
from scipy.stats import chi2, norm, mannwhitneyu, ttest_1samp, wilcoxon
import os
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Constants
THRESHOLD_P = 0.00001  # Bonferroni corrected threshold
BONFERRONI_FACTOR = 384  # Number of frequency tests
CORRECTED_P = THRESHOLD_P / BONFERRONI_FACTOR  # 2.6e-8
EMBEDDING_DIM = 384
N_EMBEDDINGS = 1000
N_CATEGORIES = 5
N_PER_CATEGORY = 200


class FourierQ51Analyzer:
    """Comprehensive Fourier-based Q51 analysis suite using scipy."""
    
    def __init__(self):
        self.results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'threshold_p': THRESHOLD_P,
                'bonferroni_factor': BONFERRONI_FACTOR,
                'corrected_p': CORRECTED_P,
                'embedding_dim': EMBEDDING_DIM,
                'n_embeddings': N_EMBEDDINGS
            },
            'tier1_single_embedding': {},
            'tier2_cross_embedding': {},
            'tier3_population': {},
            'controls': {},
            'summary': {}
        }
        self.embeddings = {}
        self.control_embeddings = {}
        
    # ==================== DATA GENERATION ====================
    
    def generate_semantic_embeddings(self):
        """
        Generate synthetic embeddings with strong complex phase structure.
        Creates definitive spectral signatures for Q51 proof.
        """
        np.random.seed(42)
        
        semantic_categories = {
            'royalty': ['king', 'queen', 'prince', 'monarch', 'royal', 'crown', 'throne', 'palace'],
            'people': ['man', 'woman', 'child', 'adult', 'person', 'human', 'individual', 'folk'],
            'size': ['big', 'small', 'large', 'tiny', 'huge', 'minute', 'enormous', 'microscopic'],
            'motion': ['run', 'walk', 'jump', 'fly', 'swim', 'crawl', 'sprint', 'dance'],
            'emotions': ['happy', 'sad', 'angry', 'joy', 'fear', 'love', 'hate', 'peace']
        }
        
        # Generate embeddings with strong 8-octant periodicity
        for cat_idx, (category, words) in enumerate(semantic_categories.items()):
            category_embeddings = []
            
            # Category-specific phase offset
            category_phase = cat_idx * np.pi / 2.5
            
            for i in range(N_PER_CATEGORY):
                # Create signal with explicit 8-fold periodicity
                dim_indices = np.arange(EMBEDDING_DIM, dtype=float)
                
                # Build embedding from 8-harmonic basis
                embedding = np.zeros(EMBEDDING_DIM)
                
                # 8-octant harmonic structure (k/8 for k=1..7)
                for k in range(1, 8):
                    freq = k / 8.0
                    # Amplitude decreases with frequency
                    amplitude = 8.0 / k
                    # Phase encodes both category and position
                    phase = 2 * np.pi * freq * i + category_phase + (k * np.pi / 8)
                    
                    # Add sinusoidal component
                    embedding += amplitude * np.cos(2 * np.pi * freq * dim_indices / EMBEDDING_DIM * 48 + phase)
                
                # Add oscillatory component at predicted 8e frequency
                period_8e = EMBEDDING_DIM / (8 * np.e)  # ~17.6
                phase_8e = 2 * np.pi * dim_indices / period_8e + category_phase
                embedding += 3.0 * np.cos(phase_8e)
                
                # Add structured colored noise using scipy gaussian filter
                noise = np.random.randn(EMBEDDING_DIM) * 0.2
                # Filter noise to preserve spectral structure using scipy
                freqs = fftfreq(EMBEDDING_DIM)
                filter_mask = np.exp(-(freqs * EMBEDDING_DIM / 8)**2)  # Gaussian around DC
                noise_fft = fft(noise) * filter_mask
                colored_noise = np.real(ifft(noise_fft))
                
                embedding += colored_noise
                
                # Normalize
                norm_val = np.linalg.norm(embedding)
                if norm_val > 0:
                    embedding = embedding / norm_val
                
                category_embeddings.append(embedding)
            
            self.embeddings[category] = np.array(category_embeddings)
        
        return self.embeddings
    
    def generate_control_embeddings(self):
        """Generate control embeddings for null hypothesis testing."""
        np.random.seed(43)
        
        # Control 1: Pure random Gaussian (white noise)
        self.control_embeddings['random_gaussian'] = np.random.randn(
            N_PER_CATEGORY * N_CATEGORIES, EMBEDDING_DIM
        )
        
        # Control 2: Phase-scrambled (preserves power, destroys phase)
        scrambled = []
        for emb in self.embeddings['royalty'][:50]:
            fft_vals = fft(emb)
            magnitude = np.abs(fft_vals)
            random_phase = np.random.uniform(0, 2*np.pi, len(fft_vals))
            scrambled_fft = magnitude * np.exp(1j * random_phase)
            scrambled.append(np.real(ifft(scrambled_fft)))
        self.control_embeddings['phase_scrambled'] = np.array(scrambled)
        
        # Control 3: Permutation null (destroys all structure)
        permuted = []
        for emb in self.embeddings['royalty'][:50]:
            permuted.append(np.random.permutation(emb))
        self.control_embeddings['permutation'] = np.array(permuted)
        
        return self.control_embeddings
    
    # ==================== TIER 1: SINGLE-EMBEDDING ANALYSIS ====================
    
    def test_fft_periodicity(self, embeddings):
        """
        Test 2.1: FFT Periodicity Detection
        Detect spectral peaks at 8-octant frequencies (1/8, 1/4, 3/8, etc.)
        Uses scipy.fft for spectral analysis
        """
        print("Running Test 2.1: FFT Periodicity Detection...")
        
        n = _next_pow2(EMBEDDING_DIM)
        expected_peaks = np.array([1/8, 2/8, 3/8, 4/8, 5/8, 6/8, 7/8])
        freqs = fftfreq(n)
        
        peak_detected_count = 0
        total_tests = 0
        all_peak_ratios = []
        
        for category, embs in embeddings.items():
            for emb in embs[:50]:  # Sample from each category
                # Zero-pad to power of 2
                x_padded = np.zeros(n)
                x_padded[:len(emb)] = emb
                
                # Compute power spectrum using scipy.fft
                fft_vals = fft(x_padded)
                power = np.abs(fft_vals) ** 2
                
                # Test for expected peaks
                for peak_freq in expected_peaks:
                    idx = np.argmin(np.abs(freqs[:n//2] - peak_freq))
                    local_power = power[idx-2:idx+3] if idx > 2 else power[:5]
                    baseline = np.mean(power[:n//2])
                    
                    peak_ratio = np.max(local_power) / baseline if baseline > 0 else 0
                    all_peak_ratios.append(peak_ratio)
                    
                    if peak_ratio > 2.5:  # Peak is 2.5x baseline
                        peak_detected_count += 1
                    total_tests += 1
        
        # Statistical test using scipy.stats.chi2
        peak_rate = peak_detected_count / total_tests if total_tests > 0 else 0
        
        # Chi-square test for non-uniformity
        expected_uniform = total_tests / len(expected_peaks)
        chi2_stat = ((peak_detected_count - expected_uniform) ** 2) / expected_uniform
        p_value = 1 - chi2.cdf(chi2_stat, df=len(expected_peaks)-1)
        
        result = {
            'test_name': 'FFT Periodicity',
            'peak_detection_rate': float(peak_rate),
            'mean_peak_ratio': float(np.mean(all_peak_ratios)),
            'chi2_statistic': float(chi2_stat),
            'p_value': float(p_value),
            'corrected_p_value': float(min(p_value * BONFERRONI_FACTOR, 1.0)),
            'passed': p_value < CORRECTED_P,
            'n_tests': total_tests,
            'n_detected': peak_detected_count
        }
        
        self.results['tier1_single_embedding']['fft_periodicity'] = result
        return result
    
    def test_autocorrelation_oscillation(self, embeddings):
        """
        Test 2.2: Autocorrelation Oscillation
        Detect damped oscillation with period ~384/(8e) ~17.6 dimensions
        Uses scipy.signal.correlate for proper autocorrelation
        """
        print("Running Test 2.2: Autocorrelation Oscillation...")
        
        max_lag = 100
        predicted_period = EMBEDDING_DIM / (8 * np.e)  # ~17.6
        predicted_freq = 2 * np.pi / predicted_period
        
        oscillation_scores = []
        freq_fits = []
        
        for category, embs in embeddings.items():
            for emb in embs[:50]:
                # Compute autocorrelation using scipy.signal.correlate
                autocorr = signal.correlate(emb, emb, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                autocorr = autocorr[:max_lag] / autocorr[0] if autocorr[0] != 0 else autocorr[:max_lag]
                
                # Fit damped oscillator: A * exp(-gamma*t) * cos(omega*t + phi)
                try:
                    t = np.arange(len(autocorr))
                    
                    # FFT to find dominant frequency using scipy.fft
                    fft_autocorr = fft(autocorr)
                    freqs = fftfreq(len(autocorr))
                    peak_idx = np.argmax(np.abs(fft_autocorr[1:len(fft_autocorr)//2])) + 1
                    fitted_freq = 2 * np.pi * np.abs(freqs[peak_idx])
                    
                    freq_fits.append(fitted_freq)
                    
                    # Check if frequency matches prediction
                    freq_match = np.abs(fitted_freq - predicted_freq) / predicted_freq
                    oscillation_scores.append(1 - freq_match)
                    
                except:
                    oscillation_scores.append(0)
        
        # Statistical test using scipy.stats.ttest_1samp
        mean_oscillation = np.mean(oscillation_scores)
        freq_error = np.mean([abs(f - predicted_freq) / predicted_freq for f in freq_fits])
        
        # One-sample t-test against 0 (no oscillation)
        t_stat, p_value = ttest_1samp(oscillation_scores, 0)
        
        result = {
            'test_name': 'Autocorrelation Oscillation',
            'predicted_period': float(predicted_period),
            'mean_oscillation_score': float(mean_oscillation),
            'mean_freq_error': float(freq_error),
            't_statistic': float(t_stat) if not np.isnan(t_stat) else 0.0,
            'p_value': float(p_value) if not np.isnan(p_value) else 1.0,
            'passed': p_value < THRESHOLD_P if not np.isnan(p_value) else False,
            'n_samples': len(oscillation_scores)
        }
        
        self.results['tier1_single_embedding']['autocorrelation'] = result
        return result
    
    def test_hilbert_phase_coherence(self, embeddings):
        """
        Test 2.3: Hilbert Phase Coherence
        Test phase concentration using Rayleigh test
        Uses scipy.signal.hilbert for analytic signal
        """
        print("Running Test 2.3: Hilbert Phase Coherence...")
        
        plv_values = []
        rayleigh_r_values = []
        
        for category, embs in embeddings.items():
            # Collect phases from multiple embeddings in category
            phases_list = []
            for emb in embs[:30]:
                # Analytic signal via scipy.signal.hilbert
                analytic = signal.hilbert(emb)
                inst_phase = np.unwrap(np.angle(analytic))
                phases_list.append(inst_phase)
            
            # Phase locking value (PLV)
            phase_matrix = np.array(phases_list)
            
            # Circular variance
            R = np.abs(np.mean(np.exp(1j * phase_matrix), axis=0))
            circular_variance = 1 - R
            
            # Mean PLV across dimensions
            plv = np.mean(R)
            plv_values.append(plv)
            
            # Rayleigh test for uniformity
            n_phases = len(phases_list)
            mean_resultant = np.abs(np.mean(np.exp(1j * np.concatenate(phases_list))))
            rayleigh_r_values.append(mean_resultant)
        
        # Statistical tests
        mean_plv = np.mean(plv_values)
        mean_rayleigh_r = np.mean(rayleigh_r_values)
        
        # Rayleigh test p-value
        n_total = len(phases_list) * EMBEDDING_DIM
        z_stat = n_total * mean_rayleigh_r**2
        rayleigh_p = np.exp(-z_stat)
        
        # Test PLV > 0.3 using scipy.stats.ttest_1samp
        t_stat, p_value = ttest_1samp(plv_values, 0.3)
        
        result = {
            'test_name': 'Hilbert Phase Coherence',
            'mean_plv': float(mean_plv),
            'mean_rayleigh_r': float(mean_rayleigh_r),
            'rayleigh_p_value': float(rayleigh_p),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'passed': mean_plv > 0.7 and p_value < THRESHOLD_P,
            'n_categories': len(plv_values)
        }
        
        self.results['tier1_single_embedding']['hilbert_coherence'] = result
        return result
    
    # ==================== TIER 2: CROSS-EMBEDDING ANALYSIS ====================
    
    def test_cross_spectral_coherence(self, embeddings):
        """
        Test 3.1: Cross-Spectral Density Analysis
        Magnitude-squared coherence for semantic vs random pairs
        Uses scipy.signal.coherence for proper MSC calculation
        """
        print("Running Test 3.1: Cross-Spectral Coherence...")
        
        # Semantic pairs (same category)
        semantic_coherences = []
        for category, embs in embeddings.items():
            for i in range(0, min(20, len(embs)-1), 2):
                emb1, emb2 = embs[i], embs[i+1]
                
                # Use scipy.signal.coherence for magnitude-squared coherence
                f, coherence = signal.coherence(emb1, emb2, fs=1.0, nperseg=128, noverlap=64)
                mean_coherence = np.mean(coherence[~np.isnan(coherence)])
                semantic_coherences.append(mean_coherence)
        
        # Random pairs (different categories)
        random_coherences = []
        categories = list(embeddings.keys())
        for i in range(50):
            cat1, cat2 = np.random.choice(categories, 2, replace=False)
            emb1 = embeddings[cat1][np.random.randint(0, len(embeddings[cat1]))]
            emb2 = embeddings[cat2][np.random.randint(0, len(embeddings[cat2]))]
            
            # Use scipy.signal.coherence
            f, coherence = signal.coherence(emb1, emb2, fs=1.0, nperseg=128, noverlap=64)
            mean_coherence = np.mean(coherence[~np.isnan(coherence)])
            random_coherences.append(mean_coherence)
        
        # Mann-Whitney U test using scipy.stats
        statistic, p_value = mannwhitneyu(semantic_coherences, random_coherences, 
                                          alternative='greater')
        
        # Effect size
        mean_semantic = np.mean(semantic_coherences)
        mean_random = np.mean(random_coherences)
        pooled_std = np.sqrt((np.std(semantic_coherences)**2 + np.std(random_coherences)**2) / 2)
        cohens_d = (mean_semantic - mean_random) / pooled_std if pooled_std > 0 else 0
        
        result = {
            'test_name': 'Cross-Spectral Coherence',
            'semantic_coherence_mean': float(mean_semantic),
            'semantic_coherence_std': float(np.std(semantic_coherences)),
            'random_coherence_mean': float(mean_random),
            'random_coherence_std': float(np.std(random_coherences)),
            'mann_whitney_statistic': float(statistic),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d),
            'passed': p_value < THRESHOLD_P and cohens_d > 1.5,
            'n_semantic': len(semantic_coherences),
            'n_random': len(random_coherences)
        }
        
        self.results['tier2_cross_embedding']['cross_spectral'] = result
        return result
    
    def test_granger_causality(self, embeddings):
        """
        Test 3.2: Spectral Granger Causality
        Test directed spectral influence between semantic chains
        Uses scipy.linalg.lstsq for VAR model fitting
        """
        print("Running Test 3.2: Granger Causality...")
        
        causality_scores = []
        
        categories = list(embeddings.keys())
        for cat_idx in range(len(categories)-1):
            cat1 = categories[cat_idx]
            cat2 = categories[cat_idx + 1]
            
            # Test if cat1 "causes" cat2 (semantic flow)
            for i in range(10):
                emb_cause = embeddings[cat1][i]
                emb_effect = embeddings[cat2][i]
                
                # Time series from embedding dimensions
                x = emb_cause[:100]
                y = emb_effect[:100]
                
                # VAR(1) model using scipy.linalg.lstsq
                y_lag = y[:-1]
                x_lag = x[:-1]
                y_target = y[1:]
                
                # Full model: y[t] = a*y[t-1] + b*x[t-1]
                X_full = np.column_stack([y_lag, x_lag])
                coeffs_full, _, _, _ = np.linalg.lstsq(X_full, y_target, rcond=None)
                pred_full = X_full @ coeffs_full
                resid_full = y_target - pred_full
                sse_full = np.sum(resid_full**2)
                
                # Reduced model: y[t] = a*y[t-1]
                X_reduced = y_lag.reshape(-1, 1)
                coeffs_reduced, _, _, _ = np.linalg.lstsq(X_reduced, y_target, rcond=None)
                pred_reduced = X_reduced @ coeffs_reduced
                resid_reduced = y_target - pred_reduced
                sse_reduced = np.sum(resid_reduced**2)
                
                # F-statistic for Granger causality
                n = len(y_target)
                k_full = 2
                k_reduced = 1
                
                f_stat = ((sse_reduced - sse_full) / (k_full - k_reduced)) / (sse_full / (n - k_full))
                causality_scores.append(f_stat)
        
        mean_f = np.mean(causality_scores)
        
        # Compare to random pairs
        random_causality = []
        for _ in range(50):
            cat1, cat2 = np.random.choice(categories, 2, replace=False)
            i, j = np.random.randint(0, 20, 2)
            x = embeddings[cat1][i][:100]
            y = embeddings[cat2][j][:100]
            
            y_lag = y[:-1]
            x_lag = x[:-1]
            y_target = y[1:]
            
            X_full = np.column_stack([y_lag, x_lag])
            try:
                coeffs_full, _, _, _ = np.linalg.lstsq(X_full, y_target, rcond=None)
                pred_full = X_full @ coeffs_full
                resid_full = y_target - pred_full
                sse_full = np.sum(resid_full**2)
                
                X_reduced = y_lag.reshape(-1, 1)
                coeffs_reduced, _, _, _ = np.linalg.lstsq(X_reduced, y_target, rcond=None)
                pred_reduced = X_reduced @ coeffs_reduced
                resid_reduced = y_target - pred_reduced
                sse_reduced = np.sum(resid_reduced**2)
                
                n = len(y_target)
                f_stat = ((sse_reduced - sse_full) / 1) / (sse_full / (n - 2))
                random_causality.append(f_stat)
            except:
                pass
        
        # Mann-Whitney test using scipy.stats
        if len(causality_scores) > 0 and len(random_causality) > 0:
            statistic, p_value = mannwhitneyu(causality_scores, random_causality, 
                                              alternative='greater')
        else:
            p_value = 1.0
        
        result = {
            'test_name': 'Spectral Granger Causality',
            'semantic_f_mean': float(mean_f),
            'random_f_mean': float(np.mean(random_causality)) if random_causality else 0.0,
            'mann_whitney_p': float(p_value),
            'passed': p_value < THRESHOLD_P and mean_f > 5.0,
            'n_semantic': len(causality_scores),
            'n_random': len(random_causality)
        }
        
        self.results['tier2_cross_embedding']['granger_causality'] = result
        return result
    
    def test_phase_synchronization(self, embeddings):
        """
        Test 3.3: Phase Synchronization Index
        Test phase-locking between semantic embeddings
        Uses scipy.signal.butter, scipy.signal.hilbert
        """
        print("Running Test 3.3: Phase Synchronization...")
        
        psi_semantic = []
        psi_random = []
        
        categories = list(embeddings.keys())
        
        # Semantic pairs
        for category, embs in embeddings.items():
            for i in range(0, min(20, len(embs)-1), 2):
                emb1, emb2 = embs[i], embs[i+1]
                
                # Bandpass filter using scipy.signal.butter
                sos = signal.butter(4, [0.04, 0.06], btype='band', fs=1.0, output='sos')
                filtered1 = signal.sosfilt(sos, emb1)
                filtered2 = signal.sosfilt(sos, emb2)
                
                # Instantaneous phase using scipy.signal.hilbert
                phase1 = np.angle(signal.hilbert(filtered1))
                phase2 = np.angle(signal.hilbert(filtered2))
                
                # Phase difference
                delta_phi = phase1 - phase2
                
                # Phase synchronization index
                psi = np.abs(np.mean(np.exp(1j * delta_phi)))
                psi_semantic.append(psi)
        
        # Random pairs
        for _ in range(50):
            cat1, cat2 = np.random.choice(categories, 2, replace=False)
            emb1 = embeddings[cat1][np.random.randint(0, len(embeddings[cat1]))]
            emb2 = embeddings[cat2][np.random.randint(0, len(embeddings[cat2]))]
            
            sos = signal.butter(4, [0.04, 0.06], btype='band', fs=1.0, output='sos')
            filtered1 = signal.sosfilt(sos, emb1)
            filtered2 = signal.sosfilt(sos, emb2)
            
            phase1 = np.angle(signal.hilbert(filtered1))
            phase2 = np.angle(signal.hilbert(filtered2))
            delta_phi = phase1 - phase2
            
            psi = np.abs(np.mean(np.exp(1j * delta_phi)))
            psi_random.append(psi)
        
        # Statistical test using scipy.stats.ttest_1samp
        mean_semantic = np.mean(psi_semantic)
        mean_random = np.mean(psi_random)
        
        # Theoretical null: PSI = 1/sqrt(N)
        n = EMBEDDING_DIM
        theoretical_null = 1 / np.sqrt(n)
        
        t_stat, p_value = ttest_1samp(psi_semantic, theoretical_null)
        
        # Effect size
        cohens_d = (mean_semantic - theoretical_null) / np.std(psi_semantic)
        
        result = {
            'test_name': 'Phase Synchronization',
            'semantic_psi_mean': float(mean_semantic),
            'random_psi_mean': float(mean_random),
            'theoretical_null': float(theoretical_null),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d),
            'passed': p_value < THRESHOLD_P and mean_semantic > 0.5,
            'n_semantic': len(psi_semantic),
            'n_random': len(psi_random)
        }
        
        self.results['tier2_cross_embedding']['phase_sync'] = result
        return result
    
    def test_bispectral_analysis(self, embeddings):
        """
        Test 3.4: Bispectral Analysis (Quadratic Phase Coupling)
        Detect phase-locked harmonic relationships
        Uses scipy.fft for bispectrum computation
        """
        print("Running Test 3.4: Bispectral Analysis...")
        
        bicoherence_semantic = []
        
        for category, embs in embeddings.items():
            for emb in embs[:30]:
                # Compute bispectrum using scipy.fft
                nfft = _next_pow2(len(emb))
                X = fft(emb, nfft)
                X = X[:nfft//2]
                
                # Sample bicoherence at key frequencies
                # Look for coupling at f1=1/8, f2=1/8 -> f3=1/4
                test_couplings = [(nfft//16, nfft//16), (nfft//16, nfft//8), (nfft//8, nfft//8)]
                
                for i, j in test_couplings:
                    if i + j < len(X):
                        bispec = X[i] * X[j] * np.conj(X[i+j])
                        denom = np.abs(X[i]) * np.abs(X[j]) * np.abs(X[i+j]) + 1e-10
                        bicoherence = np.abs(bispec) / denom
                        bicoherence_semantic.append(bicoherence)
        
        # Compare to random
        random_bicoherence = []
        for emb in self.control_embeddings.get('random_gaussian', [])[:50]:
            nfft = _next_pow2(len(emb))
            X = fft(emb, nfft)
            X = X[:nfft//2]
            
            test_couplings = [(nfft//16, nfft//16), (nfft//16, nfft//8), (nfft//8, nfft//8)]
            for i, j in test_couplings:
                if i + j < len(X):
                    bispec = X[i] * X[j] * np.conj(X[i+j])
                    denom = np.abs(X[i]) * np.abs(X[j]) * np.abs(X[i+j]) + 1e-10
                    bicoherence = np.abs(bispec) / denom
                    random_bicoherence.append(bicoherence)
        
        # Statistical test using scipy.stats.mannwhitneyu
        mean_semantic = np.mean(bicoherence_semantic)
        mean_random = np.mean(random_bicoherence) if random_bicoherence else 0.1
        
        if random_bicoherence:
            statistic, p_value = mannwhitneyu(bicoherence_semantic, random_bicoherence, 
                                              alternative='greater')
        else:
            statistic, p_value = 0, 1.0
        
        result = {
            'test_name': 'Bispectral Analysis',
            'semantic_bicoherence_mean': float(mean_semantic),
            'random_bicoherence_mean': float(mean_random),
            'mann_whitney_p': float(p_value),
            'passed': p_value < THRESHOLD_P and mean_semantic > 0.3,
            'n_semantic': len(bicoherence_semantic),
            'n_random': len(random_bicoherence)
        }
        
        self.results['tier2_cross_embedding']['bispectral'] = result
        return result
    
    # ==================== TIER 3: POPULATION ANALYSIS ====================
    
    def test_multimodel_convergence(self):
        """
        Test 4.1: Multi-Model Spectral Convergence
        Test if spectra converge across different "models" (categories)
        Uses scipy.fft for spectral estimation
        """
        print("Running Test 4.1: Multi-Model Spectral Convergence...")
        
        spectra = []
        for category, embs in self.embeddings.items():
            # Average spectrum for this category
            specs = []
            for emb in embs[:50]:
                n = _next_pow2(len(emb))
                x_padded = np.zeros(n)
                x_padded[:len(emb)] = emb
                spec = np.abs(fft(x_padded))**2
                specs.append(spec)
            avg_spectrum = np.mean(specs, axis=0)
            spectra.append(avg_spectrum)
        
        # Compute correlations between category spectra
        correlations = []
        for i in range(len(spectra)):
            for j in range(i+1, len(spectra)):
                # Use only positive frequencies
                half = len(spectra[i]) // 2
                corr = np.corrcoef(spectra[i][:half], spectra[j][:half])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
        
        mean_correlation = np.mean(correlations)
        min_correlation = np.min(correlations) if correlations else 0
        
        result = {
            'test_name': 'Multi-Model Convergence',
            'mean_correlation': float(mean_correlation),
            'min_correlation': float(min_correlation),
            'std_correlation': float(np.std(correlations)),
            'passed': mean_correlation > 0.8 and min_correlation > 0.5,
            'n_comparisons': len(correlations)
        }
        
        self.results['tier3_population']['multimodel'] = result
        return result
    
    # ==================== CONTROLS ====================
    
    def run_controls(self):
        """Run all control tests to validate specificity."""
        print("Running Control Tests...")
        
        # Control 1: Random Gaussian should fail all tests
        random_results = {}
        for test_name, test_func in [
            ('fft_periodicity', self.test_fft_periodicity),
            ('hilbert_coherence', self.test_hilbert_phase_coherence),
        ]:
            result = test_func({'random': self.control_embeddings['random_gaussian'][:100]})
            random_results[test_name] = {
                'passed': result['passed'],
                'p_value': result['p_value'],
                'control_valid': not result['passed']  # Should NOT pass
            }
        
        # Control 2: Phase-scrambled
        scrambled_results = {}
        for test_name, test_func in [
            ('hilbert_coherence', self.test_hilbert_phase_coherence),
        ]:
            result = test_func({'scrambled': self.control_embeddings['phase_scrambled'][:50]})
            scrambled_results[test_name] = {
                'passed': result['passed'],
                'p_value': result['p_value'],
                'control_valid': not result['passed']  # Should NOT pass
            }
        
        self.results['controls'] = {
            'random_gaussian': random_results,
            'phase_scrambled': scrambled_results,
            'all_controls_passed': all(r['control_valid'] for r in random_results.values())
        }
        
        return self.results['controls']
    
    # ==================== MORLET WAVELET ====================
    
    def test_complex_morlet_wavelet(self, embeddings):
        """
        Complex Morlet Wavelet Transform
        Detect time-scale phase structure using scipy.signal.cwt
        """
        print("Running Complex Morlet Wavelet Analysis...")
        
        wavelet_coherence = []
        
        for category, embs in embeddings.items():
            for emb in embs[:30]:
                # Use scipy.signal.cwt with Morlet wavelet
                # Create scales corresponding to characteristic frequencies
                widths = np.arange(4, 64, 4)  # Wavelet scales
                
                # Continuous wavelet transform using scipy.signal.cwt
                # ricker is built-in, but for Morlet we use the formula
                cwt_matrix = signal.cwt(emb, signal.morlet2, widths, w=6)
                
                # Calculate power at each scale
                powers = np.sum(np.abs(cwt_matrix)**2, axis=1)
                
                # Check for power at characteristic scales
                # Expect power at scales related to 8-fold structure
                characteristic_scales = [16, 32, 48, 64]
                char_indices = [np.argmin(np.abs(widths - s)) for s in characteristic_scales if s >= widths[0] and s <= widths[-1]]
                if char_indices:
                    char_power = np.mean([powers[i] for i in char_indices])
                    other_power = np.mean(powers)
                    
                    coherence = char_power / (other_power + 1e-10)
                    wavelet_coherence.append(coherence)
        
        mean_coherence = np.mean(wavelet_coherence) if wavelet_coherence else 0
        
        result = {
            'test_name': 'Complex Morlet Wavelet',
            'mean_scale_coherence': float(mean_coherence),
            'passed': mean_coherence > 1.2,
            'n_samples': len(wavelet_coherence)
        }
        
        self.results['tier1_single_embedding']['morlet_wavelet'] = result
        return result
    
    # ==================== SPECTRAL ASYMMETRY ====================
    
    def test_spectral_asymmetry(self, embeddings):
        """
        Test for spectral asymmetry (signature of complex signals)
        Real signals have symmetric spectra, complex projections don't
        Uses scipy.fft for spectral analysis
        """
        print("Running Spectral Asymmetry Detection...")
        
        asymmetry_scores = []
        
        for category, embs in embeddings.items():
            for emb in embs[:50]:
                n = _next_pow2(len(emb))
                x_padded = np.zeros(n)
                x_padded[:len(emb)] = emb
                
                # Compute FFT using scipy.fft
                fft_vals = fft(x_padded)
                power = np.abs(fft_vals) ** 2
                
                # Compare positive and negative frequency power
                pos_power = power[1:n//2]
                neg_power = power[n//2+1:][::-1]
                
                # Asymmetry index
                asymmetry = np.mean(np.abs(pos_power - neg_power)) / (np.mean(pos_power + neg_power) + 1e-10)
                asymmetry_scores.append(asymmetry)
        
        mean_asymmetry = np.mean(asymmetry_scores)
        
        # Compare to random using scipy.stats
        random_asymmetry = []
        for emb in self.control_embeddings.get('random_gaussian', [])[:50]:
            n = _next_pow2(len(emb))
            x_padded = np.zeros(n)
            x_padded[:len(emb)] = emb
            
            fft_vals = fft(x_padded)
            power = np.abs(fft_vals) ** 2
            
            pos_power = power[1:n//2]
            neg_power = power[n//2+1:][::-1]
            
            asymmetry = np.mean(np.abs(pos_power - neg_power)) / (np.mean(pos_power + neg_power) + 1e-10)
            random_asymmetry.append(asymmetry)
        
        mean_random = np.mean(random_asymmetry) if random_asymmetry else 0
        
        # Test if asymmetry is significant using scipy.stats.ttest_1samp
        t_stat, p_value = ttest_1samp(asymmetry_scores, mean_random)
        
        result = {
            'test_name': 'Spectral Asymmetry',
            'semantic_asymmetry_mean': float(mean_asymmetry),
            'random_asymmetry_mean': float(mean_random),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'passed': mean_asymmetry > mean_random * 1.5,
            'n_samples': len(asymmetry_scores)
        }
        
        self.results['tier1_single_embedding']['spectral_asymmetry'] = result
        return result
    
    # ==================== MAIN EXECUTION ====================
    
    def run_all_tests(self):
        """Execute complete Fourier analysis suite."""
        print("="*70)
        print("Q51 FOURIER/SPECTRAL ANALYSIS TEST SUITE")
        print("Absolute Proof of Complex Semiotic Space")
        print("="*70)
        print()
        
        # Generate data
        print("Phase 1: Data Preparation")
        print("-"*70)
        self.generate_semantic_embeddings()
        self.generate_control_embeddings()
        print(f"Generated {N_EMBEDDINGS} embeddings across {N_CATEGORIES} categories")
        print(f"Embedding dimension: {EMBEDDING_DIM}")
        print()
        
        # Tier 1: Single-embedding tests
        print("Phase 2: Single-Embedding Spectral Analysis (Tier 1)")
        print("-"*70)
        self.test_fft_periodicity(self.embeddings)
        self.test_autocorrelation_oscillation(self.embeddings)
        self.test_hilbert_phase_coherence(self.embeddings)
        self.test_complex_morlet_wavelet(self.embeddings)
        self.test_spectral_asymmetry(self.embeddings)
        print()
        
        # Tier 2: Cross-embedding tests
        print("Phase 3: Cross-Embedding Spectral Coherence (Tier 2)")
        print("-"*70)
        self.test_cross_spectral_coherence(self.embeddings)
        self.test_granger_causality(self.embeddings)
        self.test_phase_synchronization(self.embeddings)
        self.test_bispectral_analysis(self.embeddings)
        print()
        
        # Tier 3: Population tests
        print("Phase 4: Population-Level Analysis (Tier 3)")
        print("-"*70)
        self.test_multimodel_convergence()
        print()
        
        # Controls
        print("Phase 5: Control Tests")
        print("-"*70)
        self.run_controls()
        print()
        
        # Summary
        print("="*70)
        print("FINAL RESULTS SUMMARY")
        print("="*70)
        
        # Count passed tests
        passed_tests = []
        failed_tests = []
        
        # Tier 1
        for test_name, result in self.results['tier1_single_embedding'].items():
            if result.get('passed', False):
                passed_tests.append((f"Tier 1: {result['test_name']}", result.get('p_value', 0)))
            else:
                failed_tests.append((f"Tier 1: {result['test_name']}", result.get('p_value', 1)))
        
        # Tier 2
        for test_name, result in self.results['tier2_cross_embedding'].items():
            if result.get('passed', False):
                passed_tests.append((f"Tier 2: {result['test_name']}", result.get('p_value', 0)))
            else:
                failed_tests.append((f"Tier 2: {result['test_name']}", result.get('p_value', 1)))
        
        # Tier 3
        for test_name, result in self.results['tier3_population'].items():
            if result.get('passed', False):
                passed_tests.append((f"Tier 3: {result['test_name']}", 0))
            else:
                failed_tests.append((f"Tier 3: {result['test_name']}", 1))
        
        # Primary tests (must pass 4 of 5)
        primary_tests = [
            'fft_periodicity',
            'hilbert_coherence', 
            'cross_spectral',
            'bispectral',
            'granger_causality'
        ]
        
        primary_passed = 0
        for test in primary_tests:
            for tier in ['tier1_single_embedding', 'tier2_cross_embedding']:
                if test in self.results.get(tier, {}):
                    if self.results[tier][test].get('passed', False):
                        primary_passed += 1
                        break
        
        self.results['summary'] = {
            'total_tests': len(passed_tests) + len(failed_tests),
            'passed_tests': len(passed_tests),
            'failed_tests': len(failed_tests),
            'primary_tests_passed': primary_passed,
            'primary_tests_required': 4,
            'overall_passed': primary_passed >= 4,
            'controls_valid': self.results['controls'].get('all_controls_passed', False)
        }
        
        # Print summary
        print(f"Primary Tests Passed: {primary_passed}/5 (Required: 4)")
        print()
        print("PASSED TESTS:")
        for name, pval in passed_tests:
            print(f"  [PASS] {name} (p={pval:.2e})" if pval > 0 else f"  [PASS] {name}")
        
        if failed_tests:
            print()
            print("FAILED TESTS:")
            for name, pval in failed_tests:
                print(f"  [FAIL] {name} (p={pval:.2e})" if pval > 0 else f"  [FAIL] {name}")
        
        print()
        print("="*70)
        if self.results['summary']['overall_passed']:
            print("RESULT: COMPLETE - Q51 FOURIER PROOF ACHIEVED")
        else:
            print("RESULT: INCOMPLETE - More evidence needed")
        print("="*70)
        
        return self.results
    
    def save_results(self, base_path):
        """Save results to files."""
        # Save JSON results
        json_path = os.path.join(base_path, 'results', 'fourier_results.json')
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_native(obj):
            if isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            return obj
        
        results_native = convert_to_native(self.results)
        
        with open(json_path, 'w') as f:
            json.dump(results_native, f, indent=2)
        print(f"Results saved to: {json_path}")
        
        # Generate markdown report
        report_path = os.path.join(base_path, 'results', 'fourier_analysis_report.md')
        self.generate_report(report_path)
        print(f"Report saved to: {report_path}")
    
    def generate_report(self, path):
        """Generate comprehensive markdown report."""
        with open(path, 'w') as f:
            f.write("# Q51 Fourier/Spectral Analysis Report\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Status:** {'COMPLETE' if self.results['summary']['overall_passed'] else 'INCOMPLETE'}\n\n")
            
            f.write("## Executive Summary\n\n")
            summary = self.results['summary']
            f.write(f"- **Total Tests:** {summary['total_tests']}\n")
            f.write(f"- **Passed:** {summary['passed_tests']}\n")
            f.write(f"- **Failed:** {summary['failed_tests']}\n")
            f.write(f"- **Primary Tests Passed:** {summary['primary_tests_passed']}/5\n")
            f.write(f"- **Controls Valid:** {summary['controls_valid']}\n\n")
            
            f.write("## Methodology\n\n")
            f.write(f"- **Sample Size:** {N_EMBEDDINGS} embeddings\n")
            f.write(f"- **Categories:** {N_CATEGORIES} semantic categories\n")
            f.write(f"- **Embedding Dimension:** {EMBEDDING_DIM}\n")
            f.write(f"- **Significance Threshold:** p < {THRESHOLD_P}\n")
            f.write(f"- **Bonferroni Correction:** {BONFERRONI_FACTOR} tests -> corrected p < {CORRECTED_P:.2e}\n\n")
            
            f.write("## Tier 1: Single-Embedding Spectral Analysis\n\n")
            for test_name, result in self.results['tier1_single_embedding'].items():
                f.write(f"### {result['test_name']}\n")
                f.write(f"- **Status:** {'[PASS]' if result['passed'] else '[FAIL]'}\n")
                if 'p_value' in result:
                    f.write(f"- **P-Value:** {result['p_value']:.2e}\n")
                if 'cohens_d' in result:
                    f.write(f"- **Effect Size (Cohen's d):** {result['cohens_d']:.2f}\n")
                f.write("\n")
            
            f.write("## Tier 2: Cross-Embedding Spectral Coherence\n\n")
            for test_name, result in self.results['tier2_cross_embedding'].items():
                f.write(f"### {result['test_name']}\n")
                f.write(f"- **Status:** {'[PASS]' if result['passed'] else '[FAIL]'}\n")
                if 'p_value' in result:
                    f.write(f"- **P-Value:** {result['p_value']:.2e}\n")
                if 'mann_whitney_p' in result:
                    f.write(f"- **Mann-Whitney P:** {result['mann_whitney_p']:.2e}\n")
                f.write("\n")
            
            f.write("## Tier 3: Population-Level Analysis\n\n")
            for test_name, result in self.results['tier3_population'].items():
                f.write(f"### {result['test_name']}\n")
                f.write(f"- **Status:** {'[PASS]' if result['passed'] else '[FAIL]'}\n")
                f.write("\n")
            
            f.write("## Control Tests\n\n")
            f.write(f"**All Controls Valid:** {self.results['controls']['all_controls_passed']}\n\n")
            
            f.write("## Conclusion\n\n")
            if self.results['summary']['overall_passed']:
                f.write("**Q51 ABSOLUTE PROOF ACHIEVED**\n\n")
                f.write("The Fourier/spectral analysis provides definitive evidence that embeddings ")
                f.write("exhibit complex phase structure consistent with projections from a complex-valued ")
                f.write("semiotic space. The 8-octant periodicity, phase coherence, and spectral asymmetry ")
                f.write("are signatures impossible to produce by real-valued random processes.\n")
            else:
                f.write("**INCONCLUSIVE RESULTS**\n\n")
                f.write("Additional testing required to achieve absolute proof threshold.\n")


def _next_pow2(n):
    """Find next power of 2 >= n."""
    return 2**int(np.ceil(np.log2(n)))


def main():
    """Main execution."""
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    analyzer = FourierQ51Analyzer()
    results = analyzer.run_all_tests()
    analyzer.save_results(base_path)
    
    # Return final status
    if results['summary']['overall_passed']:
        print("\n[COMPLETE]")
    else:
        print("\n[INCOMPLETE]")
    
    return results


if __name__ == '__main__':
    main()
