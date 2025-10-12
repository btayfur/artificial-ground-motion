#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Iterative Spectral Matching Module.
Used to match the response spectrum of an acceleration time series to the target design spectrum.
"""

import numpy as np
from scipy import signal, interpolate
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional, Union

from design_spectrum import TBDYSpectrum
from spectral_analysis import SpectralAnalysis


class SpectralMatcher:
    """
    Class for iterative spectral matching to match the response spectrum of an acceleration time series to the target design spectrum.
    """
    
    def __init__(self):
        """Initialize the class."""
        self.time = None
        self.accel_original = None
        self.accel_matched = None
        self.dt = None
        self.target_periods = None
        self.target_spectrum = None
        self.current_spectrum = None
        self.iteration_results = []
        self.params = {}
    
    def set_motion(self, time: np.ndarray, accel: np.ndarray) -> None:
        """
        Set the acceleration time series.
        
        Args:
            time: Time array
            accel: Acceleration array
        """
        if len(time) != len(accel):
            raise ValueError("Time and acceleration arrays must be of the same length.")
        
        self.time = time
        self.accel_original = accel.copy()
        self.accel_matched = accel.copy()
        self.dt = time[1] - time[0]
        self.params = {
            'dt': self.dt,
            'duration': time[-1],
            'num_points': len(time)
        }
    
    def set_target_spectrum(self, periods: np.ndarray, 
                          spectrum_values: np.ndarray) -> None:
        """
        Set the target design spectrum.
        
        Args:
            periods: Period array
            spectrum_values: Spectral acceleration values
        """
        if len(periods) != len(spectrum_values):
            raise ValueError("Period and spectrum values must be of the same length.")
        
        self.target_periods = periods
        self.target_spectrum = spectrum_values
        
        self.params.update({
            'target_spectrum': {
                'num_periods': len(periods),
                'min_period': min(periods),
                'max_period': max(periods)
            }
        })
    
    def _check_setup(self) -> None:
        """Check if the setup is complete."""
        if self.time is None or self.accel_original is None:
            raise ValueError("You must call the set_motion method first.")
        
        if self.target_periods is None or self.target_spectrum is None:
            raise ValueError("You must call the set_target_spectrum method first.")
    
    def compute_current_spectrum(self, damping: float = 0.05) -> Dict[str, np.ndarray]:
        """
        Compute the response spectrum of the current acceleration time series.
        
        Args:
            damping: Damping ratio
            
        Returns:
            Dictionary containing response spectrum values
        """
        self._check_setup()
        
        # Create spectral analysis object
        analyzer = SpectralAnalysis()
        analyzer.set_motion(self.time, self.accel_matched)
        
        # Apply baseline correction
        analyzer.apply_baseline_correction()
        
        # Compute response spectrum
        response_spectrum = analyzer.compute_response_spectrum(
            periods=self.target_periods,
            damping=damping
        )
        
        self.current_spectrum = response_spectrum
        
        return response_spectrum
    
    def match_spectrum(self, max_iterations: int = 10, 
                     tolerance_avg: float = 0.05, 
                     tolerance_max: float = 0.10, 
                     damping: float = 0.05, 
                     freq_range: Optional[Tuple[float, float]] = None,
                     avg_weight: float = 0.7,
                     max_weight: float = 0.3,
                     max_no_improvement: int = 15,
                     max_ratio: float = 5.0,
                     min_ratio: float = 0.2) -> Dict:
        """
        Perform iterative spectral matching.
        
        Args:
            max_iterations: Maximum number of iterations
            tolerance_avg: Average error tolerance
            tolerance_max: Maximum individual error tolerance
            damping: Damping ratio
            freq_range: Frequency range for matching (min_freq, max_freq)
            avg_weight: Weight for average error metric (between 0 and 1)
            max_weight: Weight for maximum error metric (between 0 and 1)
            max_no_improvement: Maximum number of iterations to wait without improvement
            max_ratio: Maximum correction ratio (to prevent over-correction)
            min_ratio: Minimum correction ratio (to prevent over-correction)
            
        Returns:
            Dictionary containing matching results
        """
        self._check_setup()
        
        # Check error weights
        if not np.isclose(avg_weight + max_weight, 1.0):
            print(f"  WARNING: Weight sum is not 1.0 ({avg_weight + max_weight}), normalizing.")
            total = avg_weight + max_weight
            avg_weight = avg_weight / total
            max_weight = max_weight / total
        
        # Compute initial spectrum
        self.compute_current_spectrum(damping=damping)
        
        # Reset iteration results
        self.iteration_results = []
        
        # Update parameters
        self.params.update({
            'spectral_matching': {
                'max_iterations': max_iterations,
                'tolerance_avg': tolerance_avg,
                'tolerance_max': tolerance_max,
                'damping': damping,
                'freq_range': freq_range,
                'avg_weight': avg_weight,
                'max_weight': max_weight,
                'max_no_improvement': max_no_improvement,
                'max_ratio': max_ratio,
                'min_ratio': min_ratio
            }
        })
        
        # Time series length for FFT
        n = len(self.time)
        
        # Nyquist frequency
        nyquist = 0.5 / self.dt
        
        # Frequency array (Hz)
        frequencies = np.fft.rfftfreq(n, self.dt)
        
        # Frequency values corresponding to the period array (Hz)
        target_freqs = 1.0 / self.target_periods
        
        # Frequency range check
        if freq_range is not None:
            min_freq, max_freq = freq_range
            freq_mask = (target_freqs >= min_freq) & (target_freqs <= max_freq)
        else:
            freq_mask = np.ones_like(target_freqs, dtype=bool)
        
        # Variables to store the best result
        best_avg_error = float('inf')
        best_max_error = float('inf')
        best_accel_matched = None
        best_spectrum = None
        best_iteration = 0
        no_improvement_count = 0
        
        # Iterative spectral matching loop
        for iteration in range(max_iterations):
            # Compute current response spectrum (without baseline correction)
            analyzer = SpectralAnalysis()
            analyzer.set_motion(self.time, self.accel_matched)
            
            # Compute response spectrum
            response_spectrum = analyzer.compute_response_spectrum(
                periods=self.target_periods,
                damping=damping
            )
            
            self.current_spectrum = response_spectrum
            
            # Calculate the ratio between the current response spectrum and the target spectrum
            ratio = np.ones_like(self.target_periods)
            ratio[freq_mask] = self.target_spectrum[freq_mask] / self.current_spectrum['psa'][freq_mask]
            
            # Limit ratios to prevent over-correction
            ratio = np.clip(ratio, min_ratio, max_ratio)
            
            # Apply smoothing (to prevent overly sharp transitions)
            #if iteration > 0:  
                # Apply gradual correction after the first iteration
                # Reduce the weight of previous iterations (slower convergence, more stable result)
            #    smoothing_factor = min(0.5, 1.0 / (iteration + 1))
            #    ratio = 1.0 + smoothing_factor * (ratio - 1.0)
            
            # Interpolation of ratios (based on frequency)
            interpolator = interpolate.interp1d(
                target_freqs, ratio, kind='linear', 
                bounds_error=False, fill_value=1.0
            )
            
            # Calculate ratios for all frequencies
            ratio_interp = interpolator(frequencies)
            
            # Compute FFT
            accel_fft = np.fft.rfft(self.accel_matched)
            
            # Adjust Fourier amplitudes
            accel_fft_adjusted = accel_fft * ratio_interp
            
            # Convert back to time series with inverse FFT
            accel_matched = np.fft.irfft(accel_fft_adjusted, n=n)
            
            # Apply baseline correction to time series
            analyzer = SpectralAnalysis()
            analyzer.set_motion(self.time, accel_matched)
            analyzer.apply_baseline_correction()
            self.accel_matched = analyzer.accel_corrected
            
            # Compute updated spectrum
            response_spectrum = analyzer.compute_response_spectrum(
                periods=self.target_periods,
                damping=damping
            )
            
            self.current_spectrum = response_spectrum
            
            # Calculate errors
            errors = np.abs(self.current_spectrum['psa'][freq_mask] - self.target_spectrum[freq_mask]) / self.target_spectrum[freq_mask]
            avg_error = np.mean(errors)
            max_error = np.max(errors)
            
            # Record iteration result
            self.iteration_results.append({
                'iteration': iteration + 1,
                'avg_error': avg_error,
                'max_error': max_error
            })
            
            # Update the best result (using a combination metric of average and maximum error)
            # Use user-defined weights
            current_metric = avg_error * avg_weight + max_error * max_weight
            best_metric = best_avg_error * avg_weight + best_max_error * max_weight
            
            if current_metric < best_metric:
                best_avg_error = avg_error
                best_max_error = max_error
                best_accel_matched = np.copy(self.accel_matched)
                best_spectrum = response_spectrum.copy()
                best_iteration = iteration + 1
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # Terminate loop if the limit of iterations without improvement is exceeded or tolerance values are reached
            if no_improvement_count >= max_no_improvement:
                print(f"  No improvement for {no_improvement_count} iterations, best result will be used (Iteration {best_iteration}).")
                break
                
            # Check error tolerances
            if avg_error <= tolerance_avg and max_error <= tolerance_max:
                break
            
            #turned off for test mode
            #print(f"  Iteration {iteration+1}: Average error: {avg_error:.4f}, Maximum error: {max_error:.4f}")
        
        # Use the best result
        if best_accel_matched is not None and best_iteration != iteration + 1:
            print(f"  Best result found at iteration {best_iteration}. Average error: {best_avg_error:.4f}, Maximum error: {best_max_error:.4f}")
            self.accel_matched = best_accel_matched
            self.current_spectrum = best_spectrum
        
        # Return results
        return {
            'time': self.time,
            'acceleration_original': self.accel_original,
            'acceleration_matched': self.accel_matched,
            'target_periods': self.target_periods,
            'target_spectrum': self.target_spectrum,
            'matched_spectrum': self.current_spectrum,
            'iteration_results': self.iteration_results,
            'parameters': self.params,
            'best_iteration': best_iteration,
            'best_avg_error': best_avg_error,
            'best_max_error': best_max_error
        }
    
    def plot_spectra_comparison(self, figsize: Tuple[int, int] = (10, 6), 
                              log_scale: bool = True) -> plt.Figure:
        """
        Compare the target spectrum and the matched response spectrum.
        
        Args:
            figsize: Figure size
            log_scale: Whether to use logarithmic scale
            
        Returns:
            Matplotlib Figure object
        """
        self._check_setup()
        
        if self.current_spectrum is None:
            raise ValueError("You must call the match_spectrum method first.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot target spectrum
        ax.plot(self.target_periods, self.target_spectrum, 'r-', linewidth=2, label='Target Spectrum')
        
        # Plot matched spectrum
        ax.plot(self.current_spectrum['periods'], self.current_spectrum['psa'], 'b-', linewidth=2, label='Matched Spectrum')
        
        # Plot initial spectrum (optional)
        analyzer = SpectralAnalysis()
        analyzer.set_motion(self.time, self.accel_original)
        analyzer.apply_baseline_correction()
        initial_spectrum = analyzer.compute_response_spectrum(periods=self.target_periods, damping=self.params['spectral_matching']['damping'])
        ax.plot(initial_spectrum['periods'], initial_spectrum['psa'], 'g--', linewidth=1, alpha=0.6, label='Initial Spectrum')
        
        if log_scale:
            ax.set_xscale('log')
            ax.set_yscale('log')
        
        ax.set_xlabel('Period, T (s)')
        ax.set_ylabel('Spectral Acceleration, Sa(T) (g)')
        ax.set_title('Spectrum Comparison')
        ax.grid(True, which='both', linestyle='--', alpha=0.7)
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_acceleration_comparison(self, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Compare the original and matched acceleration time series.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib Figure object
        """
        self._check_setup()
        
        if self.accel_matched is None:
            raise ValueError("You must call the match_spectrum method first.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot original acceleration time series
        ax.plot(self.time, self.accel_original, 'b-', linewidth=1, alpha=0.6, label='Original')
        
        # Plot matched acceleration time series
        ax.plot(self.time, self.accel_matched, 'r-', linewidth=1, label='Matched')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Acceleration (g)')
        ax.set_title('Acceleration Time Series Comparison')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_error_convergence(self, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot the error convergence graph.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib Figure object
        """
        if not self.iteration_results:
            raise ValueError("You must call the match_spectrum method first.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Iteration numbers
        iterations = [result['iteration'] for result in self.iteration_results]
        
        # Average errors
        avg_errors = [result['avg_error'] for result in self.iteration_results]
        
        # Maximum errors
        max_errors = [result['max_error'] for result in self.iteration_results]
        
        # Error tolerances
        tolerance_avg = self.params['spectral_matching']['tolerance_avg']
        tolerance_max = self.params['spectral_matching']['tolerance_max']
        
        # Average error plot
        ax.plot(iterations, avg_errors, 'b-o', linewidth=2, label='Average Error')
        ax.axhline(y=tolerance_avg, color='b', linestyle='--', alpha=0.7, label=f'Avg. Tolerance = {tolerance_avg:.2f}')
        
        # Maximum error plot
        ax.plot(iterations, max_errors, 'r-^', linewidth=2, label='Maximum Error')
        ax.axhline(y=tolerance_max, color='r', linestyle='--', alpha=0.7, label=f'Max. Tolerance = {tolerance_max:.2f}')
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Relative Error')
        ax.set_title('Spectral Matching Convergence Graph')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Make X-axis integer
        ax.set_xticks(iterations)
        
        plt.tight_layout()
        return fig