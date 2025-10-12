#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Spectral Analysis and Baseline Correction Module.
Calculates the response spectrum and power spectral density of an acceleration time series and performs baseline correction.
"""

import numpy as np
from scipy import signal, integrate
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional, Union


class SpectralAnalysis:
    """Class for performing spectral analysis and baseline correction on an acceleration time series."""
    
    def __init__(self):
        """Initialize the class."""
        self.time = None
        self.accel = None
        self.dt = None
        self.accel_corrected = None
        self.velocity = None
        self.displacement = None
        self.periods = None
        self.response_spectrum = None
        self.frequencies = None
        self.psd = None
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
        self.accel = accel.copy()
        self.accel_corrected = None
        self.velocity = None
        self.displacement = None
        self.dt = time[1] - time[0]
        self.params = {
            'dt': self.dt,
            'duration': time[-1],
            'num_points': len(time)
        }
    
    def _check_motion(self) -> None:
        """Check if the acceleration time series is set."""
        if self.time is None or self.accel is None:
            raise ValueError("You must first call set_motion.")
    
    def apply_baseline_correction(self, cutoff_freq: float = 0.1, filter_order: int = 4) -> np.ndarray:
        """
        Apply baseline correction to the acceleration time series.
        
        Args:
            cutoff_freq: Cutoff frequency of the high-pass filter (Hz)
            filter_order: Filter order
            
        Returns:
            Corrected acceleration time series
        """
        self._check_motion()
        
        # Nyquist frequency
        nyquist = 0.5 / self.dt
        
        # Normalize cutoff frequency (0-1)
        norm_cutoff = cutoff_freq / nyquist
        
        # Butterworth high-pass filter design
        b, a = signal.butter(filter_order, norm_cutoff, btype='highpass')
        
        # Filtering (use filtfilt for zero phase shift)
        self.accel_corrected = signal.filtfilt(b, a, self.accel)
        
        self.params.update({
            'baseline_correction': {
                'cutoff_freq': cutoff_freq,
                'filter_order': filter_order,
                'nyquist': nyquist,
                'norm_cutoff': norm_cutoff
            }
        })
        
        return self.accel_corrected
    
    def compute_velocity_displacement(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate velocity and displacement from acceleration (using numerical integration).
        
        Returns:
            Velocity and displacement arrays (v, d)
        """
        self._check_motion()
        
        # If baseline correction is not applied, apply it first
        if self.accel_corrected is None:
            self.apply_baseline_correction()
        
        # Calculate velocity: v(t) = ∫ a(t) dt
        self.velocity = integrate.cumulative_trapezoid(self.accel_corrected, self.time, initial=0)
        
        # Calculate displacement: d(t) = ∫ v(t) dt
        self.displacement = integrate.cumulative_trapezoid(self.velocity, self.time, initial=0)
        
        return self.velocity, self.displacement
    
    def compute_response_spectrum(self, periods: Optional[np.ndarray] = None, 
                                damping: float = 0.05, 
                                num_periods: int = 100,
                                min_period: float = 0.01, 
                                max_period: float = 10.0,
                                log_scale: bool = True) -> Dict[str, np.ndarray]:
        """
        Calculate the response spectrum of the acceleration time series.
        
        Args:
            periods: Array of period values (None is automatically generated)
            damping: Damping ratio (default: 0.05 = 5%)
            num_periods: Number of periods to generate automatically
            min_period: Minimum period value (seconds)
            max_period: Maximum period value (seconds)
            log_scale: Whether the period values are in logarithmic scale
            
        Returns:
            Dictionary containing response spectrum values
        """
        self._check_motion()
        
        # If baseline correction is not applied, apply it first
        if self.accel_corrected is None:
            self.apply_baseline_correction()
        
        # If periods array is not provided, generate it
        if periods is None:
            if log_scale:
                periods = np.logspace(np.log10(min_period), np.log10(max_period), num_periods)
            else:
                periods = np.linspace(min_period, max_period, num_periods)
        
        self.periods = periods
        
        # Arrays to store response spectrum values
        psa = np.zeros_like(periods)  # Pseudo-Spectral Acceleration
        psv = np.zeros_like(periods)  # Pseudo-Spectral Velocity
        sd = np.zeros_like(periods)   # Spectral Displacement
        
        # For each period, solve the single-degree-of-freedom system
        for i, T in enumerate(periods):
            # Natural frequency (rad/s)
            omega = 2 * np.pi / T
            
            # Single-degree-of-freedom system parameters
            # mx'' + cx' + kx = -m*a(t)
            # ω = √(k/m), ζ = c/(2mω)
            # Solve the system for Newmark-Beta method
            
            # System matrices
            k = omega**2  # k/m ratio
            c = 2 * damping * omega  # c/m ratio
            
            # Newmark-Beta parameters
            gamma = 0.5
            beta = 0.25
            
            # State vector [x, x']
            x = np.zeros(2)
            x_next = np.zeros(2)
            
            # Maximum response values
            max_disp = 0.0
            max_vel = 0.0
            max_accel = 0.0
            
            # Time integration
            for j in range(len(self.time) - 1):
                dt = self.time[j+1] - self.time[j]
                
                # Applied force (-m*a(t))
                f = -self.accel_corrected[j]
                f_next = -self.accel_corrected[j+1]
                
                # Effective load vector
                p_eff = f_next + ((1 - gamma) * f + gamma * f_next)
                
                # Effective stiffness matrix
                k_eff = k + gamma * c / (beta * dt)
                
                # Solution
                accel = (-c * x[1] - k * x[0] + f) / 1.0  # m=1 assumed
                
                # Predicted acceleration for next step
                accel_next = (p_eff - c * (x[1] + (1 - gamma) * dt * accel) - k * (x[0] + dt * x[1] + 0.5 * dt**2 * (1 - 2 * beta) * accel)) / (1.0 + gamma * c * dt + beta * dt**2 * k)
                
                # Position and velocity update
                x_next[0] = x[0] + dt * x[1] + dt**2 * ((0.5 - beta) * accel + beta * accel_next)
                x_next[1] = x[1] + dt * ((1 - gamma) * accel + gamma * accel_next)
                
                # Update maximum values
                max_disp = max(max_disp, abs(x_next[0]))
                max_vel = max(max_vel, abs(x_next[1]))
                max_accel = max(max_accel, abs(accel_next + self.accel_corrected[j+1]))
                
                # Update values for next step
                x = x_next.copy()
            
            # Save response spectrum values
            sd[i] = max_disp
            psv[i] = omega * max_disp  # Pseudo-spectral velocity
            psa[i] = omega**2 * max_disp  # Pseudo-spectral acceleration
        
        self.response_spectrum = {
            'periods': periods,
            'sd': sd,
            'psv': psv,
            'psa': psa,
            'damping': damping
        }
        
        self.params.update({
            'response_spectrum': {
                'damping': damping,
                'num_periods': len(periods),
                'min_period': min(periods),
                'max_period': max(periods),
                'log_scale': log_scale
            }
        })
        
        return self.response_spectrum
    
    def compute_psd(self, nperseg: int = 1024, scaling: str = 'density') -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the power spectral density (PSD) of the acceleration time series.
        
        Args:
            nperseg: Number of points per segment
            scaling: Scaling type ('density' or 'spectrum')
            
        Returns:
            Frequency and PSD values (f, Pxx)
        """
        self._check_motion()
        
        # If baseline correction is not applied, apply it first
        if self.accel_corrected is None:
            self.apply_baseline_correction()
        
        # Calculate PSD using Welch method
        fs = 1 / self.dt  # Sampling frequency
        f, Pxx = signal.welch(self.accel_corrected, fs, nperseg=nperseg, scaling=scaling)
        
        self.frequencies = f
        self.psd = Pxx
        
        self.params.update({
            'psd': {
                'nperseg': nperseg,
                'scaling': scaling,
                'fs': fs
            }
        })
        
        return f, Pxx
    
    def plot_motion(self, figsize: Tuple[int, int] = (10, 10), 
                  show_uncorrected: bool = True,
                  show_velocity: bool = True,
                  show_displacement: bool = True) -> plt.Figure:
        """
        Plot the acceleration, velocity and displacement time series.
        
        Args:
            figsize: Graph size
            show_uncorrected: Whether to show uncorrected acceleration
            show_velocity: Whether to show velocity
            show_displacement: Whether to show displacement
            
        Returns:
            Matplotlib Figure object
        """
        self._check_motion()
        
        # If baseline correction is not applied, apply it first
        if self.accel_corrected is None:
            self.apply_baseline_correction()
        
        # If velocity and displacement are not calculated, calculate them
        if self.velocity is None or self.displacement is None:
            self.compute_velocity_displacement()
        
        num_plots = 1 + int(show_velocity) + int(show_displacement)
        fig, axes = plt.subplots(num_plots, 1, figsize=figsize, sharex=True)
        
        if num_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # Plot acceleration time series
        ax = axes[plot_idx]
        if show_uncorrected:
            ax.plot(self.time, self.accel, 'b-', alpha=0.5, linewidth=1, label='Uncorrected')
        ax.plot(self.time, self.accel_corrected, 'r-', linewidth=1, label='Corrected')
        ax.set_ylabel('Acceleration (g)')
        ax.set_title('Acceleration Time Series')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Plot velocity time series
        if show_velocity:
            plot_idx += 1
            ax = axes[plot_idx]
            ax.plot(self.time, self.velocity, 'g-', linewidth=1)
            ax.set_ylabel('Velocity (g*s)')
            ax.set_title('Velocity Time Series')
            ax.grid(True, linestyle='--', alpha=0.7)
        
        # Plot displacement time series
        if show_displacement:
            plot_idx += 1
            ax = axes[plot_idx]
            ax.plot(self.time, self.displacement, 'm-', linewidth=1)
            ax.set_ylabel('Displacement (g*s²)')
            ax.set_title('Displacement Time Series')
            ax.grid(True, linestyle='--', alpha=0.7)
        
        axes[-1].set_xlabel('Time (s)')
        
        plt.tight_layout()
        return fig
    
    def plot_response_spectrum(self, figsize: Tuple[int, int] = (10, 6), 
                              log_scale: bool = True) -> plt.Figure:
        """
        Plot the response spectrum.
        
        Args:
            figsize: Graph size
            log_scale: Whether to use logarithmic scale
            
        Returns:
            Matplotlib Figure object
        """
        if self.response_spectrum is None:
            raise ValueError("You must call the compute_response_spectrum method first.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot the response spectrum
        ax.plot(self.response_spectrum['periods'], self.response_spectrum['psa'], 'b-', linewidth=2)
        
        if log_scale:
            ax.set_xscale('log')
            ax.set_yscale('log')
        
        ax.set_xlabel('Period, T (s)')
        ax.set_ylabel('Spectral Acceleration, Sa(T) (g)')
        ax.set_title(f'Response Spectrum (Damping = {self.response_spectrum["damping"]*100:.0f}%)')
        ax.grid(True, which='both', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig
    
    def plot_psd(self, figsize: Tuple[int, int] = (10, 6), 
                log_scale: bool = True) -> plt.Figure:
        """
        Plot the power spectral density (PSD).
        
        Args:
            figsize: Graph size
            log_scale: Whether to use logarithmic scale
            
        Returns:
            Matplotlib Figure object
        """
        if self.frequencies is None or self.psd is None:
            raise ValueError("You must call the compute_psd method first.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot PSD
        if log_scale:
            ax.semilogy(self.frequencies, self.psd)
        else:
            ax.plot(self.frequencies, self.psd)
        
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD ((g)²/Hz)')
        ax.set_title('Power Spectral Density')
        ax.grid(True, which='both', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig
    
    def plot_spectrogram(self, figsize: Tuple[int, int] = (10, 6), 
                       nperseg: int = 256, noverlap: int = 128) -> plt.Figure:
        """
        Plot the spectrogram.
        
        Args:
            figsize: Graph size
            nperseg: Number of points per segment
            noverlap: Number of overlapping points
            
        Returns:
            Matplotlib Figure object
        """
        self._check_motion()
        
        # If baseline correction is not applied, apply it first
        if self.accel_corrected is None:
            self.apply_baseline_correction()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate and plot spectrogram
        fs = 1 / self.dt  # Sampling frequency
        f, t, Sxx = signal.spectrogram(self.accel_corrected, fs, nperseg=nperseg, noverlap=noverlap)
        
        # Power density in dB
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        
        # Plot spectrogram (pcolormesh is faster)
        pcm = ax.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='viridis')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title('Spectrogram')
        
        # Add color bar
        cbar = fig.colorbar(pcm, ax=ax)
        cbar.set_label('Power Density (dB/Hz)')
        
        plt.tight_layout()
        return fig
    
    def get_analysis_data(self) -> Dict:
        """
        Return analysis data as a dictionary.
        
        Returns:
            Dictionary containing analysis data
        """
        self._check_motion()
        
        # Check missing analyses and calculate if necessary
        if self.accel_corrected is None:
            self.apply_baseline_correction()
        
        if self.velocity is None or self.displacement is None:
            self.compute_velocity_displacement()
        
        if self.response_spectrum is None:
            self.compute_response_spectrum()
        
        if self.frequencies is None or self.psd is None:
            self.compute_psd()
        
        return {
            'time': self.time,
            'acceleration_original': self.accel,
            'acceleration_corrected': self.accel_corrected,
            'velocity': self.velocity,
            'displacement': self.displacement,
            'response_spectrum': self.response_spectrum,
            'frequencies': self.frequencies,
            'psd': self.psd,
            'parameters': self.params
        }