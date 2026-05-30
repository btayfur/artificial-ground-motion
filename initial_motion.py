#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Initial Acceleration Time Series Generation Module.
Generates filtered white noise using the Kanai-Tajimi filter.
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional


class InitialMotionGenerator:
    """Class for generating initial acceleration time series."""
    
    def __init__(self):
        """Initialize the class."""
        self.time = None
        self.accel = None
        self.params = {}
    
    def generate_white_noise(self, duration: float, dt: float, 
                             mean: float = 0.0, std: float = 1.0, 
                             seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate white noise signal.
        
        Args:
            duration: Total duration (seconds)
            dt: Time step (seconds)
            mean: Noise mean
            std: Noise standard deviation
            seed: Seed value for random number generator
            
        Returns:
            Time and acceleration arrays (t, accel)
        """
        # Set the random number generator
        if seed is not None:
            np.random.seed(seed)
        
        # Create time array
        num_samples = int(duration / dt) + 1
        self.time = np.linspace(0, duration, num_samples)
        
        # Generate white noise signal
        self.accel = np.random.normal(mean, std, size=num_samples)
        
        self.params = {
            'duration': duration,
            'dt': dt,
            'mean': mean,
            'std': std,
            'seed': seed,
            'num_samples': num_samples
        }
        
        return self.time, self.accel

    def generate_linear_congruential_noise(self, 
                                       duration: float,
                                       dt: float = 0.01,
                                       mean: float = 0.0, 
                                       std: float = 1.0, 
                                       seed: Optional[int] = None,
                                       m: int = 2**32,
                                       a: int = 1664525,
                                       c: int = 1013904223) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate noise using a Linear Congruential Generator (LCG).

        Args:
            duration: Total duration (seconds)
            dt: Time step (seconds, default: 0.01)
            mean: Desired mean of the noise
            std: Desired standard deviation of the noise
            seed: Initial seed for reproducibility (default: None)
            m: Modulus parameter for LCG (default: 2**32)
            a: Multiplier parameter for LCG (default: 1664525)
            c: Increment parameter for LCG (default: 1013904223)

        Returns:
            Tuple of (time array, noise array)
        """
        # Number of samples based on duration and dt
        num_samples = int(duration / dt) + 1
        self.time = np.linspace(0, duration, num_samples)

        # Initialize seed
        if seed is None:
            seed = np.random.SeedSequence().entropy
        x = int(seed) % m

        # Generate uniform pseudorandom values using LCG
        u = np.empty(num_samples, dtype=np.float64)
        for i in range(num_samples):
            x = (a * x + c) % m
            u[i] = x / m  # normalize to [0, 1)

        # Convert to Gaussian-like noise (Box-Muller transform)
        values = np.sqrt(-2.0 * np.log(u + 1e-12)) * np.cos(2 * np.pi * u)

        # Scale to desired mean and std
        noise = (values - np.mean(values)) / np.std(values)
        self.accel = noise * std + mean

        self.params = {
            'duration': duration,
            'dt': dt,
            'mean': mean,
            'std': std,
            'seed': seed,
            'num_samples': num_samples
        }

        return self.time, self.accel
    
    def apply_kanai_tajimi_filter(self, omega_g: float, zeta_g: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Kanai-Tajimi filter.
        
        Args:
            omega_g: Ground dominant angular frequency (rad/s)
            zeta_g: Ground damping ratio
            
        Returns:
            Time and filtered acceleration arrays (t, accel_filtered)
        """
        if self.time is None or self.accel is None:
            raise ValueError("You must first call the generate_white_noise method.")
        
        # Time step
        dt = self.params['dt']
        
        # Parameters for frequency domain
        n = len(self.time)
        freq = np.fft.rfftfreq(n, dt)  # Frequency array (Hz)
        omega = 2 * np.pi * freq  # Angular frequency (rad/s)
        
        # Kanai-Tajimi transfer function (H(ω))
        # Power spectral density: S(ω) = S₀ * |H(ω)|²
        # H(ω) = (1 + 2ζᵍi(ω/ωᵍ)) / ((1-(ω/ωᵍ)²) + 2ζᵍi(ω/ωᵍ))
        numerator = 1 + 2j * zeta_g * (omega / omega_g)
        denominator = (1 - (omega / omega_g)**2) + 2j * zeta_g * (omega / omega_g)
        H = numerator / denominator
        
        # FFT of white noise
        accel_fft = np.fft.rfft(self.accel)
        
        # Multiply FFT by Kanai-Tajimi transfer function
        accel_fft_filtered = accel_fft * H
        
        # Perform IFFT to return to time domain
        accel_filtered = np.fft.irfft(accel_fft_filtered, n=n)
        
        # Normalize the result (standard deviation will be 1)
        accel_filtered = accel_filtered / np.std(accel_filtered)
        
        # Set initial values to zero
        accel_filtered[0] = 0
        
        self.accel = accel_filtered
        self.params.update({
            'filter_type': 'kanai_tajimi',
            'omega_g': omega_g,
            'zeta_g': zeta_g
        })
        
        return self.time, self.accel
    
    def generate_kanai_tajimi_motion(self, duration: float, dt: float, 
                                     omega_g: float, zeta_g: float, 
                                     target_pga: float = 1.0, 
                                     seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Kanai-Tajimi filtered white noise acceleration time series.
        
        Args:
            duration: Total duration (seconds)
            dt: Time step (seconds)
            omega_g: Ground dominant angular frequency (rad/s)
            zeta_g: Ground damping ratio
            target_pga: Target PGA value (g or m/s²)
            seed: Seed value for random number generator
            
        Returns:
            Time and acceleration arrays (t, accel)
        """
        # Generate white noise
        self.generate_white_noise(duration, dt, seed=seed)
        
        # Apply Kanai-Tajimi filter
        self.apply_kanai_tajimi_filter(omega_g, zeta_g)
        
        # Scale according to PGA
        self.apply_scaling(target_pga)
        
        return self.time, self.accel
    
    def apply_clough_penzien_filter(self, omega_g: float, zeta_g: float, 
                                   omega_f: float, zeta_f: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Clough-Penzien filter.
        
        Args:
            omega_g: Ground dominant angular frequency (rad/s)
            zeta_g: Ground damping ratio
            omega_f: Filter dominant angular frequency (rad/s), typically smaller than omega_g
            zeta_f: Filter damping ratio
            
        Returns:
            Time and filtered acceleration arrays (t, accel_filtered)
        """
        if self.time is None or self.accel is None:
            raise ValueError("You must first call the generate_white_noise method.")
        
        # Time step
        dt = self.params['dt']
        
        # Parameters for frequency domain
        n = len(self.time)
        freq = np.fft.rfftfreq(n, dt)  # Frequency array (Hz)
        omega = 2 * np.pi * freq  # Angular frequency (rad/s)
        
        # Kanai-Tajimi transfer function (first filter)
        numerator_kt = 1 + 2j * zeta_g * (omega / omega_g)
        denominator_kt = (1 - (omega / omega_g)**2) + 2j * zeta_g * (omega / omega_g)
        H_kt = numerator_kt / denominator_kt
        
        # High-pass filter transfer function (second filter)
        numerator_hp = (omega / omega_f)**2
        denominator_hp = (1 - (omega / omega_f)**2) + 2j * zeta_f * (omega / omega_f)
        H_hp = numerator_hp / denominator_hp
        
        # Combined Clough-Penzien transfer function
        H = H_kt * H_hp
        
        # FFT of white noise
        accel_fft = np.fft.rfft(self.accel)
        
        # Multiply FFT by Clough-Penzien transfer function
        accel_fft_filtered = accel_fft * H
        
        # Perform IFFT to return to time domain
        accel_filtered = np.fft.irfft(accel_fft_filtered, n=n)
        
        # Normalize the result (standard deviation will be 1)
        accel_filtered = accel_filtered / np.std(accel_filtered)
        
        # Set initial values to zero
        accel_filtered[0] = 0
        
        self.accel = accel_filtered
        self.params.update({
            'filter_type': 'clough_penzien',
            'omega_g': omega_g,
            'zeta_g': zeta_g,
            'omega_f': omega_f,
            'zeta_f': zeta_f
        })
        
        return self.time, self.accel
    
    def generate_clough_penzien_motion(self, duration: float, dt: float, 
                                      omega_g: float, zeta_g: float,
                                      omega_f: float, zeta_f: float,
                                      target_pga: float = 1.0, 
                                      seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Clough-Penzien filtered white noise acceleration time series.
        
        Args:
            duration: Total duration (seconds)
            dt: Time step (seconds)
            omega_g: Ground dominant angular frequency (rad/s)
            zeta_g: Ground damping ratio
            omega_f: Filter dominant angular frequency (rad/s), typically smaller than omega_g
            zeta_f: Filter damping ratio
            target_pga: Target PGA value (g or m/s²)
            seed: Seed value for random number generator
            
        Returns:
            Time and acceleration arrays (t, accel)
        """
        # Generate white noise
        self.generate_white_noise(duration, dt, seed=seed)
        #self.generate_linear_congruential_noise(duration, dt)
        
        # Apply Clough-Penzien filter
        self.apply_clough_penzien_filter(omega_g, zeta_g, omega_f, zeta_f)
        
        # Scale according to PGA
        self.apply_scaling(target_pga)
        
        return self.time, self.accel
    
    def apply_scaling(self, target_pga: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scale the acceleration time series to a specific PGA (Peak Ground Acceleration) value.
        
        Args:
            target_pga: Target PGA value (g or m/s²)
            
        Returns:
            Time and scaled acceleration arrays (t, accel_scaled)
        """
        if self.time is None or self.accel is None:
            raise ValueError("You must first generate the acceleration time series.")
        
        # Calculate current PGA
        current_pga = np.max(np.abs(self.accel))
        
        # Calculate scaling factor
        scale_factor = target_pga / current_pga
        
        # Scale the acceleration
        self.accel = self.accel * scale_factor
        
        self.params['target_pga'] = target_pga
        self.params['scale_factor'] = scale_factor
        
        return self.time, self.accel
  
    def plot_acceleration(self, figsize: Tuple[int, int] = (10, 6), 
                         title: str = "Initial Acceleration Time Series") -> plt.Figure:
        """
        Plot the acceleration time series.
        
        Args:
            figsize: Figure size
            title: Figure title
            
        Returns:
            Matplotlib Figure object
        """
        if self.time is None or self.accel is None:
            raise ValueError("You must first generate the acceleration time series.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(self.time, self.accel, 'b-', linewidth=1)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Acceleration (g)')
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Show PGA value
        pga = np.max(np.abs(self.accel))
        text_str = f"PGA = {pga:.4f}g"
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        return fig
    
    def plot_psd(self, figsize: Tuple[int, int] = (10, 6), 
                title: str = "Power Spectral Density") -> plt.Figure:
        """
        Plot the Power Spectral Density (PSD).
        
        Args:
            figsize: Figure size
            title: Figure title
            
        Returns:
            Matplotlib Figure object
        """
        if self.time is None or self.accel is None:
            raise ValueError("You must first generate the acceleration time series.")
        
        # Calculate PSD using Welch method
        fs = 1 / self.params['dt']  # Sampling frequency
        f, Pxx = signal.welch(self.accel, fs, nperseg=1024, scaling='density')
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.semilogy(f, Pxx)
        
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD ((g)²/Hz)')
        ax.set_title(title)
        ax.grid(True, which='both', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig
    
    def get_motion_data(self) -> Dict:
        """
        Return the generated acceleration time series data as a dictionary.
        
        Returns:
            dict that contains motion data
        """
        if self.time is None or self.accel is None:
            raise ValueError("You must first generate the acceleration time series.")
        
        return {
            'time': self.time,
            'acceleration': self.accel,
            'parameters': self.params
        }
