#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Envelope Function Application Module.
Applies earthquake-like amplitude variation to acceleration time series.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional, Callable, Union, List


class EnvelopeFunction:
    """Enveloping class for acceleration time series."""
    
    def __init__(self):
        """Initialize the class."""
        self.time = None
        self.accel_original = None
        self.accel_enveloped = None
        self.envelope = None
        self.params = {}
    
    def set_motion(self, time: np.ndarray, accel: np.ndarray) -> None:
        """
        Set the original acceleration time series.
        
        Args:
            time: Time array
            accel: Acceleration array
        """
        self.time = time
        self.accel_original = accel.copy()
        self.accel_enveloped = None
        self.envelope = None
    
    def _check_motion(self) -> None:
        """Check if the acceleration time series is set."""
        if self.time is None or self.accel_original is None:
            raise ValueError("First call the set_motion method.")
    
    def create_custom_envelope(self, t_rise_ratio: float = 0.15, 
                                     t_steady_ratio: float = 0.7, 
                                     decay_factor: float = 0.2) -> np.ndarray:
        """
        Create the custom envelope. (Works pretty well)
        
        Args:
            t_rise_ratio: ratio of the rise time to the total time
            t_steady_ratio: ratio of the steady time to the total time
            decay_factor: decay factor
            
        Returns:
            Envelope values array
        """
        self._check_motion()
        
        # Total duration
        total_duration = self.time[-1]
        
        # Calculate the durations of the envelope sections
        t_rise = t_rise_ratio * total_duration
        t_steady = t_steady_ratio * total_duration
        t_decay_start = t_rise + t_steady
        
        # Create the envelope array
        envelope = np.zeros_like(self.time)
        
        for i, t in enumerate(self.time):
            if t <= t_rise:
                # Rise section (t^2)
                envelope[i] = (t / t_rise) ** 2
            elif t <= t_decay_start:
                # Steady section
                envelope[i] = 1.0
            else:
                # Decay section (exp(-c*t))
                envelope[i] = np.exp(-decay_factor * (t - t_decay_start))
        
        self.envelope = envelope
        self.params = {
            'envelope_type': 'custom',
            't_rise_ratio': t_rise_ratio,
            't_steady_ratio': t_steady_ratio,
            'decay_factor': decay_factor,
            't_rise': t_rise,
            't_steady': t_steady,
            't_decay_start': t_decay_start
        }
        
        return envelope
    
    def create_saragoni_hart_envelope(self, alpha: float = 0.1, beta: float = 0.5, theta: float = 0.5) -> np.ndarray:
        """
        Create the Saragoni-Hart envelope. I(t)=alpha*t^beta*e^(-theta*t)
        
        Args:
            alpha: Parameter
            beta: Parameter
            theta: Parameter
        
        Returns:
            Envelope values array
        """
        self._check_motion()
        
        # Total duration
        total_duration = self.time[-1]
        
        # Create the envelope array
        envelope = np.zeros_like(self.time)
        
        for i, t in enumerate(self.time):
            envelope[i] = alpha*t**beta*np.exp(-theta*t)

        self.envelope = envelope
        self.params = {
            'envelope_type': 'saragoni_hart',
            'alpha': alpha,
            'beta': beta,
            'theta': theta
        }
        
        return envelope
    
    def apply_envelope(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply the envelope function to the acceleration time series.
        
        Returns:
            Time and enveloped acceleration arrays (t, accel_enveloped)
        """
        self._check_motion()
        
        if self.envelope is None:
            raise ValueError("You must first create an envelope function.")
        
        # Apply the envelope function
        self.accel_enveloped = self.accel_original * self.envelope
        
        return self.time, self.accel_enveloped
    
    def plot_envelope(self, figsize: Tuple[int, int] = (10, 8), 
                     show_original: bool = True) -> plt.Figure:
        """
        Plot the envelope function and acceleration time series.
        
        Args:
            figsize: Figure size
            show_original: Whether to show the original acceleration time series
            
        Returns:
            Matplotlib Figure object
        """
        self._check_motion()
        
        if self.envelope is None:
            raise ValueError("You must first create an envelope function.")
        
        if self.accel_enveloped is None:
            self.apply_envelope()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # Plot the envelope function
        ax1.plot(self.time, self.envelope, 'r-', linewidth=2, label='Envelope')
        ax1.set_ylabel('Amplitude')
        ax1.set_title('Envelope Function')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()
        
        # Plot the acceleration time series
        if show_original:
            ax2.plot(self.time, self.accel_original, 'b-', alpha=0.5, linewidth=1, label='Original')
        ax2.plot(self.time, self.accel_enveloped, 'g-', linewidth=1, label='Enveloped')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Acceleration (g)')
        ax2.set_title('Acceleration Time Series')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()
        
        # Mark important points according to envelope type
        if self.params.get('envelope_type') == 'saragoni_hart':
            t_rise = self.params.get('t_rise', 0)
            t_decay_start = self.params.get('t_decay_start', 0)
            
            ax1.axvline(x=t_rise, color='gray', linestyle='--', alpha=0.5)
            ax1.axvline(x=t_decay_start, color='gray', linestyle='--', alpha=0.5)
            
            ax1.text(t_rise, 0.5, f'$t_{{rise}}$={t_rise:.2f}s', 
                    rotation=90, verticalalignment='center')
            ax1.text(t_decay_start, 0.5, f'$t_{{decay}}$={t_decay_start:.2f}s', 
                    rotation=90, verticalalignment='center')
            
        elif self.params.get('envelope_type') in ['trapezoidal', 'jennings']:
            t_rise = self.params.get('t_rise', 0)
            t_steady_end = self.params.get('t_steady_end', 0)
            
            ax1.axvline(x=t_rise, color='gray', linestyle='--', alpha=0.5)
            ax1.axvline(x=t_steady_end, color='gray', linestyle='--', alpha=0.5)
            
            ax1.text(t_rise, 0.5, f'$t_{{rise}}$={t_rise:.2f}s', 
                    rotation=90, verticalalignment='center')
            ax1.text(t_steady_end, 0.5, f'$t_{{steady-end}}$={t_steady_end:.2f}s', 
                    rotation=90, verticalalignment='center')
        
        plt.tight_layout()
        return fig
    
    def get_envelope_data(self) -> Dict:
        """
        Return envelope data and enveloped acceleration time series as a dictionary.
        
        Returns:
            Dictionary containing envelope data and enveloped acceleration time series
        """
        self._check_motion()
        
        if self.envelope is None:
            raise ValueError("You must first create an envelope function.")
        
        if self.accel_enveloped is None:
            self.apply_envelope()
        
        return {
            'time': self.time,
            'acceleration_original': self.accel_original,
            'acceleration_enveloped': self.accel_enveloped,
            'envelope': self.envelope,
            'parameters': self.params
        }