#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Design Spectrum Generation Module for TBDY 2018.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Dict



@dataclass
class DesignSpectrumParams:
    """Data class for design spectrum parameters."""
    Ss: float  # Short-period design spectral acceleration coefficient
    S1: float  # 1.0 second period design spectral acceleration coefficient
    soil_class: str  # Local Soil Class
    Fs: float = None  # Local soil effect coefficient for short period
    F1: float = None  # Local soil effect coefficient for 1.0 second period
    SDS: float = None  # Short-period design spectral acceleration coefficient
    SD1: float = None  # Design spectral acceleration coefficient for 1.0 second period
    TA: float = None  # Horizontal elastic design spectrum corner period
    TB: float = None  # Horizontal elastic design spectrum corner period


class TBDYSpectrum:
    """Class for calculating design spectrum according to TBDY 2018."""
    
    # Local soil effect coefficients (Fs) according to TBDY 2018 Table 3.1
    FS_TABLE = {
        'ZA': {0.25: 0.8, 0.50: 0.8, 0.75: 0.8, 1.00: 0.8, 1.25: 0.8, 1.50: 0.8},
        'ZB': {0.25: 0.9, 0.50: 0.9, 0.75: 0.9, 1.00: 0.9, 1.25: 0.9, 1.50: 0.9},
        'ZC': {0.25: 1.3, 0.50: 1.3, 0.75: 1.2, 1.00: 1.2, 1.25: 1.1, 1.50: 1.1},
        'ZD': {0.25: 1.6, 0.50: 1.4, 0.75: 1.2, 1.00: 1.1, 1.25: 1.0, 1.50: 1.0},
        'ZE': {0.25: 2.4, 0.50: 1.7, 0.75: 1.3, 1.00: 1.1, 1.25: 0.9, 1.50: 0.8},
    }
    
    # Local soil effect coefficients (F1) according to TBDY 2018 Table 3.1
    F1_TABLE = {
        'ZA': {0.10: 0.8, 0.20: 0.8, 0.30: 0.8, 0.40: 0.8, 0.50: 0.8, 0.60: 0.8},
        'ZB': {0.10: 0.8, 0.20: 0.8, 0.30: 0.8, 0.40: 0.8, 0.50: 0.8, 0.60: 0.8},
        'ZC': {0.10: 1.5, 0.20: 1.5, 0.30: 1.5, 0.40: 1.4, 0.50: 1.3, 0.60: 1.3},
        'ZD': {0.10: 2.4, 0.20: 2.2, 0.30: 2.0, 0.40: 1.9, 0.50: 1.8, 0.60: 1.7},
        'ZE': {0.10: 4.2, 0.20: 3.3, 0.30: 2.8, 0.40: 2.4, 0.50: 2.2, 0.60: 2.0},
    }
    
    def __init__(self):
        """Initialize the class."""
        self.params = None
        self.periods = None
        self.spectral_values = None
    
    def calculate_local_soil_factors(self, params: DesignSpectrumParams) -> Tuple[float, float]:
        """
        Calculate local soil effect coefficients (Fs, F1).
        
        Args:
            params: Design spectrum parameters
            
        Returns:
            Fs, F1 values (float, float)
        """
        # Invalid soil class check
        if params.soil_class not in self.FS_TABLE or params.soil_class == 'ZF':
            raise ValueError(f"Soil class '{params.soil_class}' is not supported or requires special investigation (ZF).")
        
        # Calculate Fs value by interpolation based on Ss value
        ss_keys = sorted(self.FS_TABLE[params.soil_class].keys())
        if params.Ss <= ss_keys[0]:
            fs = self.FS_TABLE[params.soil_class][ss_keys[0]]
        elif params.Ss >= ss_keys[-1]:
            fs = self.FS_TABLE[params.soil_class][ss_keys[-1]]
        else:
            for i in range(len(ss_keys) - 1):
                if ss_keys[i] <= params.Ss <= ss_keys[i+1]:
                    fs = np.interp(params.Ss, [ss_keys[i], ss_keys[i+1]], 
                                  [self.FS_TABLE[params.soil_class][ss_keys[i]], 
                                   self.FS_TABLE[params.soil_class][ss_keys[i+1]]])
                    break
        
        # Calculate F1 value by interpolation based on S1 value
        s1_keys = sorted(self.F1_TABLE[params.soil_class].keys())
        if params.S1 <= s1_keys[0]:
            f1 = self.F1_TABLE[params.soil_class][s1_keys[0]]
        elif params.S1 >= s1_keys[-1]:
            f1 = self.F1_TABLE[params.soil_class][s1_keys[-1]]
        else:
            for i in range(len(s1_keys) - 1):
                if s1_keys[i] <= params.S1 <= s1_keys[i+1]:
                    f1 = np.interp(params.S1, [s1_keys[i], s1_keys[i+1]], 
                                  [self.F1_TABLE[params.soil_class][s1_keys[i]], 
                                   self.F1_TABLE[params.soil_class][s1_keys[i+1]]])
                    break
        
        return fs, f1
    
    def calculate_spectrum_parameters(self, params: DesignSpectrumParams) -> DesignSpectrumParams:
        """
        Calculate spectrum parameters (SDS, SD1, TA, TB).
        
        Args:
            params: Design spectrum parameters
            
        Returns:
            Updated design spectrum parameters
        """
        # Calculate local soil effect coefficients
        fs, f1 = self.calculate_local_soil_factors(params)
        params.Fs = fs
        params.F1 = f1
        
        # Calculate design spectral acceleration coefficients
        params.SDS = params.Ss * params.Fs
        params.SD1 = params.S1 * params.F1
        
        # Calculate corner periods
        params.TA = 0.2 * params.SD1 / params.SDS
        params.TB = params.SD1 / params.SDS
        
        return params
    
    def generate_spectrum(self, Ss: float, S1: float, soil_class: str, 
                          min_period: float = 0.01, max_period: float = 10.0, 
                          num_points: int = 300, log_scale: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate design acceleration spectrum according to TBDY 2018.
        
        Args:
            Ss: Short-period design spectral acceleration coefficient
            S1: Design spectral acceleration coefficient for 1.0 second period
            soil_class: Local Soil Class (ZA, ZB, ZC, ZD, ZE)
            min_period: Minimum period value (seconds)
            max_period: Maximum period value (seconds)
            num_points: Number of points to calculate
            log_scale: Whether period values will be on a logarithmic scale
            
        Returns:
            Period array and spectral acceleration values (T, Sa)
        """
        # Create parameters
        params = DesignSpectrumParams(Ss=Ss, S1=S1, soil_class=soil_class)
        
        # Calculate parameters
        self.params = self.calculate_spectrum_parameters(params)
        
        # Create period array
        if log_scale:
            self.periods = np.logspace(np.log10(min_period), np.log10(max_period), num_points)
        else:
            self.periods = np.linspace(min_period, max_period, num_points)
        
        # Calculate spectral acceleration values
        self.spectral_values = np.zeros_like(self.periods)
        
        for i, T in enumerate(self.periods):
            # According to TBDY 2018 Equation 3.1a-d
            if T <= self.params.TA:
                # Equation 3.1a
                self.spectral_values[i] = (0.4 + 0.6 * T / self.params.TA) * self.params.SDS
            elif T <= self.params.TB:
                # Equation 3.1b
                self.spectral_values[i] = self.params.SDS
            elif T <= 6.0:
                # Equation 3.1c
                self.spectral_values[i] = self.params.SD1 / T
            else:
                # Equation 3.1d (T > 6.0)
                self.spectral_values[i] = self.params.SD1 * 6.0 / (T * T)
        
        return self.periods, self.spectral_values
    
    def plot_spectrum(self, title: str = "TBDY 2018 Design Spectrum", 
                      log_scale: bool = False, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot the design spectrum.
        
        Args:
            title: Graph title
            log_scale: Whether to use logarithmic scale
            figsize: Graph size
            
        Returns:
            Matplotlib Figure object
        """
        if self.periods is None or self.spectral_values is None:
            raise ValueError("You must call the generate_spectrum method first.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(self.periods, self.spectral_values, 'b-', linewidth=2)
        
        # Mark corner periods
        ax.axvline(x=self.params.TA, color='r', linestyle='--', linewidth=1, 
                  label=f"TA = {self.params.TA:.2f}s")
        ax.axvline(x=self.params.TB, color='g', linestyle='--', linewidth=1, 
                  label=f"TB = {self.params.TB:.2f}s")
        
        if log_scale:
            ax.set_xscale('log')
            ax.set_yscale('log')
        
        ax.set_xlabel('Period, T (s)')
        ax.set_ylabel('Spectral Acceleration, Sa(T) (g)')
        ax.set_title(title)
        ax.grid(True, which='both', linestyle='--', alpha=0.7)
        ax.legend()
        
        # Add text showing spectrum parameters
        text_str = f"Ss = {self.params.Ss:.2f}, S1 = {self.params.S1:.2f}\n" \
                   f"Soil Class = {self.params.soil_class}\n" \
                   f"SDS = {self.params.SDS:.2f}, SD1 = {self.params.SD1:.2f}"
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        return fig
    
    def get_spectrum_data(self) -> Dict:
        """
        Return the calculated spectrum data as a dictionary.
        
        Returns:
            Dictionary containing spectrum parameters and data
        """
        if self.params is None:
            raise ValueError("You must call the generate_spectrum method first.")
        
        return {
            'parameters': {
                'Ss': self.params.Ss,
                'S1': self.params.S1,
                'soil_class': self.params.soil_class,
                'Fs': self.params.Fs,
                'F1': self.params.F1,
                'SDS': self.params.SDS,
                'SD1': self.params.SD1,
                'TA': self.params.TA,
                'TB': self.params.TB
            },
            'periods': self.periods,
            'spectral_values': self.spectral_values
        }
