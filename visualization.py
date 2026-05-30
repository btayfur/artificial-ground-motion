#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Output Generation and Visualization Module.
Visualizes the generated artificial ground motion and related analysis results and saves them as files.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
import datetime
from typing import Tuple, Dict, List, Optional, Union

from design_spectrum import TBDYSpectrum
from spectral_analysis import SpectralAnalysis
from spectral_matching import SpectralMatcher

# Seaborn and matplotlib settings - suitable for academic paper standards
sns.set_theme(style="whitegrid", font_scale=1.2)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['axes.spines.top'] = True
plt.rcParams['axes.spines.right'] = True
plt.rcParams['axes.spines.left'] = True
plt.rcParams['axes.spines.bottom'] = True
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 100


# Color palette for academic publications
COLORS = sns.color_palette("bright")
BLUE = COLORS[0]  # For response spectrum
RED = COLORS[3]   # For target spectrum
GREEN = COLORS[2] # For initial spectrum

class OutputGenerator:
    """
    Visualizes and saves artificial ground motion and analysis results.
    """
    
    def __init__(self, output_dir: str = "outputs", damping: float = 0.05):
        """
        Initialize the class.
        
        Args:
            output_dir: Directory where outputs will be saved
            damping: Damping ratio (default: 0.05 = 5%)
        """
        self.output_dir = output_dir
        self.damping = damping
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Generate a unique project code (based on date and time)
        self.project_code = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Data variables
        self.time = None
        self.acceleration = None
        self.target_periods = None
        self.target_spectrum = None
        self.response_spectrum = None
        self.frequencies = None
        self.psd = None
        self.params = {}
    
    def set_data(self, time: np.ndarray, acceleration: np.ndarray, 
                target_periods: Optional[np.ndarray] = None, 
                target_spectrum: Optional[np.ndarray] = None,
                response_spectrum: Optional[Dict] = None,
                frequencies: Optional[np.ndarray] = None,
                psd: Optional[np.ndarray] = None,
                params: Optional[Dict] = None) -> None:
        """
        Set the data to be visualized and saved.
        
        Args:
            time: Time series
            acceleration: Acceleration series
            target_periods: Target design spectrum period series (optional)
            target_spectrum: Target design spectrum values (optional)
            response_spectrum: Response spectrum dictionary (optional)
            frequencies: Frequency series (for PSD) (optional)
            psd: Power spectral density values (optional)
            params: Additional parameters (optional)
        """
        self.time = time
        self.acceleration = acceleration
        self.target_periods = target_periods
        self.target_spectrum = target_spectrum
        self.response_spectrum = response_spectrum
        self.frequencies = frequencies
        self.psd = psd
        
        if params:
            self.params = params
    
    def _check_data(self) -> None:
        """Check if the data has been set."""
        if self.time is None or self.acceleration is None:
            raise ValueError("You must call the set_data method first.")
    
    def calculate_missing_data(self) -> None:
        """
        Calculate missing data.
        """
        self._check_data()
        
        # Create spectral analysis object
        analyzer = SpectralAnalysis()
        analyzer.set_motion(self.time, self.acceleration)
        analyzer.apply_baseline_correction()
        
        # If response spectrum has not been calculated yet
        if self.response_spectrum is None:
            # If target periods exist, use the same periods
            if self.target_periods is not None:
                self.response_spectrum = analyzer.compute_response_spectrum(
                    periods=self.target_periods,
                    damping=self.damping
                )
            else:
                self.response_spectrum = analyzer.compute_response_spectrum(
                    damping=self.damping,
                    log_scale=True
                )
                self.target_periods = self.response_spectrum['periods']
        
        # If PSD has not been calculated yet
        if self.frequencies is None or self.psd is None:
            self.frequencies, self.psd = analyzer.compute_psd()
    
    def save_acceleration_data(self, filename: Optional[str] = None, 
                             header: str = "Time (s), Acceleration (g)") -> str:
        """
        Save the acceleration time series to a text file.
        
        Args:
            filename: Name of the file to be saved (automatically generated if None)
            header: File header
            
        Returns:
            Full path of the saved file
        """
        self._check_data()
        
        # Automatically generate filename if not specified
        if filename is None:
            filename = f"accelerogram_{self.project_code}.txt"
        
        # Full path of the file
        filepath = os.path.join(self.output_dir, filename)
        
        # Save acceleration data
        data = np.column_stack((self.time, self.acceleration))
        np.savetxt(filepath, data, delimiter=",", header=header, comments="# ")
        
        return filepath
    
    def save_spectrum_data(self, filename: Optional[str] = None, 
                         header: str = "Period (s), Spectral Acceleration (g)") -> str:
        """
        Save the response spectrum to a text file.
        
        Args:
            filename: Name of the file to be saved (automatically generated if None)
            header: File header
            
        Returns:
            Full path of the saved file
        """
        self._check_data()
        
        # Calculate missing data
        if self.response_spectrum is None:
            self.calculate_missing_data()
        
        # Automatically generate filename if not specified
        if filename is None:
            filename = f"response_spectrum_{self.project_code}.txt"
        
        # Full path of the file
        filepath = os.path.join(self.output_dir, filename)
        
        # Save spectrum data
        data = np.column_stack((self.response_spectrum['periods'], self.response_spectrum['psa']))
        np.savetxt(filepath, data, delimiter=",", header=header, comments="# ")
        
        return filepath

    def save_csv_data(self, accel_filename: Optional[str] = None, 
                      spec_filename: Optional[str] = None) -> Tuple[str, str]:
        """
        Save the acceleration time series and the response spectrum to CSV files.
        
        Args:
            accel_filename: Filename for the acceleration data
            spec_filename: Filename for the spectrum data
            
        Returns:
            Tuple containing the full paths of the saved files (accel_path, spec_path)
        """
        self._check_data()
        
        if self.response_spectrum is None:
            self.calculate_missing_data()
            
        if accel_filename is None:
            accel_filename = f"accelerogram_{self.project_code}.csv"
        if spec_filename is None:
            spec_filename = f"response_spectrum_{self.project_code}.csv"
            
        accel_filepath = os.path.join(self.output_dir, accel_filename)
        spec_filepath = os.path.join(self.output_dir, spec_filename)
        
        # Save acceleration data
        accel_data = np.column_stack((self.time, self.acceleration))
        np.savetxt(accel_filepath, accel_data, delimiter=",", header="Time (s),Acceleration (g)", comments="")
        
        # Save spectrum data
        spec_data = np.column_stack((self.response_spectrum['periods'], self.response_spectrum['psa']))
        np.savetxt(spec_filepath, spec_data, delimiter=",", header="Period (s),Spectral Acceleration (g)", comments="")
        
        return accel_filepath, spec_filepath

    def save_json_data(self, filename: Optional[str] = None) -> str:
        """
        Save all analysis results, metadata, acceleration time series, target spectrum,
        response spectrum, and computed engineering parameters to a single structured JSON file.
        
        Args:
            filename: Name of the JSON file (automatically generated if None)
            
        Returns:
            Full path of the saved file
        """
        self._check_data()
        
        if self.response_spectrum is None:
            self.calculate_missing_data()
            
        if filename is None:
            filename = f"ground_motion_data_{self.project_code}.json"
            
        filepath = os.path.join(self.output_dir, filename)
        
        # Compute ground motion engineering parameters
        analyzer = SpectralAnalysis()
        analyzer.set_motion(self.time, self.acceleration)
        motion_params = analyzer.compute_ground_motion_parameters()
        
        # Convert numpy arrays to lists for JSON serialization
        data_to_save = {
            "project_code": self.project_code,
            "export_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "damping_ratio": self.damping,
            "metadata": {
                "duration_seconds": float(self.time[-1]),
                "time_step_dt": float(self.time[1] - self.time[0]),
                "num_points": int(len(self.time)),
                "pga_g": float(np.max(np.abs(self.acceleration)))
            },
            "tbdy_parameters": {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) 
                               for k, v in self.params.get('tbdy_params', {}).items()},
            "engineering_metrics": {
                "arias_intensity_m_s": float(motion_params["arias_intensity_m_s"]),
                "arias_intensity_cm_s": float(motion_params["arias_intensity_cm_s"]),
                "cumulative_absolute_velocity_cav_m_s": float(motion_params["cav_m_s"]),
                "significant_duration_5_95_seconds": float(motion_params["significant_duration_5_95"]),
                "significant_duration_5_75_seconds": float(motion_params["significant_duration_5_75"]),
                "peak_ground_velocity_pgv_m_s": float(motion_params["pgv_m_s"]),
                "peak_ground_displacement_pgd_m": float(motion_params["pgd_m"]),
                "pgv_pga_ratio_seconds": float(motion_params["pgv_pga_ratio"]),
                "time_of_pga_seconds": float(motion_params["t_pga"]),
                "num_zero_crossings": int(motion_params["num_zero_crossings"]),
                "mean_zero_crossing_rate_hz": float(motion_params["zero_crossing_rate"])
            },
            "time_history": {
                "time": self.time.tolist(),
                "acceleration": self.acceleration.tolist()
            },
            "response_spectra": {
                "periods": self.response_spectrum['periods'].tolist(),
                "spectral_acceleration_psa": self.response_spectrum['psa'].tolist(),
                "pseudo_spectral_velocity_psv": self.response_spectrum['psv'].tolist(),
                "spectral_displacement_sd": self.response_spectrum['sd'].tolist()
            }
        }
        
        if self.target_periods is not None and self.target_spectrum is not None:
            data_to_save["target_spectrum"] = {
                "periods": self.target_periods.tolist(),
                "spectral_acceleration": self.target_spectrum.tolist()
            }
            
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=4)
            
        return filepath

    def save_sap2000_acceleration_data(self, filename: Optional[str] = None, 
                                       two_column: bool = False,
                                       no_header: bool = False) -> str:
        """
        Save the acceleration time series in a format suitable for direct import into SAP2000 / ETABS.
        For one-column format: only acceleration values, one per line.
        For two-column format: Time and Acceleration pairs separated by space, with lines starting with '$' for comments/headers.
        
        Args:
            filename: Name of the file to be saved (automatically generated if None)
            two_column: If True, saves in two-column format (Time, Acceleration) separated by spaces.
                        If False (default), saves in one-column format (only Acceleration values).
            no_header: If True, outputs absolutely no comments/header lines.
                        
        Returns:
            Full path of the saved file
        """
        self._check_data()
        
        if filename is None:
            suffix = "2col" if two_column else "1col"
            filename = f"sap2000_accelerogram_{suffix}_{self.project_code}.txt"
            
        filepath = os.path.join(self.output_dir, filename)
        
        if two_column:
            with open(filepath, 'w', encoding='utf-8') as f:
                if not no_header:
                    f.write(f"$ SAP2000 / ETABS Importable Time-History Function\n")
                    f.write(f"$ Project: {self.project_code}\n")
                    f.write(f"$ Number of points: {len(self.time)}\n")
                    f.write(f"$ Time step (dt): {self.time[1] - self.time[0]:.6f} s\n")
                    f.write(f"$ Time (s)   Acceleration (g)\n")
                for t, a in zip(self.time, self.acceleration):
                    f.write(f"{t:12.6f} {a:16.8e}\n")
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                if not no_header:
                    f.write(f"$ SAP2000 / ETABS Importable Time-History Function (Equal Interval Format)\n")
                    f.write(f"$ Project: {self.project_code}\n")
                    f.write(f"$ Number of points: {len(self.time)}\n")
                    f.write(f"$ Time step (dt): {self.time[1] - self.time[0]:.6f} s\n")
                    f.write(f"$ NOTE: Import this file in SAP2000 with 'Values at Equal Intervals of' = {self.time[1] - self.time[0]:.6f}\n")
                for a in self.acceleration:
                    f.write(f"{a:16.8e}\n")
                    
        return filepath
    
    def plot_acceleration_time_history(self, figsize: Tuple[int, int] = (10, 6), 
                                     save_fig: bool = True,
                                     show_fig: bool = False,
                                     filename: Optional[str] = None) -> plt.Figure:
        """
        Plot the acceleration time series.
        
        Args:
            figsize: Figure size
            save_fig: Whether to save the figure
            show_fig: Whether to show the figure
            filename: Name of the file to be saved (automatically generated if None)
            
        Returns:
            Matplotlib Figure object
        """
        self._check_data()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot the acceleration time series
        ax.plot(self.time, self.acceleration, color=BLUE, linewidth=1.5)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Acceleration (g)')
        ax.set_title('Acceleration Time Series')
        
        # Show PGA value
        pga = np.max(np.abs(self.acceleration))
        text_str = f"PGA = {pga:.4f}g"
        props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
        ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=props)
        
        # Set axis limits
        ax.set_xlim(left=0)
        
        # Make grid thin
        ax.grid(True, linestyle='--', alpha=0.7, linewidth=0.5)
        
        plt.tight_layout()
        
        # Save the figure
        if save_fig:
            if filename is None:
                filename = f"acceleration_time_history_{self.project_code}.png"
            
            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
        
        # Show the figure
        if show_fig:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_spectrum_comparison(self, figsize: Tuple[int, int] = (10, 6), 
                               log_scale: bool = True,
                               save_fig: bool = True,
                               show_fig: bool = False,
                               filename: Optional[str] = None) -> plt.Figure:
        """
        Plot the design spectrum and response spectrum for comparison.
        
        Args:
            figsize: Figure size
            log_scale: Whether to use logarithmic scale
            save_fig: Whether to save the figure
            show_fig: Whether to show the figure
            filename: Name of the file to be saved (automatically generated if None)
            
        Returns:
            Matplotlib Figure object
        """
        self._check_data()
        
        # Calculate missing data
        if self.response_spectrum is None:
            self.calculate_missing_data()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot the response spectrum
        ax.plot(self.response_spectrum['periods'], self.response_spectrum['psa'], 
                color=BLUE, linewidth=2, label='Response Spectrum')
        
        # Plot the target spectrum if available
        if self.target_periods is not None and self.target_spectrum is not None:
            ax.plot(self.target_periods, self.target_spectrum, 
                    color=RED, linestyle='--', linewidth=2, label='Target Spectrum')
        
        if log_scale:
            ax.set_xscale('log')
            # Only x-axis is logarithmic, y is normal
        
        ax.set_xlabel('Period, T (s)')
        ax.set_ylabel('Spectral Acceleration, Sa(T) (g)')
        ax.set_title('Spectrum Comparison')
        
        # Make grid thin
        ax.grid(True, which='both', linestyle='--', alpha=0.7, linewidth=0.5)
        
        # Set axis limits
        ax.set_xlim(min(self.target_periods), max(self.target_periods))
        
        # Beautify the legend
        leg = ax.legend(frameon=True, loc='best', framealpha=0.9)
        leg.get_frame().set_edgecolor('gray')
        
        plt.tight_layout()
        
        # Save the figure
        if save_fig:
            if filename is None:
                scale_str = "loglog" if log_scale else "linear"
                filename = f"spectrum_comparison_{scale_str}_{self.project_code}.png"
            
            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
        
        # Show the figure
        if show_fig:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_psd(self, figsize: Tuple[int, int] = (10, 6), 
               log_scale: bool = True,
               save_fig: bool = True,
               show_fig: bool = False,
               filename: Optional[str] = None) -> plt.Figure:
        """
        Plot the power spectral density (PSD).
        
        Args:
            figsize: Figure size
            log_scale: Whether to use logarithmic scale
            save_fig: Whether to save the figure
            show_fig: Whether to show the figure
            filename: Name of the file to be saved (automatically generated if None)
            
        Returns:
            Matplotlib Figure object
        """
        self._check_data()
        
        # Calculate missing data
        if self.frequencies is None or self.psd is None:
            self.calculate_missing_data()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot the PSD
        if log_scale:
            ax.semilogy(self.frequencies, self.psd, color=BLUE, linewidth=1.5)
        else:
            ax.plot(self.frequencies, self.psd, color=BLUE, linewidth=1.5)
        
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD ((g)²/Hz)')
        ax.set_title('Power Spectral Density')
        
        # Make grid thin
        ax.grid(True, which='both', linestyle='--', alpha=0.7, linewidth=0.5)
        
        # Set axis limits
        ax.set_xlim(0, min(25, max(self.frequencies)))
        
        plt.tight_layout()
        
        # Save the figure
        if save_fig:
            if filename is None:
                scale_str = "log" if log_scale else "linear"
                filename = f"psd_{scale_str}_{self.project_code}.png"
            
            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
        
        # Show the figure
        if show_fig:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_spectrogram(self, figsize: Tuple[int, int] = (10, 6), 
                       nperseg: int = 256, 
                       noverlap: int = 128,
                       save_fig: bool = True,
                       show_fig: bool = False,
                       filename: Optional[str] = None) -> plt.Figure:
        """
        Plot the spectrogram.
        
        Args:
            figsize: Figure size
            nperseg: Number of points in each segment
            noverlap: Number of overlapping points
            save_fig: Whether to save the figure
            show_fig: Whether to show the figure
            filename: Name of the file to be saved (automatically generated if None)
            
        Returns:
            Matplotlib Figure object
        """
        self._check_data()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate and plot the spectrogram
        fs = 1 / (self.time[1] - self.time[0])  # Sampling frequency
        f, t, Sxx = signal.spectrogram(self.acceleration, fs, nperseg=nperseg, noverlap=noverlap)
        
        # Power density in dB
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        
        # Plot the spectrogram - colors like 'viridis' or 'inferno' are more suitable for academic publications
        pcm = ax.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='viridis')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title('Spectrogram')
        
        # Improve appearance
        ax.set_ylim(0, min(25, max(f)))  # Limit the upper frequency to a reasonable value
        
        # Add and beautify color bar
        cbar = fig.colorbar(pcm, ax=ax)
        cbar.set_label('Power Density (dB/Hz)')
        
        plt.tight_layout()
        
        # Save the figure
        if save_fig:
            if filename is None:
                filename = f"spectrogram_{self.project_code}.png"
            
            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
        
        # Show the figure
        if show_fig:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def generate_report(self, filename: Optional[str] = None) -> str:
        """
        Generate a summary report.
        
        Args:
            filename: Name of the file to be saved (automatically generated if None)
            
        Returns:
            Full path of the saved file
        """
        self._check_data()
        
        # Calculate missing data
        if self.response_spectrum is None:
            self.calculate_missing_data()
        
        # Automatically generate filename if not specified
        if filename is None:
            filename = f"report_{self.project_code}.txt"
        
        # Full path of the file
        filepath = os.path.join(self.output_dir, filename)
        
        # Summary statistics
        pga = np.max(np.abs(self.acceleration))
        duration = self.time[-1]
        dt = self.time[1] - self.time[0]
        num_points = len(self.time)
        
        # Compute ground motion engineering parameters (Arias, CAV, Significant Duration, PGV, PGD, Nonstationarity)
        analyzer = SpectralAnalysis()
        analyzer.set_motion(self.time, self.acceleration)
        motion_params = analyzer.compute_ground_motion_parameters()
        
        # TBDY parameters
        tbdy_params = self.params.get('tbdy_params', {})
        
        # Create the report
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"ARTIFICIAL GROUND MOTION REPORT (Project: {self.project_code})\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("1. ACCELERATION TIME SERIES INFORMATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Duration: {duration:.2f} s\n")
            f.write(f"Time Step: {dt:.5f} s\n")
            f.write(f"Number of Data Points: {num_points}\n")
            f.write(f"PGA (from acceleration): {pga:.4f} g ({pga * 9.81:.4f} m/s²)\n\n")
            
            f.write("2. ENGINEERING GROUND MOTION PARAMETERS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Significant Duration (D5-95): {motion_params['significant_duration_5_95']:.3f} s\n")
            f.write(f"Significant Duration (D5-75): {motion_params['significant_duration_5_75']:.3f} s\n")
            f.write(f"Arias Intensity (Ia): {motion_params['arias_intensity_m_s']:.4f} m/s ({motion_params['arias_intensity_cm_s']:.2f} cm/s)\n")
            f.write(f"Cumulative Absolute Velocity (CAV): {motion_params['cav_m_s']:.4f} m/s\n")
            f.write(f"Peak Ground Velocity (PGV): {motion_params['pgv_m_s']:.4f} m/s\n")
            f.write(f"Peak Ground Displacement (PGD): {motion_params['pgd_m']:.4f} m\n")
            f.write(f"PGV / PGA Ratio: {motion_params['pgv_pga_ratio']:.4f} s\n\n")
            
            f.write("3. NONSTATIONARITY AND SIGNAL CHARACTERISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Time of Peak Acceleration (t_PGA): {motion_params['t_pga']:.3f} s\n")
            f.write(f"Total Zero Crossings: {motion_params['num_zero_crossings']}\n")
            f.write(f"Mean Zero-Crossing Rate: {motion_params['zero_crossing_rate']:.2f} Hz\n\n")
            
            if tbdy_params:
                f.write("4. TBDY 2018 SPECTRUM PARAMETERS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Ss: {tbdy_params.get('Ss', 'N/A')}\n")
                f.write(f"S1: {tbdy_params.get('S1', 'N/A')}\n")
                f.write(f"Soil Class: {tbdy_params.get('soil_class', 'N/A')}\n")
                f.write(f"Fs: {tbdy_params.get('Fs', 'N/A')}\n")
                f.write(f"F1: {tbdy_params.get('F1', 'N/A')}\n")
                f.write(f"SDS: {tbdy_params.get('SDS', 'N/A')}\n")
                f.write(f"SD1: {tbdy_params.get('SD1', 'N/A')}\n")
                f.write(f"TA: {tbdy_params.get('TA', 'N/A')}\n")
                f.write(f"TB: {tbdy_params.get('TB', 'N/A')}\n\n")
            
            f.write("5. GENERATED FILES\n")
            f.write("-" * 40 + "\n")
            f.write(f"Directory: {os.path.abspath(self.output_dir)}\n\n")
            
            # List of files generated so far
            output_files = [file for file in os.listdir(self.output_dir) if self.project_code in file]
            for file in output_files:
                f.write(f"- {file}\n")
            
            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write(f"Report Creation Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n")
        
        return filepath
    
    def generate_all_outputs(self, show_figs: bool = False,
                             export_csv: bool = True,
                             export_json: bool = True,
                             export_sap2000: bool = True,
                             sap2000_two_column: bool = False) -> Dict[str, str]:
        """
        Generate all outputs.
        
        Args:
            show_figs: Whether to show the figures
            export_csv: Whether to save in CSV format
            export_json: Whether to save in JSON format
            export_sap2000: Whether to save in SAP2000 format
            sap2000_two_column: Whether SAP2000 format uses two columns
            
        Returns:
            Dictionary containing paths of generated files
        """
        self._check_data()
        
        # Calculate missing data
        self.calculate_missing_data()
        
        # Dictionary to hold file paths
        output_files = {}
        
        # Acceleration time series data (Standard TXT)
        output_files['acceleration_data'] = self.save_acceleration_data()
        
        # Response spectrum data (Standard TXT)
        output_files['spectrum_data'] = self.save_spectrum_data()
        
        # CSV exports if requested
        if export_csv:
            csv_accel, csv_spec = self.save_csv_data()
            output_files['acceleration_csv'] = csv_accel
            output_files['spectrum_csv'] = csv_spec
            
        # JSON export if requested
        if export_json:
            output_files['json_data'] = self.save_json_data()
            
        # SAP2000 export if requested
        if export_sap2000:
            output_files['sap2000_data'] = self.save_sap2000_acceleration_data(two_column=sap2000_two_column)
        
        # Acceleration time series plot
        self.plot_acceleration_time_history(show_fig=show_figs)
        
        # Spectrum comparison plots (log-log and linear)
        self.plot_spectrum_comparison(log_scale=True, show_fig=show_figs)
        self.plot_spectrum_comparison(log_scale=False, show_fig=show_figs)
        
        # PSD plot
        self.plot_psd(log_scale=True, show_fig=show_figs)
        
        # Spectrogram plot
        self.plot_spectrogram(show_fig=show_figs)
        
        # Summary report
        output_files['report'] = self.generate_report()
        
        return output_files