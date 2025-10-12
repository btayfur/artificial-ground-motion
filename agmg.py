#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple usage example for Artificial Ground Motion Generator.
"""

from design_spectrum import TBDYSpectrum
from initial_motion import InitialMotionGenerator
from envelope import EnvelopeFunction
from spectral_analysis import SpectralAnalysis
from spectral_matching import SpectralMatcher
from visualization import OutputGenerator
from target_pga import PGAEstimator

import matplotlib.pyplot as plt
import os
from scipy import signal
import numpy as np


def generate_motion_TDBY(Ss, S1, soil_class, max_iterations=10, max_non_improvement=2, duration=40.0, dt=0.01, pga_='tdby', omega_g=15.0, zeta_g=0.6, name='default', imageoutput=False):
    """Generate acceleration time series for TBDY
    Args:
        Ss (float): Ss value
        S1 (float): S1 value
        soil_class (str): Soil class
        duration (float): Duration (seconds)
        dt (float): Time step (seconds)
        pga (string): PGA function name (tdby, campbell, graizer) or just use a value it will be converted
        omega_g (float): Natural frequency (rad/s) 15 is default, but it should be changed when unintented peaks occurs.
        zeta_g (float): Damping ratio (%)
        name (str): Output directory name
        imageoutput (bool): Image output generation
    """
    
    # Create output directory
    output_dir = name+'_outputs2'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # -------------------------------------------------------
    # 1. Generate target design spectrum (TBDY 2018)
    # -------------------------------------------------------    
    # Create spectrum object
    spectrum = TBDYSpectrum()
    
    # Calculate design spectrum
    periods, target_spectrum = spectrum.generate_spectrum(
        Ss=Ss,
        S1=S1,
        soil_class=soil_class,
        min_period=0.001,
        max_period=8.0,
        num_points=300,
        log_scale=True
    )
    
    if imageoutput:
        # Plot spectrum
        fig1 = spectrum.plot_spectrum(log_scale=True)
        fig1.savefig(os.path.join(output_dir, "0_target_spectrum.png"), dpi=300, bbox_inches='tight')
    
    # -------------------------------------------------------
    # 2. Generate initial acceleration time series
    # -------------------------------------------------------
    pga=0.5
    # Generate initial acceleration time series
    # Maximum PGA can be estimated with pre-defined methods, but its just a beginning value. Algorithm could change that value in iterative process.
    #pga=PGAEstimator.sabetta_pugliese(6.5,20,'stiff') 
    if pga_ == 'tdby':
        pga=PGAEstimator.estimate_pga_for_tdby(Ss, S1, soil_class)
    else:
        pga=float(pga_)

    motion_gen = InitialMotionGenerator()
    t, accel = motion_gen.generate_clough_penzien_motion(
        duration=duration,
        dt=dt,
        omega_g=omega_g,
        zeta_g=zeta_g,
        omega_f=0.1,
        zeta_f=0.6,
        target_pga=pga,
        #seed=42  # For reproducibility (for test you can use seed, but if you are searching for a good motion, try different seeds or let it be None)
    )

    # -------------------------------------------------------
    # Apply lowpass filter after signal generation
    # Pass frequencies above 5x10^(-2) period (pass low frequencies)
    # -------------------------------------------------------
    cutoff_period = 2.5e-2  # Cutoff period (seconds)
    cutoff_freq = 1.0 / cutoff_period  # Cutoff frequency (Hz)
    
    # Nyquist frequency
    nyquist = 0.5 / dt
    
    # Butterworth lowpass filter design
    filter_order = 4  # Filter order
    norm_cutoff = cutoff_freq / nyquist  # Normalized cutoff frequency
    
    # Calculate filter coefficients
    b, a = signal.butter(filter_order, norm_cutoff, btype='lowpass')
    
    # Filtering (use filtfilt for zero phase shift)
    accel_filtered = signal.filtfilt(b, a, accel)
    
    if imageoutput:
        # Plot filtered acceleration time series
        plt.figure(figsize=(10, 6))
        plt.plot(t, accel, 'b-', alpha=0.5, label='Original Acceleration')
        plt.plot(t, accel_filtered, 'r-', label='Filtered Acceleration')
        plt.title(f"Lowpass Filter Applied Acceleration (T > {cutoff_period} s, f < {cutoff_freq} Hz)")
        plt.xlabel("Time (s)")
        plt.ylabel("Acceleration (g)")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(output_dir, "0_lowpass_filtered_motion.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Use filtered data
    accel = accel_filtered
    
    if imageoutput:
        # Plot acceleration time series
        fig2 = plt.figure(figsize=(10, 6))
        plt.plot(t, accel)
        plt.title("Filtered Acceleration Time Series")
        plt.xlabel("Time (s)")
        plt.ylabel("Acceleration (g)")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "0_initial_motion.png"), dpi=300, bbox_inches='tight')
        plt.close(fig2)
    
    # Calculate PSD and plot
    fs = 1 / dt  # Sampling frequency
    f, Pxx = signal.welch(accel, fs, nperseg=1024, scaling='density')
    
    if imageoutput:
        # Plot PSD
        fig3 = plt.figure(figsize=(10, 6))
        plt.semilogy(f, Pxx)
        plt.title("Filtered Signal Power Spectral Density")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("PSD (g²/Hz)")
        plt.grid(True)
        plt.axvline(x=cutoff_freq, color='r', linestyle='--', label=f'Cutoff Frequency ({cutoff_freq} Hz)')
        plt.legend()
        plt.savefig(os.path.join(output_dir, "0_initial_psd.png"), dpi=300, bbox_inches='tight')
        plt.close(fig3)
    
    # -------------------------------------------------------
    # 3. Apply envelope function
    # -------------------------------------------------------
    envelope = EnvelopeFunction()
    envelope.set_motion(t, accel)
    
    # Select and create envelope
    envelope.create_custom_envelope(t_rise_ratio=0.15, t_steady_ratio=0.05, decay_factor=0.8)
    #envelope.create_saragoni_hart_envelope(alpha=1, beta=0.5, theta=0.5)
    
    
    # Apply envelope
    t, accel_enveloped = envelope.apply_envelope()

    if imageoutput:
        # Plot envelope and acceleration time series
        fig4 = envelope.plot_envelope()
        fig4.savefig(os.path.join(output_dir, "0_envelope.png"), dpi=300, bbox_inches='tight')
    
    # -------------------------------------------------------
    # 4. Spectral analysis and baseline correction
    # -------------------------------------------------------
    spec_analysis = SpectralAnalysis()
    spec_analysis.set_motion(t, accel_enveloped)
    
    # Apply baseline correction
    spec_analysis.apply_baseline_correction(cutoff_freq=0.1)
    
    # Calculate velocity and displacement
    spec_analysis.compute_velocity_displacement()
    
    if imageoutput:
        # Plot acceleration, velocity and displacement
        fig5 = spec_analysis.plot_motion()
        fig5.savefig(os.path.join(output_dir, "0_motion_after_baseline.png"), dpi=300, bbox_inches='tight')
    
    # Compute response spectrum
    spec_analysis.compute_response_spectrum(periods=periods, damping=0.05)
    
    if imageoutput:
        # Plot response spectrum
        fig6 = spec_analysis.plot_response_spectrum(log_scale=True)
        fig6.savefig(os.path.join(output_dir, "0_initial_response_spectrum.png"), dpi=300, bbox_inches='tight')
    
    # -------------------------------------------------------
    # 5. Iterative spectral matching
    # -------------------------------------------------------
    matcher = SpectralMatcher()
    matcher.set_motion(t, accel_enveloped)
    matcher.set_target_spectrum(periods, target_spectrum)
    
    # Spectral matching
    results = matcher.match_spectrum(
        max_iterations=max_iterations,
        tolerance_avg=0.075,
        tolerance_max=0.15,
        damping=0.05,
        freq_range=(0.1, 25.0),
        max_ratio=15.0,          # Maximum fixing ratio
        min_ratio=0.00005,          # Minimum fixing ratio
        avg_weight=0.95,         # Average error weight (more important than maximum error)
        max_weight=0.05,         # Maximum error weight
        max_no_improvement=max_non_improvement    # Stop after fewer iterations
    )
    
    if imageoutput:
        # Plot spectrum comparison
        fig7 = matcher.plot_spectra_comparison(log_scale=True)
        fig7.savefig(os.path.join(output_dir, "spectrum_comparison.png"), dpi=300, bbox_inches='tight')
        # Plot acceleration comparison
        fig8 = matcher.plot_acceleration_comparison()
        fig8.savefig(os.path.join(output_dir, "acceleration_comparison.png"), dpi=300, bbox_inches='tight')
        # Plot convergence graph
        fig9 = matcher.plot_error_convergence()
        fig9.savefig(os.path.join(output_dir, "convergence.png"), dpi=300, bbox_inches='tight')

    if imageoutput:
        # Result statistics
        print("Spectral Matching Results:")
        print(f"Completed Iterations: {len(matcher.iteration_results)}")
        print(f"Final Average Error: {matcher.iteration_results[-1]['avg_error']:.4f}")
        print(f"Final Maximum Error: {matcher.iteration_results[-1]['max_error']:.4f}")
    
    # -------------------------------------------------------
    # 6. Output generation
    # -------------------------------------------------------
    # Get TBDY parameters
    tbdy_params = spectrum.get_spectrum_data()['parameters']
    
    if imageoutput:
        # Output generator
        output_gen = OutputGenerator(output_dir=output_dir)
        output_gen.set_data(
            time=t,
            acceleration=results['acceleration_matched'],
            target_periods=periods,
            target_spectrum=target_spectrum,
            response_spectrum=results['matched_spectrum'],
            params={'tbdy_params': tbdy_params}
        )
    
        # Create all outputs
        output_files = output_gen.generate_all_outputs(show_figs=False)
    
        print(f"\nOutputs saved to '{os.path.abspath(output_dir)}' directory.")
        print(f"Report: {os.path.basename(output_files['report'])}")

    return results['best_avg_error'], results['best_max_error']




best_avg_error, best_max_error = generate_motion_TDBY(Ss=1.3, S1=0.4, soil_class='ZB', max_iterations=10, max_non_improvement=5, duration=40.0, dt=0.01, pga_="tdby", omega_g=15.0, zeta_g=0.6, name='example', imageoutput=True)
print(f"Best Average Error: {best_avg_error:.4f}")
print(f"Best Maximum Error: {best_max_error:.4f}")

def test_function(Ss_upper, Ss_lower, Ss_number, S1_upper, S1_lower, S1_number, repeat_times):
    ''' Test framework for range of parameters
    test for all Ss, s1 and soil class values and save all combinations to a csv file.'''
    Ss_range = np.linspace(Ss_lower, Ss_upper, Ss_number)
    S1_range = np.linspace(S1_lower, S1_upper, S1_number)
    soil_class_range = ['ZA','ZB', 'ZC', 'ZD', 'ZE']
    import csv

    total_repeat_times = Ss_number * S1_number * len(soil_class_range) * repeat_times
    total_completed_repeat_times = 0

    with open('_test_results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Ss', 'S1', 'soil_class', 'repeat', 'best_avg_error', 'best_max_error'])

        for Ss in Ss_range:
            for S1 in S1_range:
                for soil_class in soil_class_range:
                    for i in range(repeat_times):
                        best_avg_error, best_max_error = generate_motion_TDBY(Ss, S1, soil_class, max_iterations=20, max_non_improvement=2, duration=40.0, dt=0.01, pga_='tdby', omega_g=15.0, zeta_g=0.6, name='example', imageoutput=False)
                        writer.writerow([Ss, S1, soil_class, i, best_avg_error, best_max_error])
                        file.flush()
                        total_completed_repeat_times += 1
                        #clear console
                        os.system('cls' if os.name == 'nt' else 'clear')
                        print(f"Completed {total_completed_repeat_times} of {total_repeat_times} repeat times")

    print("Test results saved to '_test_results.csv'")

#test_function(Ss_upper=2.5, Ss_lower=0.25, Ss_number=11, S1_upper=0.8, S1_lower=0.1, S1_number=8, repeat_times=5)
