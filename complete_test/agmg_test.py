#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple usage example for Artificial Ground Motion Generator.
"""

import multiprocessing
from design_spectrum import TBDYSpectrum
from initial_motion import InitialMotionGenerator
from envelope import EnvelopeFunction
from spectral_analysis import SpectralAnalysis
from spectral_matching import SpectralMatcher
from visualization import OutputGenerator
from target_pga import PGAEstimator
import subprocess
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import os
from scipy import signal
import numpy as np
import traceback
import time


def generate_motion_TDBY(Ss, S1, soil_class, max_iterations=10, max_non_improvement=2, duration=40.0, dt=0.01, pga_='tdby', omega_g=15.0, zeta_g=0.6, name='default', imageoutput=False):
    """Generate acceleration time series for TBDY
    Args:
        Ss (float): Ss value
        S1 (float): S1 value
        soil_class (str): Soil class
        max_iterations (int): Maximum iterations for spectral matching
        max_non_improvement (int): Maximum iterations without improvement
        duration (float): Duration (seconds)
        dt (float): Time step (seconds)
        pga_ (string): PGA function name (tdby, campbell, graizer) or just use a value it will be converted
        omega_g (float): Natural frequency (rad/s)
        zeta_g (float): Damping ratio (%)
        name (str): Output directory name
        imageoutput (bool): Image output generation
    """
    
    # Create output directory only if image output is requested
    output_dir = None
    if imageoutput:
        output_dir = name+'_outputs'
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


def single_test(args):
    """Wrapper function for running a single test with error handling
    
    Args:
        args: A tuple containing (Ss, S1, soil_class, repeat_index)
        
    Returns:
        A tuple containing (Ss, S1, soil_class, repeat_index, best_avg_error, best_max_error)
        or (Ss, S1, soil_class, repeat_index, None, None) if an error occurred
    """
    Ss, S1, soil_class, repeat_index = args
    try:
        # Create a name for logging purposes only
        name = f"{soil_class}_Ss{Ss:.2f}_S1{S1:.2f}_run{repeat_index}"
        
        print(f"Running test: {name}")
        start_time = time.time()
        
        best_avg_error, best_max_error = generate_motion_TDBY(
            Ss, S1, soil_class, 
            max_iterations=20, 
            max_non_improvement=2, 
            duration=40.0, 
            dt=0.01, 
            pga_='tdby', 
            omega_g=15.0, 
            zeta_g=0.6, 
            name=name,  # This is only used if imageoutput=True
            imageoutput=False  # Set to False to avoid creating folders
        )
        
        elapsed_time = time.time() - start_time
        print(f"Completed test: {name} in {elapsed_time:.2f} seconds")
        
        return (Ss, S1, soil_class, repeat_index, best_avg_error, best_max_error, elapsed_time)
    except Exception as e:
        print(f"Error in test {Ss}-{S1}-{soil_class}-{repeat_index}: {str(e)}")
        traceback.print_exc()
        return (Ss, S1, soil_class, repeat_index, None, None, None)


def run_tests_parallel(subprocess_count=10, Ss_upper=2.5, Ss_lower=0.25, Ss_number=11, 
                       S1_upper=0.8, S1_lower=0.1, S1_number=8, repeat_times=5):
    """Run tests in parallel for various parameter combinations
    
    Args:
        subprocess_count (int): Number of processes to use
        Ss_upper (float): Upper bound for Ss values
        Ss_lower (float): Lower bound for Ss values
        Ss_number (int): Number of Ss values to test
        S1_upper (float): Upper bound for S1 values
        S1_lower (float): Lower bound for S1 values
        S1_number (int): Number of S1 values to test
        repeat_times (int): Number of times to repeat each parameter combination
    """
    print(f"Starting parallel tests with {subprocess_count} processes")
    print(f"Parameter ranges: Ss=[{Ss_lower}-{Ss_upper}], S1=[{S1_lower}-{S1_upper}]")
    print(f"Soil classes: ZA, ZB, ZC, ZD, ZE")
    print(f"Each combination will be repeated {repeat_times} times")
    
    start_time = time.time()
    
    # Generate parameter ranges
    Ss_range = np.linspace(Ss_lower, Ss_upper, Ss_number)
    S1_range = np.linspace(S1_lower, S1_upper, S1_number)
    soil_class_range = ['ZA', 'ZB', 'ZC', 'ZD', 'ZE']

    # Create a list of arguments for each test including repeat index
    args = []
    for Ss in Ss_range:
        for S1 in S1_range:
            for soil_class in soil_class_range:
                for repeat in range(repeat_times):
                    args.append((Ss, S1, soil_class, repeat))

    total_tests = len(args)
    print(f"Total test cases: {total_tests}")
    
    # Run tests in parallel
    with multiprocessing.Pool(processes=subprocess_count) as pool:
        results = []
        for i, result in enumerate(pool.imap_unordered(single_test, args), 1):
            results.append(result)
            print(f"Progress: {i}/{total_tests} tests completed", end='\r')
    
    print()  # Move to the next line after progress output

    # Filter out failed tests and extract results
    valid_results = [r for r in results if r[4] is not None]
    failed_count = len(results) - len(valid_results)
    
    if failed_count > 0:
        print(f"Warning: {failed_count} tests failed out of {total_tests}")
    
    # Create DataFrame from results
    df = pd.DataFrame(valid_results, 
                     columns=['Ss', 'S1', 'soil_class', 'repeat', 'best_avg_error', 'best_max_error', 'runtime'])
    
    # Save results to CSV
    output_file = 'test_results.csv'
    df.to_csv(output_file, index=False)
    
    elapsed_time = time.time() - start_time
    print(f"All tests completed in {elapsed_time:.2f} seconds")
    print(f"Results saved to {output_file}")
    
    return df



def create_test_results_csv():
    # Load the test results
    df = pd.read_csv('test_results.csv')
    
    # 1) Plot: x: S1 values, y: avg errors, columns by soil class
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='S1', y='best_avg_error', hue='soil_class', ci='sd')
    plt.title('Average Error Mean by S1 Value and Soil Class')
    plt.xlabel('S1 Value')
    plt.ylabel('Average Error Mean')
    plt.legend(title='Soil Class')
    plt.grid(True)
    plt.savefig('avg_error_by_S1_and_soil_class.png', dpi=300, bbox_inches='tight')
    
    # 2) Plot: x: Ss values, y: avg errors, columns by soil class
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Ss', y='best_avg_error', hue='soil_class', ci='sd')
    plt.title('Average Error Mean by Ss Value and Soil Class')
    plt.xlabel('Ss Value')
    plt.ylabel('Average Error Mean')
    plt.legend(title='Soil Class')
    plt.grid(True)
    plt.savefig('avg_error_by_Ss_and_soil_class.png', dpi=300, bbox_inches='tight')
    
    # 3) Plot: x: soil classes, y: avg errors
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='soil_class', y='best_avg_error', ci='sd')
    plt.title('Average Error Mean by Soil Class')
    plt.xlabel('Soil Class')
    plt.ylabel('Average Error Mean')
    plt.grid(True)
    plt.savefig('avg_error_by_soil_class.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    # Run the parallel tests with default parameters
    #run_tests_parallel(
    #    subprocess_count=60, 
    #    Ss_upper=2.5, 
    #    Ss_lower=0.25, 
    #    Ss_number=10, 
    #    S1_upper=0.8, 
    #    S1_lower=0.1, 
    #    S1_number=10,
    #    repeat_times=5
    #)

    create_test_results_csv()
