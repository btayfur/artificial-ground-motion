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


def generate_motion_TDBY(Ss, S1, soil_class, omega_g, zeta_g, omega_f, zeta_f, t_rise_ratio, t_steady_ratio, decay_factor, max_iterations=2, max_non_improvement=1, duration=40.0, dt=0.01, pga_='tdby', name='default', imageoutput=False):
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
        omega_f=omega_f,
        zeta_f=zeta_f,
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
    envelope.create_custom_envelope(t_rise_ratio, t_steady_ratio, decay_factor)
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
        tolerance_avg=0.01,
        tolerance_max=0.05,
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

    objective_function= results['best_avg_error']*0.5+results['best_max_error']*0.5

    return objective_function




def TeachingLearningBasedOptimization(population_size, limits):
    '''
    TLBO algorithm is used to optimize the parameters of the ground motion generator. Aim is achieving aggresively matching the target spectrum.
    input parameters:
        population_size: number of individuals in the population
        limits: list of limits        
    '''
    import csv
    import numpy as np

    # Algorithm parameters
    max_iterations = 50
    dimension = len(limits)
    
    # Initialize best solution
    best_solution = None
    best_fitness = float('inf')
    
    # Store fitness values for each individual
    fitness_values = []

    # Generate initial population randomly within the given limits
    population = []
    for _ in range(population_size):
        individual = [np.random.uniform(low, high) for low, high in limits]
        population.append(individual)
    
    population = np.array(population)  # Convert to numpy array for better operations

    # Open CSV file for writing fitness data
    with open('fitness_data.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        header = ['Iteration', 'Individual'] + [f'Parameter{i+1}' for i in range(dimension)] + ['Fitness']
        writer.writerow(header)
    
        # Main TLBO loop
        for iteration in range(max_iterations):
            # Teacher Phase
            # Find the teacher (best solution)
            current_fitness = []
            for i in range(population_size):
                # Calculate fitness for each individual
                fitness = generate_motion_TDBY(
                    Ss=0.781,
                    S1=0.217,
                    soil_class='ZC',
                    omega_g=population[i][0],
                    zeta_g=population[i][1],
                    omega_f=population[i][2],
                    zeta_f=population[i][3],
                    t_rise_ratio=population[i][4],
                    t_steady_ratio=population[i][5],
                    decay_factor=population[i][6]
                )
                current_fitness.append(fitness)
                
                # Write fitness data to CSV - fixed list concatenation
                row_data = [iteration + 1, i + 1] + population[i].tolist() + [fitness]
                writer.writerow(row_data)
                
                # Update best solution
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_solution = population[i].copy()
            
            # Calculate mean of the population
            mean_population = np.mean(population, axis=0)
            
            # Teaching factor (TF) - randomly chosen as 1 or 2
            TF = np.random.choice([1, 2])
            
            # Update each individual based on teacher
            for i in range(population_size):
                # Calculate difference between teacher and mean
                difference = best_solution - TF * mean_population
                
                # Generate new solution
                new_solution = population[i] + np.random.random() * difference
                
                # Ensure bounds
                for j in range(dimension):
                    new_solution[j] = np.clip(new_solution[j], limits[j][0], limits[j][1])
                
                # Calculate fitness of new solution
                new_fitness = generate_motion_TDBY(
                    Ss=0.781,
                    S1=0.217,
                    soil_class='ZC',
                    omega_g=new_solution[0],
                    zeta_g=new_solution[1],
                    omega_f=new_solution[2],
                    zeta_f=new_solution[3],
                    t_rise_ratio=new_solution[4],
                    t_steady_ratio=new_solution[5],
                    decay_factor=new_solution[6]
                )
                
                # Accept if better
                if new_fitness < current_fitness[i]:
                    population[i] = new_solution
                    current_fitness[i] = new_fitness
            
            # Learner Phase
            for i in range(population_size):
                # Select random partner
                partner_idx = np.random.randint(0, population_size)
                while partner_idx == i:
                    partner_idx = np.random.randint(0, population_size)
                
                # Update based on partner
                if current_fitness[i] < current_fitness[partner_idx]:
                    difference = population[i] - population[partner_idx]
                else:
                    difference = population[partner_idx] - population[i]
                
                # Generate new solution
                new_solution = population[i] + np.random.random() * difference
                
                # Ensure bounds
                for j in range(dimension):
                    new_solution[j] = np.clip(new_solution[j], limits[j][0], limits[j][1])
                
                # Calculate fitness of new solution
                new_fitness = generate_motion_TDBY(
                    Ss=0.781,
                    S1=0.217,
                    soil_class='ZC',
                    omega_g=new_solution[0],
                    zeta_g=new_solution[1],
                    omega_f=new_solution[2],
                    zeta_f=new_solution[3],
                    t_rise_ratio=new_solution[4],
                    t_steady_ratio=new_solution[5],
                    decay_factor=new_solution[6]
                )
                
                # Accept if better
                if new_fitness < current_fitness[i]:
                    population[i] = new_solution
                    current_fitness[i] = new_fitness
                    
                    # Update best solution if needed
                    if new_fitness < best_fitness:
                        best_fitness = new_fitness
                        best_solution = new_solution.copy()
            
            # Store best fitness for this iteration
            fitness_values.append(best_fitness)
            
            # Print progress
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{max_iterations}, Best Fitness: {best_fitness:.6f}")
    
    return best_solution, best_fitness, fitness_values


if __name__ == "__main__":
    # Initialize population
    population_size = 3
    limits = [
    [0.1, 15], # omega_g: reasonable range for natural frequency of ground motion
    [0.01, 0.7], # zeta_g: reasonable range for damping ratio of ground motion
    [0.1, 15], # omega_f: reasonable range for natural frequency of the filter
    [0.01, 0.7], # zeta_f: reasonable range for damping ratio of the filter
    [0, 1], # t_rise_ratio  // will be normalized with following parameters
    [0, 1], # t_steady_ratio
    [0, 1], # decay_factor 
    ]    

    # Run TLBO
    best_solution, best_fitness, fitness_values = TeachingLearningBasedOptimization(population_size, limits)
    print(best_solution)
    print(best_fitness)
    print(fitness_values)