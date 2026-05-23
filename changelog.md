# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-05-23

### Added

- **Dependency Management (`requirements.txt`)**: Prepared a comprehensive `requirements.txt` file listing all required third-party Python libraries (`numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`) with compatible version constraints corresponding to the active development environment.

### Changed

- **Reproducibility with Controllable Seeds (`agmg.py`)**:
  - Enhanced the `generate_motion_TDBY` function signature to accept an optional `seed` parameter (defaults to `None`).
  - Implemented automatic global seeding (`random.seed` and `np.random.seed`) inside `generate_motion_TDBY` when `seed` is provided.
  - Propagated the `seed` argument to `InitialMotionGenerator.generate_clough_penzien_motion` to ensure the stochastic process of initial acceleration generation is fully deterministic.
  - Updated the example runner at the bottom of `agmg.py` to demonstrate deterministic and reproducible execution using `seed=1`, `max_iterations=30`, and `max_non_improvement=15`.

## [0.1.0] - 2026-05-23

### Added

- Initial release of the Artificial Ground Motion Generator (AGMG) tool.
- Support for target design spectra generation according to the Turkish Earthquake Code (TBDY 2018).
- Stochastic generation of initial acceleration time series via Clough-Penzien/Kanai-Tajimi filtered white noise.
- Customizable amplitude envelopes (e.g., Saragoni-Hart, custom trapezoidal) to shape ground motion over time.
- Interactive response spectrum matching through frequency-domain Fourier adjustment.
- Baseline correction (highpass filtering) and numerical integration for velocity and displacement.
- Plotting and visualization tools (`visualization.py`) with customizable plot options.
