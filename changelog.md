# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2026-05-24

### Added

- **Multi-Format Ground Motion Exporter (`visualization.py`)**:
  - Implemented `.csv` export functionality (`save_csv_data`) for both acceleration time history and response spectrum outputs, ensuring compatibility with standard spreadsheet editors (Microsoft Excel), MATLAB, R, etc.
  - Implemented `.json` export package (`save_json_data`) containing structured metadata, tbdy parameters, calculated time histories (Time, Acceleration), response spectra, and all computed ground motion engineering metrics (Arias, CAV, PGA, PGV, PGD, zero crossings, etc.).
  - Added dedicated SAP2000 / ETABS import format (`save_sap2000_acceleration_data`) with support for robust 1-column (acceleration only) and 2-column (Time and Acceleration space-separated) files. SAP2000 comments starting with `$` are automatically included to document import properties (e.g. constant $dt = 0.01$ s).
- **Customizable Target/Matching Damping Ratio**:
  - Fully parameterized the damping ratio (`damping` parameter in `generate_motion_TDBY`, defaults to `0.05` / `5%`) in all matching iterations, spectral analysis, and visual report generation, addressing reviewer comments.

### Changed

- **Generator Parameterization & Outputs**:
  - Updated the signature of `generate_motion_TDBY` in `agmg.py`, `agmg_TLBO.py`, and `complete_test/agmg_test.py` to support `damping` and formatting flags (`export_csv`, `export_json`, `export_sap2000`, `sap2000_two_column`).
  - Integrated summary printing of all generated formats (TXT, CSV, JSON, SAP2000) in final console outputs.

## [0.1.1] - 2026-05-23

### Added

- **Dependency Management (`requirements.txt`)**: Prepared a comprehensive `requirements.txt` file listing all required third-party Python libraries (`numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`) with compatible version constraints corresponding to the active development environment.
- **Engineering Ground Motion Parameters & Reporting**:
  - Implemented mathematical calculations for **Significant Durations** ($D_{5-95}$ and $D_{5-75}$), **Arias Intensity** ($I_a$), **Cumulative Absolute Velocity** (CAV), and peak metrics (PGA in metric units, PGV, PGD, PGV/PGA ratio).
  - Integrated zero-crossing rate computation to characterize signal frequency nonstationarity.
  - Automatically calculate and present these parameters at the final finalized stage of ground motion generation, writing them to standard output and generated text reports without interfering with the TLBO optimization process.

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
