# Artificial Ground Motion Generator

A Python-based tool for generating artificial earthquake ground motions that match target design spectra according to the Turkish Earthquake Code (TBDY 2018). I'll add some other major regulations as soon as possible. The structure of the tool is very practical for adding any other function in any step.

![Spectrum comparison](example/imgs/07.png)
![Spectogram of the final motion](example/imgs/08.png)

## Overview

This repository provides a comprehensive toolkit for generating earthquake acceleration time series using stochastic methods to match specific design response spectra. The generated motions can be used for dynamic structural analysis when recorded ground motions are not available or when specific spectral characteristics are required.

## Features

- Generate design response spectra according to TBDY 2018 (Turkish Earthquake Code)
- Create initial acceleration time series using Kanai-Tajimi method
- Apply customizable envelope functions to shape the motion's amplitude over time
- Perform spectral matching to align the motion's response spectrum with the target design spectrum
- Apply baseline correction and filtering
- Comprehensive visualization of results
- **Flexible Multi-Format Exports**: Automatically exports ground motions and spectra to Plain Text (`.txt`), Comma-Separated Values (`.csv` for Excel/MATLAB/R), a program-friendly JSON package (`.json` with all time histories, metadata, and computed ground motion parameters), and dedicated SAP2000/ETABS import formats (robust 1-column and 2-column configurations).
- **Customizable Damping Ratio**: Configure and target sönüm (damping) values other than the standard 5% (e.g., 2%, 10%) directly throughout matching iterations and analyses.
- **Controllable Reproducibility**: Support for random seed generation to ensure identical, deterministic results across runs.

## Components

The toolkit consists of several modules:

- `design_spectrum.py`: Generates design spectra according to TBDY 2018 specifications
- `initial_motion.py`: Creates initial acceleration time series using stochastic methods
- `envelope.py`: Applies amplitude envelope functions to shape the motion over time
- `spectral_analysis.py`: Performs response spectrum calculation and motion analysis
- `spectral_matching.py`: Iteratively adjusts the motion to match the target spectrum
- `visualization.py`: Generates plots and output files
- `agmg.py`: Provides a complete usage example and main generator entry point

## Installation

No special installation is required. Simply clone the repository and ensure you have the necessary dependencies installed using the provided requirements file:

```bash
git clone https://github.com/btayfur/artificial-ground-motion.git
cd artificial-ground-motion
pip install -r requirements.txt
```

## Usage

The basic workflow for generating an artificial ground motion is demonstrated in `agmg.py`:

```python
from agmg import generate_motion_TDBY

# Generate a motion matching TBDY 2018 spectrum (with optional seed and custom formats/damping)
best_avg_error, best_max_error = generate_motion_TDBY(
    Ss=1.3,             # Short-period spectral acceleration coefficient
    S1=0.4,             # 1.0 second period spectral acceleration coefficient
    soil_class='ZB',    # Local soil class
    max_iterations=30,  # Maximum matching iterations
    max_non_improvement=15, # Wait limit without error improvement
    duration=40.0,      # Motion duration (seconds)
    dt=0.01,            # Time step (seconds)
    pga_="tdby",        # Target PGA method or specific value
    omega_g=15.0,       # Kanai-Tajimi ground natural frequency (rad/s)
    zeta_g=0.6,         # Kanai-Tajimi ground damping ratio
    name='example',     # Output directory name
    imageoutput=True,   # Generate visualization images
    seed=0,             # Pass an integer seed to make calculations fully reproducible (default is None)
    damping=0.05,       # Target sönüm (damping) ratio (default is 0.05 = 5%)
    export_csv=True,    # Save output as CSV (Time-History & Response Spectrum)
    export_json=True,   # Save output as JSON package with metadata & engineering metrics
    export_sap2000=True, # Save output formatted for direct SAP2000 import
    sap2000_two_column=False # Use robust 1-column acceleration format for SAP2000
)
```

The generated motion and analysis results will be saved in a directory named `{name}_outputs`.

## Advanced Configuration

The toolkit provides numerous parameters for fine-tuning the generated motions:

- **Design Spectrum**: Customize soil classes, spectral acceleration coefficients
- **Initial Motion**: Adjust frequency content parameters, target PGA, and filters
- **Envelope Function**: Select from various envelope types or create custom shapes
- **Spectral Matching**: Control iteration limits, convergence criteria, and frequency ranges
- **Visualization**: Generate detailed reports and plots for analysis

Refer to the docstrings and `agmg.py` for detailed parameter descriptions.

## Output

The toolkit generates various outputs such as:

- Acceleration, velocity, and displacement time series
- Response spectrum comparison plots
- Convergence metrics and error analysis
- Motion statistics and parameters
- Data files in various formats

## License

This project is created for academic purposes. Contributions and pull requests are warmly welcomed.

## How to cite
