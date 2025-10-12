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
- Output generation in various formats

## Components

The toolkit consists of several modules:

- `design_spectrum.py`: Generates design spectra according to TBDY 2018 specifications
- `initial_motion.py`: Creates initial acceleration time series using stochastic methods
- `envelope.py`: Applies amplitude envelope functions to shape the motion over time
- `spectral_analysis.py`: Performs response spectrum calculation and motion analysis
- `spectral_matching.py`: Iteratively adjusts the motion to match the target spectrum
- `visualization.py`: Generates plots and output files
- `example.py`: Provides a complete usage example

## Installation

No special installation is required. Simply clone the repository and ensure you have the necessary dependencies installed:

```bash
git clone https://github.com/your-username/artificial-ground-motion-generator.git
cd artificial-ground-motion-generator
pip install numpy scipy matplotlib
```

## Usage

The basic workflow for generating an artificial ground motion is demonstrated in `example.py`:

```python
from design_spectrum import TBDYSpectrum
from initial_motion import InitialMotionGenerator
from envelope import EnvelopeFunction
from spectral_analysis import SpectralAnalysis
from spectral_matching import SpectralMatcher
from visualization import OutputGenerator

# Generate a motion matching TBDY 2018 spectrum
generate_motion_TDBY(
    Ss=0.3,             # Short-period spectral acceleration coefficient
    S1=0.15,            # 1.0 second period spectral acceleration coefficient
    soil_class='ZD',    # Local soil class
    duration=40.0,      # Motion duration (seconds)
    dt=0.01,            # Time step (seconds)
    pga=0.3,            # Target peak ground acceleration (g)
    omega_g=15.0,       # Kanai-Tajimi natural frequency (rad/s)
    zeta_g=0.6,         # Kanai-Tajimi damping ratio
    name='example',     # Output directory name
    imageoutput=True    # Generate visualization images
)
```
Actually there are tons of parameters in the code. Each one can be subject of an optimization problem, if its your aim.

The generated motion and analysis results will be saved in a directory named `{name}_outputs`.

## Advanced Configuration

The toolkit provides numerous parameters for fine-tuning the generated motions:

- **Design Spectrum**: Customize soil classes, spectral acceleration coefficients
- **Initial Motion**: Adjust frequency content parameters, target PGA, and filters
- **Envelope Function**: Select from various envelope types or create custom shapes
- **Spectral Matching**: Control iteration limits, convergence criteria, and frequency ranges
- **Visualization**: Generate detailed reports and plots for analysis

Refer to the docstrings and example.py for detailed parameter descriptions.

## Output

The toolkit generates various outputs such as:

- Acceleration, velocity, and displacement time series
- Response spectrum comparison plots
- Convergence metrics and error analysis
- Motion statistics and parameters
- Data files in various formats

## License

This project was created for academic purposes. Contributions and pull requests are warmly welcomed.

## How to cite

