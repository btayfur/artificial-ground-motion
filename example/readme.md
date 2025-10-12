# Example

This example demonstrates the process of spectral matching using the `generate_motion_TDBY` function. 

```python
generate_motion_TDBY(
    Ss=0.3,
    S1=0.15,
    soil_class='ZD',
    max_iterations=20,
    max_non_improvement=3,
    duration=40.0,
    dt=0.01,
    pga=0.3,
    omega_g=15.0,
    zeta_g=0.6,
    name='example',
    imageoutput=True)
```

## Design Spectrum (Target Spectrum)
- TBDY
- ASCE (soon)

![Target Spectrum](imgs/00.png)

## Initial Motion
- Simple White Noise

### PGA estimation algorithms (optional)
- Sabetta-Pugliese Model
- Graizer-Kalkan 2015
- Campbell & Bozorgnia 2014

### Filters (optional but highly recommended)
- Kanai Tajimi Filter
- Clough Penzien Filter

![White Noise filtered w/Kanai Tajimi](imgs/01.png)

External lib used for low pass filter:

![Filtered signal](imgs/02.png)

![Low-Pass Filter](imgs/02_2.png)

## Enveloping
- Custom envelope 
- Saragoni and Hart envelope function

![Envelope function w/custom envelope](imgs/03.png)

![Time series](imgs/04.png)

## Spectral Matching Process
There are too many parameters in that class. In most cases, it doesn't need to be tune.

![Stop criteria is met in that run. Its not happen always :)](imgs/05.png)

## Results

### Comparison of spectrums
![Comparison of spectrums](imgs/06.png)

![Comparison of spectrums in log scale](imgs/07.png)

### Spectogram of the motion
![Spectogram](imgs/08.png)

### Final ground motion
![Final ground motion](imgs/09.png)
