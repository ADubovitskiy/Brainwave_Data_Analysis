# Work in Progress! 

## Introduction 
This project has been acomplished as a part of the programming course during my MSc studies. 
We have 2 datasets: "Training" and "Real". Both datasets represe The scope of this project is to analyse the data, demonstrate noise reduction techniques. 
We would then split the data into groups, and train a machine learning algorithm to detect those groups. 
Then, we can try running this ML on the "Real" dataset and see how it performs. 



### EEG Dataset
The provided Training data is shown in Figure 1. The file contains 1,440,000 recordings collected over 57.6 seconds. The recording involves spikes of 5 different classes and a large amount of background noise. It's impossible to see any trends or patterns in this until we filter the data, so that's going to be our first step. 

![Raw Data](https://github.com/ADubovitskiy/Spike-Sorting-EEG-Dataset/blob/main/Gallery/Raw_data.png) 
<p align="center">
<B><i> Figure 1 - Raw "Training" dataset </i></B>
</p>

### Filtering 
There is nothing to do in the time-domain, until we remove some noise. To remove noise, let's have a look at the signal in the frequency domain, using Fast Fourier Transforms. 

> [!NOTE]
>  In Python, the signal is transformed to the frequency domain with an  `numpy fft` function like so: 
```python
samples = mat['d']
t = 25000

# Fourier and PSD 
n = len(samples)
fhat= np.fft.fft(samples,n)
PSD = fhat * np.conj(fhat)/n
freq = (1/(t*n)) * np.arange(n)
L = np.arange(1, np.floor(n/2),dtype = 'int')
```
. Signal power is concentrated below ~1 kHz, while the low-power components spread across the rest of the spectrum are dominated by noise. Keeping only the frequency components whose power exceeds a threshold and reconstructing with an inverse FFT suppresses that noise while leaving the spike waveforms intact. The same 120 ms window after filtering is shown in Figure 4 — the spikes are preserved and the baseline is visibly cleaner.


![Power Spectral Density  Plot](https://github.com/ADubovitskiy/Spike-Sorting-EEG-Dataset/blob/main/Gallery/PSD.png)
<p align="center">
<B><i> Figure 2 - Power Spectral Density  </i></B>
</p>

![Denoised Data](https://github.com/ADubovitskiy/Spike-Sorting-EEG-Dataset/blob/main/Gallery/Data_Denoised.png)
<p align="center">
<B><i> Figure 3 - Denoised Data </i></B>
</p>

### Finding Peaks 
![Alt text](https://github.com/ADubovitskiy/Spike-Sorting-EEG-Dataset/blob/main/Gallery/Peak_Detection.png)
<p align="center">
<B><i> Figure 4 - Peak Detection </i></B>
</p>

### Windowing
![Alt text](Gallery/Windows.png?raw=true "Windows")

## Aligned Waveforms
Each window is aligned to its peak and the waveforms are overlaid, coloured by class (Figure 6). Five distinct shapes emerge — the reason the classes can be told apart.
![Alt text](https://github.com/ADubovitskiy/Spike-Sorting-EEG-Dataset/blob/main/Gallery/Data_aligned.png) "Figure X - Aligned spike waveforms"


Averaging the aligned waveforms within each class gives a mean template per spike type( Figure X). Each class has a characteristic amplitude and shape.

## Clusters or KNN? 
Now we nned to label things up. Two main options are clusters or KNN
![Clusters](https://github.com/ADubovitskiy/Spike-Sorting-EEG-Dataset/blob/main/Gallery/Clusters.png)
