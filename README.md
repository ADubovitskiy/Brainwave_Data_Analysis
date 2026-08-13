# Work in Progress! 

## Introduction 
This project demonstrates a method of analysing noisy data and sorting it in Python. 

We have 2 datasets: "Training" and "Real". Both datasets represent 1 minute of real-life brainwave data. The scope of this project is to analyse the data, demonstrate noise reduction techniques. We would then split the data into groups and train a machine learning algorithm to detect those groups. 

Then, we can try running this ML on the "Real" dataset and see how it performs.


## Training Dataset
The provided Training data is shown in Figure 1. The file contains 1,440,000 recordings collected over 57.6 seconds. The recording involves spikes of 5 different classes and a large amount of background noise. It's impossible to see any trends or patterns in this until we filter the data, so that's going to be our first step.

![Raw Data](https://github.com/ADubovitskiy/Spike-Sorting-EEG-Dataset/blob/main/Gallery/Raw_data.png) 
<p align="center">
<B><i> Figure 1 - Raw "Training" dataset </i></B>
</p>

## Filtering 
There is not much to do in the time-domain, until we remove some noise. To remove noise, let's have a look at the signal in the frequency domain, using Fast Fourier Transforms.

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
We can see that the signal is mainly concentrated below ~1 kHz. However, there is a lot of noise stretching all the way up to 3kHz. We can pick an arbitrary cut-off amplitude (0.5mV in this example), so if the signal is ≤ 0.5 we assume it is noise and just remove it.

```python
# Removing all the noise through the PSD plot
threshold = 0.5
mask = PSD > threshold                  # Real signal is above this cutoff
clean = np.fft.ifft(fhat * mask).real     
```
![Power Spectral Density  Plot](https://github.com/ADubovitskiy/Spike-Sorting-EEG-Dataset/blob/main/Gallery/PSD.png)
<p align="center">
<B><i> Figure 2 - Power Spectral Density  </i></B>
</p>

> [!TIP]
> Figure 3 below demonstrates a 120ms snippet of the data before and after we removed the noise. 
> It is now much easier to work with.

![Denoised Data](https://github.com/ADubovitskiy/Spike-Sorting-EEG-Dataset/blob/main/Gallery/Data_Denoised.png)
<p align="center">
<B><i> Figure 3 - Denoised Data </i></B>
</p>

### Finding Peaks 
With the clean data, we can now see some distinct signal spikes. However, we want to know if they belong to the same category - according to the "Training" dataset, there are 5 different types of spikes.

```python
# Find peaks
threshold = 4 * np.median(np.abs(spikes)) / 0.6745
peaks, _  = find_peaks(spikes, height=threshold, distance=int(0.001 * fs))
print(f'{len(peaks)} spikes detected')  
```

![Alt text](https://github.com/ADubovitskiy/Spike-Sorting-EEG-Dataset/blob/main/Gallery/Peak_Detection.png)
<p align="center">
<B><i> Figure 4 - Peak Detection </i></B>
</p>


## Aligned Waveforms
One way to compare the spikes is to overlay them one on top of each other. To do that, we select a certain number of points either side of each peak, essentially placing a peak in a "window". Each window is aligned to its peak and the waveforms are overlaid, coloured by class (Figure 5). Now we can see 5 distinct shapes - just like we were told by the "Training" data.

```python
PRE, POST = 20, 40     # number of samples before and after the peak
windows, labels = [], []
for idx, c in zip(Index, Class):
    pk = idx - 5 + np.argmax(spikes[idx-5 : idx+25])   # align to the local peak
    windows.append(spikes[pk-PRE : pk+POST])
    labels.append(c)
windows = np.array(windows)
labels  = np.array(labels)
```

![Aligned spikes](https://github.com/ADubovitskiy/Spike-Sorting-EEG-Dataset/blob/main/Gallery/Data_aligned.png)
<p align="center">
<B><i> Figure 5 - Aligned spikes </i></B>
</p>

> [!TIP]
> Averaging the aligned waveforms within each class gives a mean template per spike type.
> Each class has a characteristic amplitude and shape.

![Data spikes placed into groups](https://github.com/ADubovitskiy/Brainwave_Data_Analysis/blob/main/Gallery/Aligned_Groups.png)
<p align="center">
<B><i> Figure 5 - Aligned & Grouped spikes </i></B>
</p>

## Clusters or KNN? 
The final step is to learn how to tell those spikes apart. 
![Clusters](https://github.com/ADubovitskiy/Spike-Sorting-EEG-Dataset/blob/main/Gallery/Clusters.png)

## Real Dataset

