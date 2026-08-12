# Work in Progress! 

## Introduction 
This project has been acomplished as a part of the programming course during my MSc studies. 
We have 2 datasets: "Training" and "Real". Both datasets represe The scope of this project is to analyse the data, demonstrate noise reduction techniques. 
We would then split the data into groups, and train a machine learning algorithm to detect those groups. 
Then, we can try running this ML on the "Real" dataset and see how it performs. 



## Spike-Sorting: EEG Dataset
## Data Processing
The provided “training” data is shown in Figure 1. The file contains 1 440 000 recordings collected over  57.6 seconds. Recordings involve spikes of 5 different classes and a large amount of background noise (Figure 2). To continue working with this data and to reduce any errors caused by noise, filtering is applied. 

### Filtering 
### Finding Peaks 
![Alt text](Gallery/Peaks.png?raw=true "Peaks")
### Windowing
![Alt text](Gallery/Windows.png?raw=true "Windows")

## Multi-Layer Perceptron (MLP)
![Alt text](Gallery/output.png?raw=true "output")

