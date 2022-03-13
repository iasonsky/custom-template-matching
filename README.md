# Custom Template Matching Implementation
Object detection using color histograms over overlapping patches using custom thresholds.

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Installation
1. Clone this repository.
2. Install ```Python 3.7``` or older versions
3. Ensure OpenCV library is installed using ```pip install opencv-python```
4. Ensure `matplolib` is installed.
### Implementation
The implemented algorithm uses the sliding window approach. The algorithm takes each time a patch from the image that has the same size as the selected template. For each individual patch the histogram is compared with the histogram of the template and if the histograms show a similarity value greater than a certain threshold then a frame will be drawn on the image. In this experiment 2 different similarity metrics are used: ***correlation*** and ***chi squared***.

![Alt Text](https://user-images.githubusercontent.com/17927250/158079149-208c6619-5366-4aff-8a84-9da4b16fd068.gif)

## Results
### Correlation Results
<img src="https://user-images.githubusercontent.com/17927250/158079895-45ccb381-09bd-4b92-999b-1adf04a21901.jpg" width="700" height="500"> 

### Chi Square Results
<img src="https://user-images.githubusercontent.com/17927250/158079901-c370c526-a610-4c2f-8195-248b5da11ff0.jpg" width="700" height="500"> 
