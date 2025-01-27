# OnDeviceDL


#


## Final Project Proposal

### 1. Problem Definition
Microphones capture audio of varying quality. The goal of this project is to use Deep Learning models on a hardware device to clarify audio and output a smoother, refined version of the same audio accurately high clarity. 

### 2. Input Requirements
Max7800FTHR microcontroller with built-in Digital Microphone to collect test data during model deployment. 
Will use pre-recorded datasets for model training. 

### 3. Output Specifications
1. Short Time Fourier Transformation applied on Audio spectral analysis graphs. These will be passed onto some CNN model such as U-Net to separate audio noise from input audio to produce new refined spectral analysis graphs.
2. Inverse Short-time Fourier Transformation will be applied to get audio signals back from frequency domain to time domain to be sent to speaker for output.
3. Will use built-in Low-Power Stereo Audio CODEC for audio output. 

### 4. Hardware Specifications
1. Max78000FTHR with Low-Power Stereo Audio CODEC and Digital Microphone for Inference

### 5. Implementation Considerations
TinyML models such as Tensorflow Lite or Tensorflow Edge

### 6. Evaluation Criteria
Performance Metrics: Accuracy, Speed/Latency and Power Consumption
Accuracy Requirements: Remove 75% of noise without removing more than 85% of the speech 
Speed/Latency: 40 milliseconds or less for Inference
Power Consumption: ~100 milliWatts or under for Inference

