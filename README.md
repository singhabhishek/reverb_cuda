Convolution Reverb Effect:

Reverb is a acoustic environment that surrounds a sound. Natural reverb exists everywhere in the forms of echoes we hear.
Created when a sound is reflected or absorbed by different sources in environment. Convolution reverb effect is process
of digitally simulating reverberation effect using software tool by convolving input signal with impulse response.

This project compares performance of convolution reverb effect on CPU and GPU. 

Installation:

Dependencies:
- sndfile

// Switch to CPU Package:
- $ cd CPU_code 

// Switch to GPU Package:
- $ cd GPU_code 

// To build package
 - $ make all

// To clean package
- $ make clean

How to Use:
- ./reverb_effect song impulse output
 
Example Usage:
 - ./reverb_effect dog_11025_stereo_10sec.wav imp_11025_stereo_9sec.wav new.wav



