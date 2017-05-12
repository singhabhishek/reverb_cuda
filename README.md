# Convolution Reverb Effect:

Reverb is a acoustic environment that surrounds a sound. Natural reverb exists everywhere in the forms of echoes we hear.
Created when a sound is reflected or absorbed by different sources in environment. Convolution reverb effect is process
of digitally simulating reverberation effect using software tool by convolving input signal with impulse response.

This project compares performance of convolution reverb effect on CPU and GPU. 

## Dependencies:
- sndfile


## Building Package:
### For CPU package:
#### Change directory to CPU folder
- $ cd CPU_code  

#### To Build package
- $ make all

#### To clean package
- $ make clean

#### Example Usage:
 - ./reverb_effect dog_11025_stereo_10sec.wav imp_11025_stereo_9sec.wav new.wav

### For GPU package:
#### Change directory to GPU folder
- $ cd GPU_code 

#### To Build package with both stream and without stream support
- $ make all

#### To Build package with stream
- $ make reverb_effect_with_stream 

#### To Build package without stream support
- $ make reverb_effect_without_stream

#### To clean package
- $ make clean // To clean package


#### Example Usage:
**With Stream support**
 - ./reverb_effect_with_stream dog_11025_stereo_10sec.wav imp_11025_stereo_9sec.wav new.wav

**Without Stream support**
 - ./reverb_effect_without_stream dog_11025_stereo_10sec.wav imp_11025_stereo_9sec.wav new.wav

### NOTE:
Package only support wav audio file with 1/2 channels. 
Mono audio files are not support with stream optimization




