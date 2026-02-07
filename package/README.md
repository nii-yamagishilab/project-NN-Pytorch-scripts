# Note

This folder contains packages of scripts and tools.

List of packages:
* `whiperx`: to get the alignment using [WhiperX](https://github.com/m-bain/whisperX)
    * input: waveform
    * output: json with word-level time stamp
* `vocoding`: using pre-trained vocoders to do waveform vocoding (copy-synthesis)
    * input: waveform
    * output: vocoded waveform
    * [this notebook](https://colab.research.google.com/drive/1xObWejhqcdSxFAjfWI7sudwPPMoCx-vA?usp=sharing) explains the vocoding process using tools in this package
* `wave-replacement`: currently, using vocoded waveform segment to replace the real audio
    * input: real waveform, vocoded waveform, json time alingment
    * output: partially replaced waveform, json with replaced words tagged
    
# Usage

Please check the `README.md` in each package. 
