# Scripts to vocode waveform


## setup:
`bash 00_setup.sh`. 

This will install environment and download pre-trained models

## Use DSP-based vocoder

[abs_dsp](./abs_dsp): Scripts to vocode waveforms using DSP-based tools

Demonstration using downloaded toy data
```bash
cd abs_dsp
bash slurm.sh
```

To do vocoding on your own data
1. edit slurm.sh and change INPUTDIR, INPUTLIST and OUTDIR
2. bash slurm.sh



## Use NN-based vocoder 

[abs_nn](./abs_nn): Scripts using neural vocoders


Demonstration using downloaded toy data
```bash
cd abs_nn/code
bash 01_feature_extraction.sh

Use file list ...
Processing ...
Processing 12 files
LA_T_9174733
LA_T_6601259
LA_T_1794966
LA_T_6936229
LA_T_9234278
LA_T_1447757
LA_T_2217523
LA_T_5322386
LA_T_7087021
LA_T_7244018
LA_T_5872241
LA_T_6642565
Acoustic features saved to ...
```

Then run 
```bash
bash 02_try_vocoding_hifigan.sh

...
Start inference (generation):                                                                          
Generate minibatch indexed within [0,12)                                                               
Generating 1, 0,LA_T_1447757,0,66880,0, time cost: 0.761s                                              
Generating 2, 1,LA_T_1794966,0,51360,0, time cost: 0.049s                                              
Generating 3, 2,LA_T_2217523,0,27200,0, time cost: 0.055s                                              
Generating 4, 3,LA_T_5322386,0,124640,0, time cost: 0.066s                                             
Generating 5, 4,LA_T_5872241,0,51200,0, time cost: 0.047s                                              
Generating 6, 5,LA_T_6601259,0,29280,0, time cost: 0.048s                                              
Generating 7, 6,LA_T_6642565,0,51840,0, time cost: 0.052s                                              
Generating 8, 7,LA_T_6936229,0,78560,0, time cost: 0.052s
Generating 9, 8,LA_T_7087021,0,56640,0, time cost: 0.058s
Generating 10, 9,LA_T_7244018,0,52640,0, time cost: 0.054s
Generating 11, 10,LA_T_9174733,0,68960,0, time cost: 0.059s
Generating 12, 11,LA_T_9234278,0,50560,0, time cost: 0.047s
Inference time cost: 1.348093s
Output data has been saved to ...
```
You can try other vocoders: 02_try_vocoding_hn-sinc-nsf.sh, 02_try_vocoding_hn-sinc-nsf-hifi.sh, 02_try_vocoding_waveglow.sh

To do vocoding using your own data
1. change `configurations` in `abs_nn/code/01_feature_extraction.sh`
2. change `configurations` in `abs_nn/code/02_try*.sh`
3. run `cd abs_nn/code; bash 01_feature_extraction.sh; bash 02_try.sh`

# Notes
1. the code was created when using legacy GPU & Torch & CUDA. To use the latest GPU card, please install torch and CUDA dependency properly.
2. the code is old, and it assumes that the file list does not contain file name extension
```bash
filename     OK
filename.wav NOT OK
```

