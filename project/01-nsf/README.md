# Neural source-filter waveform models


<img src="https://nii-yamagishilab.github.io/samples-nsf/_images/fig_timeline.png" alt="drawing" width="400"/>

This is for the Pytorch re-implementation of NSF models. 

Two resources to mention:
1. Please visit this NSF home page https://nii-yamagishilab.github.io/samples-nsf/. It includes
    * Audio samples for each NSF model
    * Reference list
2. Detailed hands-on tutorials on NSF models available [../../tutorials/b1_neural_vocoder/README.md](../../tutorials/b1_neural_vocoder/README.md). **These tutorials are highly recommended.** There is no need to set up the environment, and the notebook can run on Google Colab!

3. Note that the tutorial chapter **chapter_a3_pretrained_vocoders.ipynb** includes pre-trained NSF models on VoxCeleb2 dev and other speech datasets [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xObWejhqcdSxFAjfWI7sudwPPMoCx-vA?usp=sharing).

## Models included

Not all the models are re-implemented. 

```sh
| - DATA: folder to store data
|
| - cyc-noise-nsf-4: cyclic-noise hn-sinc-NSF
|
| - hn-nsf: harmonic-plus-noise NSF
|
| - hn-sinc-nsf-9: harmonic-plus-noise NSF with a trainable sinc filter
|
| - hn-sinc-nsf-10: hn-sinc-nsf-9 with the BLSTM in condition module replaced by CNNs
|
| - hn-sinc-nsf-hifigan: hn-sinc-nsf 9 + hifi-gan discriminator 
```

## Quick start

step.1 choose one project
```
cd hn-nsf 
```

step.2 load dependency and PYTHONPATH
```
source ../../../env.sh 
```

step.3 run script
```
bash 00_demo.sh
```

This script will 
1. download the CMU database and pre-extracted features.
2. generates audio using a pre-trained model and the pre-extracted features.
3. trains a new model on the CMU data.

Pre-trained models are either included in `__pre-trained` or downloaded through `00_demo.sh`. Training may take a few days or more. You may run `00_demo.sh` in the background.
``` sh
bash 00_demo.sh >log_batch 2>&1 &
```

## Notes

1. To accelerate training: the default script uses `torch.backends.cudnn.deterministic= True` and `torch.backends.cudnn.benchmark = False` for reproducibility https://pytorch.org/docs/stable/notes/randomness.html. If you want to accelerate training, add options to the command line in 00_demo.sh
```   
python main.py --num-workers 10 --cudnn-deterministic-toggle --cudnn-benchmark-toggle
```
This will set `torch.backends.cudnn.deterministic=False` and `torch.backends.cudnn.benchmark = True`

2. To use a batch-size > 1:
```
python main.py --num-wokers 10 --batch-size N 
```

If you have any questions, please contact Xin.

---
That's all