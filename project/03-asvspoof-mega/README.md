# A Comparative Study on Recent Neural Spoofing Countermeasures for Synthetic Speech Detection


<img src="../../misc/fig_eer_table.png" alt="drawing" width="600"/>

This project compares a few popular CMs. The main findings are:
1. CM performance may be heavily influenced by the random initial seed. Multiple running is recommended.
2. The difference between two EERs may be statistically insignificant. Be careful when reporting the result. For statistical analysis, please check the tutorial notebook in [../../tutorials/b2_anti_spoofing](../../tutorials/b2_anti_spoofing/chapter_a1_stats_test.ipynb)


Arxiv link: [https://arxiv.org/abs/2103.11326](https://arxiv.org/abs/2103.11326)

```
Xin Wang, and Junichi Yamagishi. A Comparative Study on Recent Neural Spoofing Countermeasures for Synthetic Speech Detection. In Proc. Interspeech, 4259–4263. doi:10.21437/Interspeech.2021-702. 2021.

Xin Wang, and Junichi Yamagishi. A Practical Guide to Logical Access Voice Presentation Attack Detection. In Frontiers in Fake Media Generation and Detection, 169–214. Springer. doi:10.1007/978-981-19-1524-6_8. 2022.

@inproceedings{wang2021comparative,
author = {Wang, Xin and Yamagishi, Junichi},
booktitle = {Proc. Interspeech},
doi = {10.21437/Interspeech.2021-702},
pages = {4259--4263},
title = {{A comparative study on recent neural spoofing countermeasures for synthetic speech detection}},
year = {2021}
}

@incollection{Wang2022,
author = {Wang, Xin and Yamagishi, Junichi},
booktitle = {Frontiers in Fake Media Generation and Detection},
doi = {10.1007/978-981-19-1524-6_8},
pages = {169--214},
publisher = {Springer},
title = {{A Practical Guide to Logical Access Voice Presentation Attack Detection}},
url = {https://link.springer.com/10.1007/978-981-19-1524-6_8},
year = {2022}
}
```


## How-to

### step.1 setup conda and Pytorch environment

See [../../README.md](../../README.md)


### step.2 prepare data

1. Download [ASVspoof 2019 LA](https://doi.org/10.7488/ds/2555) and convert FLAC to WAV.
2. Put eval set waves to `./DATA/asvspoof2019_LA/eval`
3. Put train and dev sets to `./DATA/asvspoof2019_LA/train_dev`. You may also link eval and train_dev to the folder of waveforms.
   
4. Make sure that the two folders contain the waveform files: 
```
   $: ls DATA/asvspoof2019_LA/eval 
   LA_E_1000147.wav
   LA_E_1000273.wav
   LA_E_1000791.wav
   LA_E_1000841.wav
   LA_E_1000989.wav
   ...
   $: ls DATA/asvspoof2019_LA/eval | wc -l
   71237

   $: ls DATA/asvspoof2019_LA/train_dev
   LA_D_1000265.wav
   LA_D_1000752.wav
   LA_D_1001095.wav
   LA_D_1002130.wav
   LA_D_1002200.wav
   LA_D_1002318.wav
   ...

   $: ls DATA/asvspoof2019_LA/train_dev | wc -l
   50224
```

### step.3 run script
For example
```sh
bash 00_demo.sh lfcc-lcnn-lstmsum-p2s/01 > log_batch 2>$1 &
```

This script will 
1. score the test data using the pre-trained model by Xin
2. train a new model and score the test data again

It will take more than 10 hours to finish.

### step.4 try score fusing
```sh
bash 00_demo_fuse_score.sh
```

This script will fuse scores produced from pre-trained models. 

You can also try other scripts after modifying this simple script.


## Folder structure


The name of each folder is `FEAT-lcnn-NET-LOSS`, where
* FEAT: lfb, lfcc, or spec2
* NET: fixed (fixed size input), lstm-sum, or attention
* LOSS: oc (one-class), am (additive margin), p2s (p2sgrad), sig (sigmoid)

```sh
|- 00_demo.sh: 
|   Demonstration script for model evaluation and training 
|   Please check the document of this script.
|
|- 01_config.py: 
|   basic configuration for models (used by all models)
|- 01_main.py:
|   basic main.py to train and evaluate (used by all models)
|- 02_evaluate.py:
|   script wrapper to compute EER and min tDCF
|
|- DATA/asvspoof2019_LA/
|  |- protocol.txt: 
|  |   concatenation of protocol files ASVspoof2019_LA_cm_protocols/*.txt. 
|  |   this will be loaded by pytorch code for training and evaluation
|  |- scp: 
|  |   list of files for train, dev, and eval sets
|  |- train_dev: 
|  |   link to the folder that stores both train and dev waveform. 
|  |   both train and dev data are saved in the same folder. 
|  |- eval: 
|  |   link to the folder that stores test set waveforms
|
|- conv
|   Cached data on the trial length for specific databases
|   They are automatically produced by Pytorch code, given config.py
|   
|- 99_get.sh
|   script used by Xin to copy files
```

In each model folder, for example
```
|- lfcc-lcnn-lstmsum-p2s
|  |- 01: 
|  |  folder for the models in the 1st training-evaluation round
|  |  |- 00_train.sh: recipe to train the model
|  |  |- 01_eval.sh: command line to evaluate the model on eval set
|  |  |- model.py: definition of model in Pytorch code
|  |  |- __pretrained
|  |       |- trained_network.pt: pre-trained model 
|  |       |- log_output_testset: log that saves the score of eval trials
|  |
|  |- 02:
|     folder for the models in the 2nd training-evaluation round
```

The six folders share the same model definition (model.py). They are
just trained using different initial random seeds.


## Note

1. Since this commit, it is also OK to directly load *.flac.
   1. make sure you have installed `soundfile` (see ../../env.yml)
   2. put flac in DATA/asvspoof2019_LA/eval and DATA/asvspoof2019_LA/train_dev
   3. change in 01_config.py: input_exts = ['.wav'] to input_exts = ['.flac']
   
2. Running `00_demo.sh` requires GPU card with sufficient memory. If GPU memory is insufficient, please reduce `--batch-size` in `*/*/00_train.sh`

3. The `00_demo.sh` works in this way (for training or evaluation)
   1. go to a model folder
   2. copy 01_main.py and 01_config.py as main.py and config.py
   3. run `python main.py`. By default, it will load `model.py` and `config.py` in the model folder 
   4. `main.py` asks `core_scripts/data_io/default_data_io.py` to scan the files and 
      prepare `torch.utils.Dataset`, given the information in `config.py`
   5. `main.py` load the model definition in `model.py` and call `core_scripts/nn_manager` to run the training or evaluation loop

4. Input data are waveforms. Features such as LFCC will be produced by the code internally. 

5. Target labels should be provided in protocol.txt (see DATA/asvspoof2019_la/protocol.txt). The `model.py` will parse `protocol.txt` and use the "bonafide" and "spoof" as target labels. The model.py will internally convert "bonafide" to 1 and "spoof" to 0. Accordingly, the output score from the model looks like:
```
Output, File name, label, score
Output, LA_E_8688127, 0, -0.011127
Output, LA_E_2504134, 1, 0.999660
```
Where 0 and 1 denote spoof and bona fide, respectively.

6. `model.py` in this project was created from a common template. It has many functions unnecessary for ASVspoof model but required for the Pytorch scripts: `model.prepare_mean_std, model.normalize_input, model.normalize_target,  model.denormalize_output.`

    1. For ASVspoof models, we set input_norm = [False] in config.py, In this case, mean = 0 and std = 1 are used for input data (i.e., waveform)
   
    2. After waveform is read, the front-end computes the acoustic features in each mini-batch. No normalization is conducted.
   
    3. ASVspoof models do not load target labels from disk. It gets the target label from the protocol (i.e., `target_vec` in `model.py`). Thus, `model.normalize_target` is not used on the target label. 
     
    4. model.forward() returns `[scores, target_vec, True]` to the script, and the script will give [scores, target_vec, True] to `Loss().compute()` as the 1st non-self argument. Thus, in `Loss().compute(outputs, target)`, `outputs[0]` denotes scores, and `outputs[1]` denotes target_vec.
    
    5. Note that the `target` in `Loss()` is an empty list []. This argument `target` is for target data loaded from disk. But for ASVspoof models the target labels are not loaded from disk, thus `target` is []. This behavior is controled by `output_dirs = []` in `config.py`. It tells the script NOT to load anything as "target".
   
   6. For a high-level explanation on how the functions and objects work in the script, please check ../../README.md

If you want to use the code for other databases, please check `../04-asvspoof2021-toy/`
and try the toy example. It will include a toy dataset for the training process.

---
That's all

