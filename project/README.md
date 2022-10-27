# Welcome

This repository hosts the following projects:


## [01-nsf](./01-nsf) Neural source-filter waveform model 

<img src="https://nii-yamagishilab.github.io/samples-nsf/_images/fig_timeline.png" alt="drawing" width="400"/>

Projects for neural source-filter waveform models.

* All the projects include a pre-trained model and a one-click demo script. 

* Tutorial on NSF models is available in [../tutorials/b1_neural_vocoder](../tutorials/b1_neural_vocoder).

```
Xin Wang, Shinji Takaki, and Junichi Yamagishi. Neural Source-Filter Waveform Models for Statistical Parametric Speech Synthesis. IEEE/ACM Transactions on Audio, Speech, and Language Processing 28: 402–415. doi:10.1109/TASLP.2019.2956145. 2020.

Xin Wang, Shinji Takaki, and Junichi Yamagishi. Neural Source-Filter-Based Waveform Model for Statistical Parametric Speech Synthesis. In Proc. ICASSP, 5916–5920. IEEE. doi:10.1109/ICASSP.2019.8682298. 2019.

Xin Wang, and Junichi Yamagishi. Neural Harmonic-plus-Noise Waveform Model with Trainable Maximum Voice Frequency for Text-to-Speech Synthesis. In Proc. SSW, 1–6. ISCA: ISCA. doi:10.21437/SSW.2019-1. 2019.

Xin Wang, and Junichi Yamagishi. Using Cyclic Noise as the Source Signal for Neural Source-Filter-Based Speech Waveform Model. In Proc. Interspeech, 1992–1996. ISCA: ISCA. doi:10.21437/Interspeech.2020-1018. 2020.
```


## [05-nn-vocoders](./05-nn-vocoders) Neural waveform models 

Projects for other waveform models, including WaveNet vocoder, WaveGlow, Blow, and iLPCNet.

* All the projects include a pre-trained model and a one-click demo script. 

* Tutorial on NSF models is also available in [./tutorials/b1_neural_vocoder](./tutorials/b1_neural_vocoder).


### [03-asvspoof-mega](./03-asvspoof-mega) Spoofing countermeasures on ASVspoof 2019 LA 

<img src="https://d3i71xaburhd42.cloudfront.net/7d3d547871324093c76b2f994493c77bbd842285/4-Table1-1.png" alt="drawing" width="600"/>

<img src="https://www.researchgate.net/publication/357734435/figure/fig5/AS:1110903252627457@1641871383208/DET-curves-left-of-LCNN-LSTM-sum-with-LFCC-and-vanilla-softmax-sm-in-six-training-and.png" alt="drawing" width="400"/>

Project to compare 36 models.

* Pre-trained models, scores, training recipes are all available. 

* For statistical analysis, please check the tutorial notebook in [./tutorials/b2_anti_spoofing](../tutorials/b2_anti_spoofing/chapter_a1_stats_test.ipynb)


```
Xin Wang, and Junichi Yamagishi. A Comparative Study on Recent Neural Spoofing Countermeasures for Synthetic Speech Detection. In Proc. Interspeech, 4259–4263. doi:10.21437/Interspeech.2021-702. 2021.

Xin Wang, and Junichi Yamagishi. A Practical Guide to Logical Access Voice Presentation Attack Detection. In Frontiers in Fake Media Generation and Detection, 169–214. Springer. doi:10.1007/978-981-19-1524-6_8. 2022.
```


### [06-asvspoof-ood](./06-asvspoof-ood) Spoofing countermeasures confidence estimation 

<img src="https://www.researchgate.net/publication/355224018/figure/fig2/AS:1079187767066625@1634309822768/Scatter-plot-and-histogram-of-CM-and-confidence-scores-Each-sub-caption-lists.ppm" alt="drawing" width="300"/>

Project to estimate confidence for speech spoofing countermeasure

```
Xin Wang, and Junichi Yamagishi. Estimating the Confidence of Speech Spoofing Countermeasure. In Proc. ICASSP, 6372–6376. 2022.
```

### [07-asvspoof-ssl](./07-asvspoof-ssl) Spoofing countermeasures using SSL-based front end 

<img src="https://pbs.twimg.com/media/FEc07JGacAA2kfc?format=png&name=small" alt="drawing" width="300"/>

Project to use SSL-based front end

* Pre-trained models and recipes are all available.

* Dependency, pre-trained models, and SSL models will be automatically downloaded using a simple `00_demo.sh`

```
Xin Wang, and Junichi Yamagishi. Investigating Self-Supervised Front Ends for Speech Spoofing Countermeasures. In Proc. Odyssey, 100–106. doi:10.21437/Odyssey.2022-14. 2022.
```


### [08-asvspoof-activelearn](./08-asvspoof-activelearn) Spoofing countermeasures using active learning 

<img src="https://pbs.twimg.com/media/FPFN_3AaIAM5ANl?format=png&name=medium" alt="drawing" width="400"/>

Project to use active learning to train the model on a large pool data set

* Pre-trained models and recipes are all available.

* Dependency, pre-trained models, and SSL models will be automatically downloaded using a simple `00_demo.sh`

```
Xin Wang, and Junichi Yamagishi. Investigating Active-Learning-Based Training Data Selection for Speech Spoofing Countermeasure. In Proc. SLT, accepted. 2023.
```

---
That's all