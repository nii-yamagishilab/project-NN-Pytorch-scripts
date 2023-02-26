# Tutorials on DNN-based Vocoders

These are tutorials on some deep-neural-network vocoders in Pytorch and Python. 

Features of these tutorials:
1. Pre-trained model is provided to produce audio samples.
2. No painful installation of dependency. Just directly run the notebook on Google Colab.
3. Very detailed implementations, for example, how to cache intermediate output in causal dilated convolution.
4. Not only DNN but also DSP techniques are explained, e.g., linear prediction, overlap-add ...

All are hosted on the Google Colab platform.


| Link | Chapter |  |
| --- | :-- | :-- |
| | **Introduction and basics**  | |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1EO-ggi1U9f2zXwTiqg7AEljVx11JKta7?usp=sharing)| **chapter_1_introduction.ipynb** | entry point and Python/Pytorch conventions |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1mZo73dbKeWr4hDHftDQI9rlDK1HyMf5C?usp=sharing) | **chapter_2_DSP_tools_Python.ipynb** | selected DSP tools for speech processing
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1BEVR6jPFelczCPM5NZvuk8cRy8YpGacR?usp=sharing) | **chapter_3_DSP_tools_in_DNN_Pytorch.ipynb** |  selected DSP tools implemented as layers in neural networks;
| | **DSP-based Vocoder** | 
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19Ky2T3hIbHpGK57IQ0tAZc3pX6HLhs97?usp=sharing) | **chapter_4_DSP-based_Vocoder** | traditional DSP-based vocoder included in [SPTK toolkit](https://github.com/sp-nitech/SPTK);
| | **Neural vocoders** |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Cn75nsytkYDFhRoQnOC7URnDVl510FfO?usp=sharing) |   **chapter_5_DSP+DNN_NSF.ipynb** | neural source-filter model
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KWPf3dm9XhHi5v3gZXxZNG-VSqfxXT43?usp=sharing) | **chapter_6_AR_WaveNet.ipynb** | Autogressive WaveNet vocoder
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1e5y39ol37pRoecxUlhdBbhAcNOeoMykc?usp=sharing) | **chapter_7_AR_iLPCNet.ipynb** | Autogressive iLPCNet
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1W1Itp-1SLL3fZxm0vfAfqZnbTs40u0V4?usp=sharing) | **chapter_8_Flow_WaveGlow.ipynb** | Flow-based WaveGlow model
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ekU2YlG-05FaMvWGPEpYIRFvqxSaMsqE?usp=sharing) | **chapter_9_GAN_HiFiGAN_NSFw/GAN.ipynb** | HiFiGAN, and NSF + HiFiGAN 
| | **Appendix** |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1g-_rveOLSdqFtnl2IrLyHOUyW01-gN6y?usp=sharing) | **chapter_a1_Linear_prediction.ipynb** | Details on a naive implementation of Linear Prediction;
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1G8lUTlEQmKinh80OdP5NY7tflOyvVQAo?usp=sharing) | **chapter_a2_Music_NSF.ipynb** | Application of NSF to music instrumental audios.
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xObWejhqcdSxFAjfWI7sudwPPMoCx-vA?usp=sharing) | **chapter_a3_pretrained_vocoders.ipynb** | Pretrained neural vocoders on a few speech datasets.


Click `Open in Colab` will open the book. You can also download them from [Google Drive](https://drive.google.com/drive/folders/1lbIzlIWEDasNZFz9oerWAJmQC6YQIJyl?usp=sharing).

Models and implementations are for the tutorial, therefore lacking intensive tuning and optimization. Neither am I good at that. If you have ideas on how to improve, your feedback is appreciated!

The above notebooks were used in ICASSP 2022 short course and [ISCA Speech Processing Course in Crete](https://www.csd.uoc.gr/~spcc/).

```sh
@misc{Stylianou2022,
author = {Stylianou, Yannis and Tsiaras, Vassilis and Conkie, Alistair and Maiti, Soumi and Yamagishi, Junichi and Wang, Xin and Chen, Yutian and Slaney, Malcom and Petkov, Petko and Padinjaru, Shifas and Kafentzis, George},
mendeley-groups = {misc,self-arxiv},
title = {{ICASSP2022 Shortcouse: Inclusive Neural Speech Synthesis -iNSS}},
year = {2022}
}
```

---
By [Xin Wang](https://github.com/TonyWangX/TonyWangX.github.io)