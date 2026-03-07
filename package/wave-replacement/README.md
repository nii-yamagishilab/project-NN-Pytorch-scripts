# Wave replacement (vocoded)

This is used to randomly replace segment from the corresponding segment from a vocoded version


## Dependency

```bash
# install conda environment
conda env create -f env.yaml
```

The same environment as in `../vocoding` can be used.

## Demonstration

1. `conda activate vocoding`

2. `bash run.sh`


This will do replacement using the example data in `./data`. The output will be in `./data_output`


The logs will be loke
```bash
data/real/11463_11657_000002.flac
Successfully replaced: bleak,
Successfully replaced: like
Successfully replaced: on
Successfully replaced: cheek,
Successfully replaced: your
data_output/partial_json_raw/VOC_11463_11657_000002.json
```

Note that
* `./data_output/audio`: partially replaced audio.
* `./data_output/partial_json`: json file with time stamp
    * `['replacements']` lists the words replaced
    * `['segment']` shows the text with time stamp `!!!!!` that marks a vocoded word




## Usage on your own data

1. get the time alignment using `../Whisperx` script. The output of this step is json files with time stamp of each word.

2. do vocoding using `../vocoding`. The vocoded waveforms using the same vocoder should be saved in the same folder
 
3. change configurations in `run.sh` and `bash run.sh`

```bash
    # a tag that will be added in front of the output file name
    tag=VOC
    # real audio data folder
    bon_folder=$PWD/data/real
    # json (with time alingment) folder
    json_folder=$PWD/data/json
    # vocoded data folder
    voc_folder=$PWD/data/voc
    # file list
    filelst=$PWD/data/file.lst
    # file extension
    ext=.flac

    # output folder to save partially replaced wav
    outwav=$PWD/data_output/audio
    # output folder to save raw json (without vocoded time tag)
    outjson=$PWD/data_output/partial_json_raw
    # output folder to save raw json (with vocoded time tag !!!!!!)
    outjson_tagged=$PWD/data_output/partial_json
```