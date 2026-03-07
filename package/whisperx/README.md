# WhisperX alignment

This script uses whisperx to do alignment.

The basic usage is from https://github.com/m-bain/whisperX?tab=readme-ov-file#python-usage-

* step 1: install whisperx following the above github
* step 2: change configurations in run.sh

```
# list of input file
input_file_list=
# directory of the input audio
datadir=
# directory of the output folder to save json
savedir=

By default, assuming the input is .wav
By default, assuming the path of input and output be:
# for fileid in input_file_list
#   input = input_dir/fileid
#   output = output_dir/fileid
#   ...
If not, please customize the code in whisperx_align.py
```

* step 3: bash run.sh. This will submit the jobs to TSUBAME

The output will be in ${savedir}

Note that the script `submit.sh` and `run.sh` are customized for TSUBAME. 

You can directly run `whisperx_align.py`