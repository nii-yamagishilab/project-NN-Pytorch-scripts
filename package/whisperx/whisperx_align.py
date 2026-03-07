#!/usr/bin/env python
#
# Usage: python whisperx_align input_file_list input_dir output_dir
#
# By default, assuming the input is .wav
# By default, assuming the path of input and output be:
#  if not, please customize the code
# 
# for fileid in input_file_list
#   input = input_dir/fileid
#   output = output_dir/fileid
#   ...

import os
import sys
import json
from pathlib import Path
import whisperx
import gc

device = "cuda"
batch_size = 16 # reduce if low on GPU mem
compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)
language = 'en'

# 1. Transcribe with original whisper (batched)
# save model to local path (optional)
# model_dir = "/path/"
# model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)

model = whisperx.load_model("large-v2", device, compute_type=compute_type)
# 2. Align whisper output
model_a, metadata = whisperx.load_align_model(language_code=language, device=device)



def get_align_json(audio_file, output_file):
    
    #audio_file = "/gs/bs/tgh-25IAC/ud03523/DATA/sample.wav"
    #output_file = './sample.json'

    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)
    #print(result["segments"]) # before alignment

    # delete model if low on GPU resources
    # import gc; import torch; gc.collect(); torch.cuda.empty_cache(); del model
    
    language = result['language']
    #model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)

    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    result['language'] = language

    with open(output_file, 'w') as txtfile:
        json.dump(result, txtfile, indent=4)


if __name__ == "__main__":
    
    file_list = sys.argv[1]
    input_dir = sys.argv[2]
    out_dir = sys.argv[3]

    with open(file_list, 'r') as file_ptr:
        for line in file_ptr:
            file_path = line.rstrip()
            print(file_path)
            input_path = (Path(input_dir) / file_path).with_suffix('.wav')
            output_path = (Path(out_dir) / file_path).with_suffix('.json')

            if output_path.is_file():
                pass
            else:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                get_align_json(input_path, output_path)
