#!/usr/bin/env python
#
# python replace.py input_file.wav vocoded_file.wav time_json.json output_wav output_json
#


import os
import sys
import json
import random
from pathlib import Path
import soundfile as sf
import overlap_cat

random.seed(int(1234))

def get_all_words(json_path):
    """Extract all words with their timings"""
    with open(json_path) as f:
        data = json.load(f)
    
    # Handle both 'word_segments' and 'segments.words' formats
    word_segments = []
    if 'word_segments' in data:
        word_segments = data['word_segments']
    elif 'segments' in data:
        for segment in data['segments']:
            if 'words' in segment:
                word_segments.extend(segment['words'])
    else:
        print("Wrong format {:s}".format(json_path))
        sys.exit(1)
    
    # Return words with their timings (None if missing)
    return [(w['word'], w.get('start'), w.get('end')) for w in word_segments]

def replace_random_words(bon_file, voc_file, json_file, out_file, out_json, 
                         num_word_replace=1):
    print(bon_file)
    # load wavefile
    bon_wav, bon_sr = sf.read(bon_file)
    voc_wav, voc_sr = sf.read(voc_file)
    assert bon_sr == voc_sr, "unequal sampling rate"
    
    # load words
    words = get_all_words(json_file)
    
    # randomly select words to be replaced
    flag = True
    cnt = 0
    while flag:
        if cnt >= 10:
            print("Fail to process {:s}".format(str(bon_file)))
            
        selected = random.sample(words, min(num_word_replace, len(words)))
        wav_partial = bon_wav.copy()
        replacement_info = []

        for idx, (word, orig_start, orig_end) in enumerate(selected):

            tmp, _, _, _ = overlap_cat.replace_word_segment(
                wav_partial, orig_start, orig_end, 
                voc_wav, orig_start, orig_end, bon_sr)

            if tmp is not None:
                wav_partial = tmp
                replacement_info.append({
                    'word': word,
                    'original_start': orig_start,
                    'original_end': orig_end,
                    'vocoded_start': orig_start,
                    'vocoded_end': orig_end,
                    'replacement_start': orig_start,
                    'replacement_end': orig_end,
                    'lags': []
                })
                print(f"Successfully replaced: {word}")
            else:
                pass

        if not replacement_info:
            cnt += 1
        else:
            flag = False

    # Save output files                              
    sf.write(out_file, wav_partial, bon_sr)

    # Load original JSON and add replacement info                       
    with open(json_file) as f:
        full_json = json.load(f)
    full_json["replacements"] = []
    for rep in replacement_info:
        full_json["replacements"].append({
            "word": rep["word"],
            "replacement_start": rep["replacement_start"],
            "replacement_end": rep["replacement_end"]
        })
    with open(out_json, 'w') as f:
        json.dump(full_json, f, indent=2)
        
    return 

if __name__ == "__main__":

    bon_file = Path(sys.argv[1])
    voc_file = Path(sys.argv[2])
    json_file = Path(sys.argv[3])
    out_file = Path(sys.argv[4])
    out_json = Path(sys.argv[5])
    
    # number of words to replace
    num_word_replace = random.randint(0, 3)
    replace_random_words(bon_file, voc_file, json_file, 
                         out_file, out_json, num_word_replace)