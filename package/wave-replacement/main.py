#!/usr/bin/bash

import os
import sys
import replace
from pathlib import Path
from multiprocessing import Pool

import random
random.seed(int(1234))

if __name__ == "__main__":
    bon_folder = Path(sys.argv[1])
    voc_folder = Path(sys.argv[2])
    json_folder = Path(sys.argv[3])
    outwav_folder = Path(sys.argv[4])
    outjson_folder = Path(sys.argv[5])
    filelist = Path(sys.argv[6])
    prefix = sys.argv[7]
    ext = sys.argv[8]
    
    with open(filelist, 'r') as file_ptr:
        filelist = [x.rstrip('\n') for x in file_ptr]
        
    def wrapper(filename):
        bon_file = (bon_folder / filename).with_suffix(ext)
        voc_file = (voc_folder / filename).with_suffix(ext)
        json_file = (json_folder / filename).with_suffix('.json')
        out_wav = (outwav_folder/ (prefix+filename)).with_suffix(ext)
        out_json = (outjson_folder / (prefix+filename)).with_suffix('.json')
        
        if out_wav.is_file() and out_json.is_file():
            print("Files already exist. Skip {:s}".format(str(bon_file)))
            return
        else:
            # number of words to replace
            num_word_replace = random.randint(2, 5)
            replace.replace_random_words(bon_file, voc_file, json_file, 
                                         out_wav, out_json, num_word_replace)
            return
        
    with Pool(4) as p:
        p.map(wrapper, filelist)
        
    
        