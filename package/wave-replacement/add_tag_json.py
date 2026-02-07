#!/usr/bin/env python
#
# python replace.py input_file.wav vocoded_file.wav time_json.json output_wav output_json
#


import os
import sys
import json

vocoding_label = "!!!!!!"


def check(word_list, word):
    for word_a in word_list:
        if word_a['word'] == word['word'] and word_a['replacement_start'] == word['start'] and word_a['replacement_end'] == word['end']:
            return True
    return False

def get_all_words(json_path, out_json):
    """Extract all words with their timings"""
    with open(json_path) as f:
        data = json.load(f)
    
    # collect vocoded words 
    replaced_words = data['replacements']
    cnt = 0
    
    if 'segments' in data:
        for segment in data['segments']:
            text = []
            for word in segment['words']:
                if check(replaced_words, word):
                    text.append(vocoding_label + ' ' + word['word'])
                    cnt += 1
                else:
                    text.append(word['word'])
            segment['text'] = ' '.join(text)
        assert cnt == len(replaced_words), "Unmatched"
    else:
        print("Wrong format {:s}".format(json_path))
        sys.exit(1)
    
    with open(out_json, 'w') as f:
        json.dump(data, f, indent=2)
    
if __name__ == "__main__":
    #print(sys.argv[1])
    get_all_words(sys.argv[1], sys.argv[2])
    

