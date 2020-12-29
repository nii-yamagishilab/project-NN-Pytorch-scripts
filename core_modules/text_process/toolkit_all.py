#!/usr/bin/env python
"""
Simple text processer for all languages

Based on https://github.com/fatchord/WaveRNN
"""

import os
import sys
import re

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2021, Xin Wang"

#####
## Parse the curly bracket
##### 

# from https://github.com/fatchord/WaveRNN
_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')

# symbol to indicate phonemic annotation
_curly_symbol = '{'

def parse_curly_bracket(text):
    """ Prase the text based on curly brackets
    Inspired by https://github.com/fatchord/WaveRNN: when input text
    is mixed with raw text and phonemic annotation, the {} pair indicates
    the phonemic part
    
    input
    -----
      text: str
    
    output
    ------
      text_list: list of str

    For example, 'text {AH II} test' -> ['text ', 'AH II', ' test']
    """
    text_list = []
    text_tmp = text

    while len(text_tmp):
        re_matched = _curly_re.match(text_tmp)
        
        if re_matched:
            # e.g., 'text {AH II} test'
            # group(1), group(2) -> ['text ', 'AH II']
            text_list.append(re_matched.group(1))
            text_list.append(_curly_symbol + re_matched.group(2))
            # group(3) -> ' test'
            text_tmp = re_matched.group(3)
        else:
            text_list.append(text_tmp)
            break
    return text_list


if __name__ == "__main__":
    print("Definition of text processing tools for all languages")
