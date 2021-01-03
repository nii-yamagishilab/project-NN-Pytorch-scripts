#!/usr/bin/env python
"""
Simple text processer for English

Based on https://github.com/fatchord/WaveRNN
"""

import os
import sys
import re

from core_scripts.data_io.text_process import toolkit_all

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2021, Xin Wang"

######
## Pool of symbols
######

# symbols
_pad = '_'
_punctuation = '!\'(),.:;? '
_special = '-'
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_skip_symbols = ['_', '~']

# ARPAbet symbols 
# http://www.speech.cs.cmu.edu/cgi-bin/cmudict?in=C+M+U+Dictionary
# vowels are combined with the three possible lexical stree symbols 0 1 2
# see http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b.symbols
_arpabet_symbols_raw = [
    'AA', 'AA0', 'AA1', 'AA2', 'AE', 'AE0', 'AE1', 'AE2', 
    'AH', 'AH0', 'AH1', 'AH2', 'AO', 'AO0', 'AO1', 'AO2', 
    'AW', 'AW0', 'AW1', 'AW2', 'AY', 'AY0', 'AY1', 'AY2',
    'B',  'CH',  'D',   'DH',  'EH', 'EH0', 'EH1', 'EH2', 
    'ER', 'ER0', 'ER1', 'ER2', 'EY', 'EY0', 'EY1', 'EY2', 
    'F',  'G', 'HH', 'IH', 'IH0', 'IH1', 'IH2', 'IY', 'IY0', 
    'IY1', 'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OW0', 
    'OW1', 'OW2', 'OY', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 
    'SH', 'T', 'TH', 'UH', 'UH0', 'UH1', 'UH2', 'UW', 'UW0', 
    'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH']
_arpabet_symbol_marker = '@'
_arpabet_symbols = [_arpabet_symbol_marker + x for x in _arpabet_symbols_raw]

# create pool of symbols
_symbols = [_pad] + list(_special) + list(_punctuation) \
           + list(_letters) + _arpabet_symbols

# create the mapping table
#  x+1 so that 0 can be reserved for other purposes
_symbol_to_index = {y: x+1 for x, y in enumerate(_symbols)}

def symbol_num():
    return len(_symbols)

def symbol2index(x):
    return _symbol_to_index[x]

def index2symbol(x):
    return _symbols[x-1]

#####
## Functions for text normalization
## I cannot write a full fludged text normalizer here.
## Just for place holder
#####
_whitespace_re = re.compile(r'\s+')

_number_map = {'1': 'one', '2': 'two', '3': 'three',
               '4': 'four', '5': 'five', '6': 'six', 
               '7': 'seven', '8': 'eight', '9': 'nine', '0': 'zero'}

def text_numbers(text):
    """ Place holder, just convert individual number to alphabet
    """
    def _tmp(tmp_text):
        if all([x in _number_map for x in tmp_text]):
            return ' '.join([_number_map[x] for x in tmp_text])
        else:
            return tmp_text
    tmp = ' '.join([_tmp(x) for x in text.split()])
    if text.startswith(' '):
        tmp = ' ' + tmp
    return tmp

def text_case_convert(text):
    """ By default, use lower case
    """
    return text.lower()

def text_whitespace_convert(text):
    """ Collapse all redundant white spaces
    e.g., 'qweq 1231   123151' -> 'qweq 1231 123151'
    """
    return re.sub(_whitespace_re, ' ', text)

def text_normalizer(text):
    """ Text normalizer

    In this code, only lower case conversion and white space is handled
    """
    return text_whitespace_convert(text_numbers(text_case_convert(text)))

#####
## Functions to convert symbol to index
#####


def flag_convert_symbol(symbol):
    """ check whether input symbol should be converted or not

    input
    -----
      symbol: str
    
    output
    ------
      bool
    """
    return symbol in _symbol_to_index and symbol not in _skip_symbols

def rawtext2indices(text):
    """ Look up the table and return the index for input symbol in input text
    
    input
    -----
      text: str
    
    output
    ------
      list of indices

    for example, 'text' -> [23, 16, 28, 23]
    """
    return [symbol2index(x) for x in text if flag_convert_symbol(x)]

def arpabet2indices(arpa_text):
    """ Look up the table and return the index for input symbol in input text

    input
    -----
      arpa_text: str
    
    output
    ------
      list of indices

    for example, 'AH HH' -> [12 19]
    """
    tmp = [_arpabet_symbol_marker + x for x in arpa_text.split()]
    return [symbol2index(x) for x in tmp if flag_convert_symbol(x)]
    
#####
## Main function
#####


def text2code(text):
    """ Convert English text and ARPAbet into code symbols (int)
    """
    if text.startswith(toolkit_all._curly_symbol):
        # phonemic annotation, no normalization
        return arpabet2indices(text.lstrip(toolkit_all._curly_symbol))
    else:
        # normal text, do normalization before conversion
        # text normalization
        text_normalized = text_normalizer(text)
        return rawtext2indices(text_normalized)
    # done

def code2text(codes):
    # x-1 because  _symbol_to_index
    txt_tmp = [index2symbol(x) for x in codes]
    txt_tmp = ''.join(txt_tmp)
    return text_whitespace_convert(txt_tmp.replace(_arpabet_symbol_marker, ' '))
    


if __name__ == "__main__":
    print("Definition of text processing toolkit for English")
