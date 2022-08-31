#!/usr/bin/env python
"""
Simple converted to convert text string into indices
Used for text-to-speech synthesis

Based on https://github.com/fatchord/WaveRNN
"""

import os
import sys
import re
import numpy as np

from core_scripts.other_tools import display as nii_warn
from core_scripts.data_io.text_process import toolkit_all
from core_scripts.data_io.text_process import toolkit_en
from core_scripts.other_tools import str_tools as nii_str_tk
from core_scripts.data_io import conf as nii_dconf

try:
    from g2p_en import G2p
    g_g2p_API = G2p()
except ModuleNotFoundError:
    #nii_warn.f_print("g2p_en is not found, we cannot use it", opt='warning')
    pass
    
    
__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2021, Xin Wang"



def text2code(text, flag_lang='EN'):
    """ Convert text string into code indices
    
    input
    -----
      text: string
      flag_lang: string, 'EN': English

    output
    ------
      code_seq: list of integers
    """
    code_seq = []

    # parse the curly bracket
    text_trunks = toolkit_all.parse_curly_bracket(text)

    # parse
    if flag_lang == 'EN':
        # English text
        for idx, text_trunk in enumerate(text_trunks):
            # whether this is the last trunk
            flag_eos = idx >= (len(text_trunks)-1)
            code_seq += toolkit_en.text2code(text_trunk, flag_eos)
    else:
        # unsupporte languages
        nii_warn.f_die("Error: text2code cannot handle {:s}".format(flag_lang))
    
    # convert to numpy format
    code_seq = np.array(code_seq, dtype=nii_dconf.h_dtype)

    return code_seq

def code2text(codes, flag_lang='EN'):
    """ Convert text string into code indices
    
    input
    -----
      code_seq: numpy arrays of integers
      flag_lang: string, 'EN': English

    output
    ------
      text: string
    """
    # convert numpy array backto indices
    codes_tmp = [int(x) for x in codes]

    output_text = ''
    if flag_lang == 'EN':
        output_text = toolkit_en.code2text(codes_tmp)
    else:
        nii_warn.f_die("Error: code2text cannot handle {:s}".format(flag_lang))
    return output_text

def symbol_num(flag_lang='EN'):
    """ Return the number of symbols defined for one language
    
    input
    -----
      flag_lange: string, 'EN': English

    output
    ------
      integer
    """
    if flag_lang == 'EN':
        return toolkit_en.symbol_num()
    else:
        nii_warn.f_die("Error: symbol_num cannot handle {:s}".format(flag_lang))
    return 0

def eos_index(flag_lang='EN'):
    """ Return the index for end of sentence symbol
    input
    -----
      flag_lange: string, 'EN': English

    output
    ------
      integer
    """
    if flag_lang == 'EN':
        return toolkit_en.eos_index()
    else:
        nii_warn.f_die("Error: eos_index cannot handle {:s}".format(flag_lang))
    return 0


def textloader(file_path, flag_lang='EN', g2p_tool=None):
    """ Load text and return the sybmol sequences
    input
    -----
      file_path: string, absolute path to the text file
      flag_lang: string, 'EN' by default, the language option to process text
    
    output
    ------
      output: np.array of shape (L), where L is the number of chars 
    """
    # load lines and chop '\n', join into one line
    text_buffer = [nii_str_tk.string_chop(x) for x in open(file_path, 'r')]
    text_buffer = ' '.join(text_buffer)

    # convert to indices
    if g2p_tool is None:
        return text2code(text_buffer, flag_lang)
    else:
        return g2p2code(text_buffer, flag_lang)
    

def g2p2code(text, flag_lang='EN'):
    """ Load text, do g2p, produce the indices
    The g2p tool is based on https://github.com/Kyubyong/g2p

    input
    -----
      text: string
      flag_lang: string, 'EN': English

    output
    ------
      code_seq: list of integers
    
    """
    # precheck whether input has ARPABet symbol
    text_trunks = toolkit_all.parse_curly_bracket(text)
    if len(text_trunks) > 1 or text_trunks[0][0] == toolkit_all._curly_symbol:
        # unsupporte languages
        nii_warn.f_die("Error: g2p2code don't process: {:s}".format(text))
    
    if flag_lang == 'EN':
        # g2p
        try:
            output = g_g2p_API(text)
        except NameError:
            nii_warn.f_die("G2p tool not found. Please install g2p_en")
            
        # remove the trailing space when there is 
        output = toolkit_en.g2poutput_process(output)
        # pack it inside '{ }' to be compatible with the API in toolkit_en
        output = '{' + '_'.join(output) + '}'
        # 
        code_seq = text2code(output, flag_lang)
    else:
        # unsupporte languages
        nii_warn.f_die("Error: text2code cannot handle {:s}".format(flag_lang))

    return code_seq


if __name__ == "__main__":
    print("Definition of text2code tools")
    text = 'hello we are {AY2_AY2_ _AY2_AY2} the same 123'
    indices = text2code(text)
    text2 = code2text(indices)
    print(text)
    print(indices)
    print(text2)

    print(code2text(textloader('./tmp.txt')))
