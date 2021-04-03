#!/usr/bin/env python
"""
seq_info

A class to log the information for one sample.
This data sequence could be one segment within a long utterance

"""
from __future__ import absolute_import

import os
import sys

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"


class SeqInfo():
    """ Definition of sequence information
    Save the information about one utterance (which may be a trunck from
    the original data utterance)
    """
    def __init__(self,
                 length = 0,
                 seq_name = '',
                 seg_idx = 0,
                 start_pos = 0,
                 info_id = 0):
        """
        Args:
            length: length this utterance segment
            seq_name: name of the utterance
            seg_idx: idx of this segment in the original utterance
            start_pos: from which step does this segment start in the
                       original utterance
            info_id: idx of this seq segment in training set
        """
        self.length = int(length)
        self.seq_name = seq_name
        self.seg_idx = seg_idx
        self.start_pos = int(start_pos)
        self.info_id = info_id
        
        
    def print_to_dic(self):
        """
        Print to dictionary format in order to dump
        """
        return {"length": self.length,
                "seq_name": self.seq_name,
                "seg_idx": self.seg_idx,
                "start_pos": self.start_pos,
                "info_id": self.info_id}
    
    def load_from_dic(self, dic):
        """
        Load seq informaiton from dictionary
        """
        try:
            self.length = dic["length"]
            self.seq_name = dic["seq_name"]
            self.seg_idx = dic["seg_idx"]
            self.start_pos = dic["start_pos"]
            self.info_id = dic["info_id"]
        except KeyError:
            nii_warn.f_die("Seq infor %s invalid" % str(dic))

    def print_to_str(self):
        """
        Print infor to str
        """
        temp = "{:d},{},{:d},{:d},{:d}".format(self.info_id, \
                                               self.seq_name, \
                                               self.seg_idx, \
                                               self.length, \
                                               self.start_pos)
        return temp

    def parse_from_str(self, input_str):
        """
        Parse a input string (which should be generated from print_to_str)
        """
        temp = input_str.split(',')
        self.seq_name = temp[1]
        try:
            self.info_id = int(temp[0])
            self.seg_idx = int(temp[2])
            self.length = int(temp[3])
            self.start_pos = int(temp[4])
        except ValueError:
            nii_warn.f_die("Seq infor cannot parse {}".format(input_str))
        return
    
    def seq_length(self):
        return self.length

    def seq_tag(self):
        return self.seq_name

    def seq_start_pos(self):
        return self.start_pos

############
### Util to parse the output from print_to_str
############

def parse_length(input_str):
    return int(input_str.split(',')[3])

def parse_filename(input_str):
    return input_str.split(',')[1]

    
if __name__ == "__main__":
    print("Definition of seq_info class")
