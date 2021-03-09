#!/usr/bin/env python
"""
data_warehouse

Simple tools to manage data from text file
"""
from __future__ import absolute_import

import os
import sys
import numpy as np

from core_scripts.other_tools import list_tools

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2021, Xin Wang"

class DataEntry:
    """DataEntry to store data for one entry
    """
    def __init__(self, data, tags, comment=""):
        """DataEntry(data, tags, comment)
        
        args: 
          data: any kind of python object
          tags: list of str, tags of the data entry
          comment: coment
        """
        self.data_value = data
        self.tags = self._parse_tag(tags)
        self.comment = comment
        
    def _parse_tag(self, tags):
        """[tag_1, tag_2, tag_3]
        """
        temp = {x:y for x, y in enumerate(tags)}
        return temp
    
    def get_value(self):
        return self.data_value
    
    def get_tag(self, tag_idx):
        return self.tags[tag_idx]
        
        
class DataWarehouse:
    """DataWarehouse to manage data with multi-view
    """
    def __init__(self, orig_file_path, parse_value_methods, parse_tag_methods):
        """DataWarehouse(orig_file_path, parse_methods)
        input:
          orig_file_path: str, path to the original file
          parse_methods: list of functions, to parse the data entry
        """
        self.file_path = orig_file_path
        self.parse_v_methods = parse_value_methods
        self.parse_t_methods = parse_tag_methods
        self.data_list = []
        self.tag_list = {}
        self.data_entries = self._parse_file()
        
    def _parse_file(self):
        data_content = list_tools.read_list_from_text(self.file_path)
        for data_entry in data_content:
            for parse_v_method, parse_t_method in \
                zip(self.parse_v_methods, self.parse_t_methods):
                data_value = parse_v_method(data_entry)
                tags = [x(data_entry) for x in parse_t_method]
                if data_value is None or None in tags:
                    continue
                tmp_data_entry = DataEntry(data_value, tags)
                self.data_list.append(tmp_data_entry)
                for tag_id, tag_val in enumerate(tags):
                    self._add_tag(tag_id, tag_val)
        return
    
    def _add_tag(self, tag_id, tag_val):
        if tag_id in self.tag_list:
            if not tag_val in self.tag_list[tag_id]:
                self.tag_list[tag_id].append(tag_val)
        else:
            self.tag_list[tag_id] = [tag_val]
        return
    
    def get_view(self, tag_idx, tag_value, score_parse = None):
        data_view = np.array([x.get_value() for x in self.data_list \
                              if x.get_tag(tag_idx) == tag_value])
        if score_parse is not None:
            data_view = np.array([score_parse(x) for x in data_view])
        return data_view
    
    def get_views(self, tag_idx, tag_values, score_parse=None, to_numpy=False):
        data_list = []
        for tag_value in tag_values:
            data_list.append(self.get_view(tag_idx, tag_value, score_parse))
            
        if to_numpy:
            num_col = len(data_list)
            num_row = max([len(x) for x in data_list])
            data = np.ones([num_row, num_col]) * np.inf
            for col_idx in range(num_col):
                length = data_list[col_idx].shape[0]
                data[0:length, col_idx] = data_list[col_idx]
            return data
        else:
            return data_list
    
    def get_tags(self, tag_idx):
        if tag_idx in self.tag_list:
            return self.tag_list[tag_idx]
        else:
            return None


if __name__ == "__main__":
    print("tools for data warehouse")
