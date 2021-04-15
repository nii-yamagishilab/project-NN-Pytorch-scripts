#!/usr/bin/env python
"""
data_warehouse

Simple tools to manage data from text file
"""
from __future__ import absolute_import

import os
import sys
import itertools
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
        """[tag_1, tag_2, tag_3] -> {1: tag1, 2: tag2, 3: tag3}
        """
        temp = {x:y for x, y in enumerate(tags)}
        return temp
    
    def get_value(self):
        return self.data_value
    
    def get_tag(self, tag_idx):
        return self.tags[tag_idx]
    
    def check_tags(self, tag_indices, tag_values):
        """check_tags(tag_indices, tag_values)
        check whether the specified tag is equal to the tag value
        
        input:
          tag_indices: list, self.tags[tag_index] should be accessible
          tag_values: list, self.tags[tag_index] == tag_value?
        
        output:
          True: if tag_values are matched with tags if this data
        """
        for tag_idx, tag_value in zip(tag_indices, tag_values):
            if self.tags[tag_idx] != tag_value:
                return False
        return True
        
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
        # load list
        data_content = list_tools.read_list_from_text(self.file_path)
        
        for data_entry in data_content:
            # iterate over parse methods
            for parse_v_method, parse_t_method in \
                zip(self.parse_v_methods, self.parse_t_methods):
                
                # get value
                data_value = parse_v_method(data_entry)
                # get tag
                tags = [x(data_entry) for x in parse_t_method]
                
                # skip invalid line
                if data_value is None or None in tags:
                    continue
                    
                # create data entry
                tmp_data_entry = DataEntry(data_value, tags)
                self.data_list.append(tmp_data_entry)
                
                # add tag to the self.tag_list
                for tag_id, tag_val in enumerate(tags):
                    self._add_tag(tag_id, tag_val)
        return
    
    def _add_tag(self, tag_id, tag_val):
        # collect all possible tags for the tag_id-th tag
        if tag_id in self.tag_list:
            if not tag_val in self.tag_list[tag_id]:
                self.tag_list[tag_id].append(tag_val)
        else:
            self.tag_list[tag_id] = [tag_val]
        return
    
    
    def get_view(self, tag_idxs, tag_values, score_parse = None):
        """ get_view(tag_idxs, tag_values, score_parse = None)
        
        input:
          tag_idxs: list, the index of the tag slot to check
          tag_values: list, the value of the tag slot to compare
          score_parse: function, a function to extract score from entry
        
        output:
          data_view: list of data
        """
        
        data_view = [x.get_value() for x in self.data_list \
                     if x.check_tags(tag_idxs, tag_values)]
        if score_parse is not None:
            return [score_parse(x) for x in data_view]
        else:
            return data_view
    
    def _to_numpy(self, data_list, dims, statistics):
        """ convert data_list to numpy
        """
        # maximum length of one data entry
        max_length = max([len(x) for x in data_list])
        # create data array
        if statistics is None:
            data_array = np.ones([np.prod(dims), max_length]) * np.inf

            for idx, data_entry in enumerate(data_list):
                data_array[idx, 0:len(data_entry)] = np.array(data_entry)
            return np.reshape(data_array, dims + [max_length])
        else:
            data_array = np.ones([np.prod(dims)])

            for idx, data_entry in enumerate(data_list):
                if data_entry:
                    data_array[idx] = statistics(data_entry)
            return np.reshape(data_array, dims)
       
    
    def get_views_cross(self, tag_idxs, tag_values, 
                        score_parse=None, to_numpy=False, statistics=None):
        """get_views_cross(self, tag_idxs, tag_values, 
                           score_parse=None, to_numpy=False, statistics=None)
        input:
          tag_idxs: list, list of tag indices to check
          tag_values: list of list, for each tag_index, 
              A list of tags will be created through this cross:
              tag_values[0] x tag_values[1] x ... 
              
              Then, each combination is used to retrieve the data
              output data will be a tensor of 
                 [len(tag_values[0]), len(tag_values[1]), ...]
          
        output:
           data_list:
        """
        data_list = []
        data_mat_size = [len(x) for x in tag_values]
        
        tag_iter = itertools.product(*tag_values)
        for tag_ent in tag_iter:
            data_list.append(self.get_view(tag_idxs, tag_ent, score_parse))
        
        if to_numpy:
            return self._to_numpy(data_list, data_mat_size, statistics)
        else:
            return data_list
        
    def get_tags(self, tag_idx):
        if tag_idx in self.tag_list:
            return self.tag_list[tag_idx]
        else:
            return None

if __name__ == "__main__":
    print("tools for data warehouse")
