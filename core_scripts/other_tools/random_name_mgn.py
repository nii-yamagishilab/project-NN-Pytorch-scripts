#!/usr/bin/env python
"""
Random name manager

Used for produce protocols
"""
from __future__ import absolute_import

import os
import sys
from core_scripts.other_tools import list_tools
from core_scripts.data_io import io_tools

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2021, Xin Wang"

def list_loader(list_file):
    """ output_list = list_loader(list_file)
    Load a text file as a list of string. This function will use __cache to save
    the loaded list

    Args:
      list_file: str, path to the input text file
    Return:
      output_list: list,
    
    Note, a __cache will be created along list_file
    """
    cache_dir = os.path.join(os.path.dirname(list_file), '__cache')
    return io_tools.wrapper_data_load_with_cache(
        list_file, list_tools.read_list_from_text, cache_dir)

class RandomNameMgn:
    """ Class to manage the list of random file names
    """
    def __init__(self, file_random_name_list, verbose=False):
        """ RandomNameMgn(file_random_name_list, verbose=False)
        Create a random name manager
        
        Args:
          file_random_name_list: str, path to the text file of random names.
          verbose: bool, default False, print information during initialziation
        """
        if verbose:
            print("Loading random name tables")
            
        ## For unused random name list
        # load entries in the list
        self.unused_entries = list_loader(file_random_name_list)
        # prepare dictionary
        self.mapper = {x : None for x in self.unused_entries}
        # reverse dictionary
        self.mapper_rev = {}

        # print some informaiton
        if verbose:
            self.print_info()

        self.verbose = verbose
        # done
        return

    def print_info(self):
        mes = "Number of unused random file names: {:d}".format(
            len(self.unused_entries))
        print(mes)
        return

    def retrieve_rand_name(self, filename):
        """ rand_name = retrieve_rand_name(filename)
        
        filename: str, input, input file name
        rand_name: str, output, the random file name
        """
        if filename in self.mapper_rev:
            return self.mapper_rev[filename]
        else:
            rand_name = self.unused_entries.pop()
            self.mapper[rand_name] = filename
            self.mapper_rev[filename]= rand_name
            return rand_name

    def save_unused_name(self, save_file):
        """ save_unused_name(save_file)
        
        save_file: str, input, the path to save random names that
            have NOT been used
        """
        with open(save_file, 'w') as file_ptr:
            for entry in self.unused_entries:
                file_ptr.write(entry+'\n')
        
        if self.verbose:
            self.print_info()
            print("Save unused random names to {:s}".format(save_file))
        return
    
    def retrieve_filename(self, random_name):
        if random_name in self.mapper:
            return self.mapper[random_name]
        else:
            print("Random name {:s} has not been logged".format(random_name))
            sys.exit(1)
    
if __name__ == "__main__":
    print("")
