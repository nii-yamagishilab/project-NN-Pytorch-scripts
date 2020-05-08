#!/usr/bin/env python
"""
config_parse

Configuration parser

"""
from __future__ import absolute_import

import os
import sys
import configparser

import core_scripts.other_tools.list_tools as nii_list_tools
import core_scripts.other_tools.display as nii_display

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"


class ConfigParse:
    """ ConfigParse
    class to parse input configuration file
    
    
    """
    def __init__(self, config_path):
        """ initialization
        """
        # get configuration path
        self.m_config_path = None
        if os.path.isfile(config_path):
            self.m_config_path = config_path
        else:
            nii_display.f_die("Cannot find %s" % (config_path), 'error')
    
        # path configuration file
        self.m_config = self.f_parse()
        if self.m_config is None:
            nii_display.f_die("Fail to parse %s" % (config_path), 'error')
            
        # done
        return
            
    def f_parse(self):
        """ f_parse
        parse the configuration file
        """
        if self.m_config_path is not None:
            tmp_config = configparser.ConfigParser()
            tmp_config.read(self.m_config_path)
            return tmp_config
        else:
            nii_display.f_print("No config file provided", 'error')
            return None

    def f_retrieve(self, keyword, section_name=None, config_type=None):
        """ f_retrieve(self, keyword, section_name=None, config_type=None)
        retrieve the keyword from config file
        
        Return:
           value: string, int, float
        
        Parameters:
           keyword: 'keyword' to be retrieved
           section: which section is this keyword in the config. 
                    None will search all the config sections and 
                    return the first
           config_type: which can be 'int', 'float', or None.
                    None will return the value as a string
        """
        tmp_value = None
        if section_name is None:
            # if section is not given, search all the sections
            for section_name in self.m_config.sections():
                tmp_value = self.f_retrieve(keyword, section_name, \
                                            config_type)
                if tmp_value is not None:
                    break
        elif section_name in self.m_config.sections() or \
             section_name == 'DEFAULT':
            tmp_sec = self.m_config[section_name]
            # search a specific section
            if config_type == 'int':
                tmp_value = tmp_sec.getint(keyword, fallback=None)
            elif config_type == 'float':
                tmp_value = tmp_sec.getfloat(keyword, fallback=None)
            elif config_type == 'bool':
                tmp_value = tmp_sec.getboolean(keyword, fallback=None)
            else:
                tmp_value = tmp_sec.get(keyword, fallback=None)
        else:
            nii_display.f_die("Unknown section %s" % (section_name))
        return tmp_value

