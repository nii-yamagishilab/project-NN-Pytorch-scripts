#!/usr/bin/env python
"""Utilities for speechbrain collate functions

Adopted from speechbrain PaddedBatch

Motivation:
In ASV system with multiple enrollments, 
we may need to load multiple enrollments for each trial.
We can simply concatenate enrollments into a one tensor along the time axis,
in shape (L_1 + L_2 + L_3 ... L_N).
but this is not very convenient to segment them later.

Another option is to stack the enrollments like in shape (N, max_n(L_n)).
PaddBatch will further stack them as (B, N, max_n(L_n)) for the mini-batch,
given the 1, 2, ..., B trials.

However, N varies for different trials in the mini-batch. 
The N for the 1st trial may be 1 (only 1 enrollment), while the N for the 2nd
trial may be 3 (i.e., 3 enrollments). 
PaddBatch will raise an error to stack 

The goal here is to pad and concatate (N, max_n(L_n)) along the 1st dimension,
in shape (N_1 + N_2 + N_3 ... + N_B, max(L)).
"""
from __future__ import absolute_import

import itertools
import collections

import torch
import torch.nn as nn

from speechbrain.utils.data_utils import mod_default_collate
from speechbrain.utils.data_utils import recursive_to
from speechbrain.utils.data_utils import batch_pad_right
from torch.utils.data._utils.collate import default_convert
from torch.utils.data._utils.pin_memory import (
    pin_memory as recursive_pin_memory,
)




############
# data IO
############
def batch_list_pad_right(tensors: list, mode="constant", value=0):
    """tensor = batch_list_pad_right(tensors, mode, value)

    wrapper of batch_pad_right to handle an additional case where input 
    is a list of tensors
    
    input
    -----
      tensors: list of tensors, or list of list of tensors
      mode: str,        method for padding, default "constant"
      value: float,     dummy value used to pad
    
    output
    ------
      tensor:  pad and stacked tensors
    """
    if type(tensors[0]) is list:
        # if tensors is a list of list of tensors, chain them together
        # chain([[tensor1, tensor2, tensor3], [tensor4, tensor5]])
        # -> [tensor1, tensor2, tensor3, tensor4, tensor5]
        data_concat = list(itertools.chain.from_iterable(tensors))
    else:
        data_concat = tensors
    return batch_pad_right(data_concat, mode, value)
    

# copied from speechbrain code
PaddedData = collections.namedtuple("PaddedData", ["data", "lengths"])

class PaddedBatch_customize:
    """
    """
    def __init__(
        self,
        examples,
        padded_keys=None,
        device_prep_keys=None,
        padding_func=batch_list_pad_right,
        padding_kwargs={},
        apply_default_convert=True,
        nonpadded_stack=True,
    ):
        self.__length = len(examples)
        self.__keys = list(examples[0].keys())
        self.__padded_keys = []
        self.__device_prep_keys = []
        for key in self.__keys:
            values = [example[key] for example in examples]
            # Default convert usually does the right thing (numpy2torch etc.)
            if apply_default_convert:
                values = default_convert(values)
            
            # change: the thrid condition check whether values is a list of list
            if (padded_keys is not None and key in padded_keys) \
               or (padded_keys is None and isinstance(values[0],torch.Tensor)) \
               or (padded_keys is None and isinstance(values[0], list) and 
                   isinstance(values[0][0], torch.Tensor)):
                # Padding and PaddedData
                self.__padded_keys.append(key)
                padded = PaddedData(*padding_func(values, **padding_kwargs))
                setattr(self, key, padded)
            else:
                # Default PyTorch collate usually does the right thing
                # (convert lists of equal sized tensors to batch tensors, etc.)
                if nonpadded_stack:
                    values = mod_default_collate(values)
                setattr(self, key, values)

            # change: the third condition check whether values is a list of list
            if (device_prep_keys is not None and key in device_prep_keys) \
               or (device_prep_keys is None and isinstance(values[0], torch.Tensor)) \
               or (device_prep_keys is None and isinstance(values[0], list) and
                   isinstance(values[0][0], torch.Tensor)):
                self.__device_prep_keys.append(key)

    def __len__(self):
        return self.__length

    def __getitem__(self, key):
        if key in self.__keys:
            return getattr(self, key)
        else:
            raise KeyError(f"Batch doesn't have key: {key}")

    def __iter__(self):
        """Iterates over the different elements of the batch.

        Example
        -------
        >>> batch = PaddedBatch([
        ...     {"id": "ex1", "val": torch.Tensor([1.])},
        ...     {"id": "ex2", "val": torch.Tensor([2., 1.])}])
        >>> ids, vals = batch
        >>> ids
        ['ex1', 'ex2']
        """
        return iter((getattr(self, key) for key in self.__keys))

    def pin_memory(self):
        """In-place, moves relevant elements to pinned memory."""
        for key in self.__device_prep_keys:
            value = getattr(self, key)
            pinned = recursive_pin_memory(value)
            setattr(self, key, pinned)
        return self

    def to(self, *args, **kwargs):
        """In-place move/cast relevant elements.

        Passes all arguments to torch.Tensor.to, see its documentation.
        """
        for key in self.__device_prep_keys:
            value = getattr(self, key)
            moved = recursive_to(value, *args, **kwargs)
            setattr(self, key, moved)
        return self

    def at_position(self, pos):
        """Gets the position."""
        key = self.__keys[pos]
        return getattr(self, key)

    @property
    def batchsize(self):
        """Returns the bach size"""
        return self.__length

if __name__ == "__main__":
    print(__doc__)
