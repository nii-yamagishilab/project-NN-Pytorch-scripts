#!/usr/bin/env python
"""
customize_collate_fn

Customized collate functions for DataLoader, based on 
github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py

PyTorch is BSD-style licensed, as found in the LICENSE file.
"""

from __future__ import absolute_import

import os
import sys
import torch
import re
from torch._six import container_abcs, string_classes, int_classes

"""
The primary motivation is to handle batch of data with varied length.
Default default_collate cannot handle that because of stack:
github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py

Here we modify the default_collate to take into consideration of the 
varied length of input sequences in a single batch.

Notice that the customize_collate_fn only pad the sequences.
For batch input to the RNN layers, additional pack_padded_sequence function is 
necessary. For example, this collate_fn does something similar to line 56-66,
but not line 117 in this repo:
https://gist.github.com/HarshTrivedi/f4e7293e941b17d19058f6fb90ab0fec
"""

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"


np_str_obj_array_pattern = re.compile(r'[SaUO]')

customize_collate_err_msg = (
    "customize_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


def pad_sequence(batch, padding_value=0.0):
    """ pad_sequence(batch)
    
    Pad a sequence of data sequences to be same length.
    Assume batch = [data_1, data2, ...], where data_1 has shape (len, dim, ...)
    
    This function is based on 
    pytorch.org/docs/stable/_modules/torch/nn/utils/rnn.html#pad_sequence
    """
    max_size = batch[0].size()
    trailing_dims = max_size[1:]
    max_len = max([s.size(0) for s in batch])
    
    if all(x.shape[0] == max_len for x in batch):
        # if all data sequences in batch have the same length, no need to pad
        return batch
    else:
        # we need to pad
        out_dims = (max_len, ) + trailing_dims
        
        output_batch = []
        for i, tensor in enumerate(batch):
            # check the rest of dimensions
            if tensor.size()[1:] != trailing_dims:
                print("Data in batch has different dimensions:")
                for data in batch:
                    print(str(data.size()))
                raise RuntimeError('Fail to create batch data')
            # save padded results
            out_tensor = tensor.new_full(out_dims, padding_value)
            out_tensor[:tensor.size(0), ...] = tensor
            output_batch.append(out_tensor)
        return output_batch


def customize_collate(batch):
    """ customize_collate(batch)
    
    Collate a list of data into batch. Modified from default_collate.
    
    """

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        # this is the main part to handle varied length data in a batch
        # batch = [data_tensor_1, data_tensor_2, data_tensor_3 ... ]
        # 
        batch_new = pad_sequence(batch)
        
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy

            # allocate the memory based on maximum numel
            numel = max([x.numel() for x in batch_new]) * len(batch_new)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch_new, 0, out=out)

    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(customize_collate_err_msg.format(elem.dtype))
            # this will go to loop in the last case
            return customize_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
        
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: customize_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(customize_collate(samples) \
                           for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in batch should be of equal size')
        
        # zip([[A, B, C], [a, b, c]])  -> [[A, a], [B, b], [C, c]]
        transposed = zip(*batch)
        return [customize_collate(samples) for samples in transposed]

    raise TypeError(customize_collate_err_msg.format(elem_type))



def customize_collate_from_batch(batch):
    """ customize_collate_existing_batch
    Similar to customize_collate, but input is a list of batch data that have
    been collated through customize_collate.
    The difference is use torch.cat rather than torch.stack to merge tensors.
    Also, list of data is directly concatenated

    This is used in customize_dataset when merging data from multiple datasets.
    It is better to separate this function from customize_collate
    """

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        batch_new = pad_sequence(batch)        
        out = None
        if torch.utils.data.get_worker_info() is not None:
            numel = max([x.numel() for x in batch_new]) * len(batch_new)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        # here is the difference
        return torch.cat(batch_new, 0, out=out)

    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(customize_collate_err_msg.format(elem.dtype))
            return customize_collate_from_batch(
                [torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, tuple):
        # concatenate two tuples
        tmp = elem
        for tmp_elem in batch[1:]:
            tmp += tmp_elem 
        return tmp
    elif isinstance(elem, container_abcs.Sequence):
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in batch should be of equal size')
        transposed = zip(*batch)
        return [customize_collate_from_batch(samples) for samples in transposed]

    raise TypeError(customize_collate_err_msg.format(elem_type))


if __name__ == "__main__":
    print("Definition of customized collate function")
