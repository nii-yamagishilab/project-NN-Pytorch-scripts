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
import collections

#from torch._six import container_abcs, string_classes, int_classes
from torch._six import string_classes

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
    """ output_batch = pad_sequence(batch)
    
    input
    -----
      batch: list of tensor, [data_1, data2, ...], and data_1 is (len, dim, ...)
    
    output
    ------
      output_batch: list of tensor, [data_1_padded, data_2_padded, ...]

    Pad a batch of data sequences to be same length (maximum length in batch).
    This function is based on 
    pytorch.org/docs/stable/_modules/torch/nn/utils/rnn.html#pad_sequence.

    Output list of tensor can be stacked into (batchsize, len, dim,...).
    See customize_collate(batch) below
    """
    # get the rest of the dimensions (dim, ...)
    dim_size = batch[0].size()
    trailing_dims = dim_size[1:]

    # get the maximum length
    max_len = max([s.size(0) for s in batch])
    
    if all(x.shape[0] == max_len for x in batch):
        # if all data sequences in batch have the same length, no need to pad
        return batch
    else:
        # else, we need to pad 
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
            # updated according to latest collate function
            # otherwise, it raises warning
            # pytorch/blob/master/torch/utils/data/_utils/collate.py
            out = elem.new(storage).resize_(
                len(batch_new), *list(batch_new[0].size()))
            #print(batch_new.shape[0], batch_new.shape[1])
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
    #elif isinstance(elem, int_classes):
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    #elif isinstance(elem, container_abcs.Mapping):
    elif isinstance(elem, collections.abc.Mapping):
        return {key: customize_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(customize_collate(samples) \
                           for samples in zip(*batch)))
    #elif isinstance(elem, container_abcs.Sequence):
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in batch should be of equal size')
        
        # zip([[A, B, C], [a, b, c]])  -> [[A, a], [B, b], [C, c]]
        transposed = zip(*batch)
        return [customize_collate(samples) for samples in transposed]

    raise TypeError(customize_collate_err_msg.format(elem_type))




def pad_sequence_batch(list_batch, padding_value=0.0):
    """ output_batch = pad_sequence(list_batch)

    input
    -----
      batch: list of batch, [batch_1, batch_2, ...], and batch_1 is 
      (batch_size, len, dim1, dim2, ...)
    
    output
    ------
      output_batch: list of tensor, [batch_1_padded, batch_2_padded, ...]

    Different from pad_sequence, list_batch is a list of batched tensors
    """
    # each batched_tensor has shape (batch, length, dim1, ...)
    #  get dimensions for (dim1, ...)
    dim_size = list_batch[0].size()

    if len(dim_size) <= 2:
        return list_batch

    trailing_dims = dim_size[2:]
    
    # get the maximum length for each batched tensor
    max_len = max([s.size(1) for s in list_batch])
    
    if all(x.shape[1] == max_len for x in list_batch):
        # if all data sequences in batch have the same length, no need to pad
        return list_batch
    else:
        output_batch = []
        for i, tensor in enumerate(list_batch):

            # shape (batch, max_len, dim1, dim2, ...)
            out_dims = (tensor.shape[0], max_len, ) + trailing_dims
        
            # check the rest of dimensions
            if tensor.size()[2:] != trailing_dims:
                print("Data in batch has different dimensions:")
                raise RuntimeError('Fail to pad batched data')

            # save padded results
            out_tensor = tensor.new_full(out_dims, padding_value)
            out_tensor[:, :tensor.size(1), ...] = tensor
            output_batch.append(out_tensor)
        return output_batch


def customize_collate_from_batch(batch):
    """ output = customize_collate_from_batch
    
    input
    -----
      batch: list of tensor, [tensor1, tensor2, ...], where
             each tensor has shape (batch, length, dim1, dim2, ...)

    output
    ------
      output: tensor (batch_sum, length, dim1, dim2, ...)
    
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
        batch_new = pad_sequence_batch(batch)        
        out = None
        if torch.utils.data.get_worker_info() is not None:
            numel = max([x.numel() for x in batch_new]) * len(batch_new)
            storage = elem.storage()._new_shared(numel)
            # we need to resize_ to suppress a warning
            # this is based on 
            # pytorch/blob/master/torch/utils/data/_utils/collate.py
            # [batch_1, length, dim], [batch_2, length, dim] ...
            #   batch_new[0][0].size() -> length, dim, ...
            #   [x.shape[0] for x in batch_new] -> [batch_1, batch_2, ...]
            out = elem.new(storage).resize_(
                sum([x.shape[0] for x in batch_new]), 
                *list(batch_new[0][0].size()))

        # here is the difference
        # concateante (batch1, length, dim1, dim2, ...) (batch2, length, ...)
        #  into (batch1+batch2+..., length, dim1, dim2, ...)
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
    #elif isinstance(elem, int_classes):
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, tuple):
        # concatenate two tuples
        tmp = elem
        for tmp_elem in batch[1:]:
            tmp += tmp_elem 
        return tmp
    #elif isinstance(elem, container_abcs.Sequence):
    elif isinstance(elem, collections.abc.Sequence):
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in batch should be of equal size')
        transposed = zip(*batch)
        return [customize_collate_from_batch(samples) for samples in transposed]

    raise TypeError(customize_collate_err_msg.format(elem_type))


if __name__ == "__main__":
    print("Definition of customized collate function")
