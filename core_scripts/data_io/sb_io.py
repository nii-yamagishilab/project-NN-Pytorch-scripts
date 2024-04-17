#!/usr/bin/env python
"""
data_io based speechbrain

Adopted from speechbrain.core

"""
from __future__ import absolute_import

import speechbrain as sb
from speechbrain.dataio.sampler import ReproducibleRandomSampler


class SBDatasetWrapper:
    """
    """
    def __init__(self, dataset, flag_train, loader_kwargs):
        # link dataset
        self.dataset = dataset
        # 
        self.flag_train = flag_train
        # create data loader
        self.data_loader = make_dataloader(dataset, flag_train, **loader_kwargs)
        # name tagger
        self.db_type = 'speechbrain'
        return
        
    def print_info(self):
        return

    def get_loader(self):
        return self.data_loader
    
    def get_seq_num(self):
        return len(self.dataset.data)

    def putitem(self, data_gen, save_dir, name, seq_info):
        return

def make_dataloader(dataset, flag_train, **loader_kwargs):
    """Creates DataLoaders for Datasets.
    
        This is used by ``fit()`` and ``evaluate()`` if they just receive
        Datasets.

        Alternatively, this can be called from outside the Brain subclass.
        In that case, the DataLoader should be passed to ``fit()`` in place
        of the dataset.

        The Stage.TRAIN DataLoader is handled specially. It has extra args for
        shuffle and drop_last. In DDP a DistributedSampler is created (unless
        the dataset is an IterableDataset).

        NOTE
        ----
        Some important DataLoader arguments are passed via **loader_kwargs,
        e.g., batch_size, num_workers, pin_memory.

        NOTE
        ----
        By default, ``evaluate()`` specifies ckpt_prefix=None to stop the test
        DataLoader being added to the checkpointer. If you need to add a
        recoverable after saving checkpoints (e.g., at test time, after
        checkpointing the training), and still be able to recover reasonably,
        you should probably specify ``allow_partial_load=True``.

        Arguments
        ---------
        dataset : Dataset
            A set of data to use to create data loader. If the Dataset is a
            DynamicItemDataset, PaddedBatch is used as the default collate_fn,
            unless specified in loader_kwargs.
        stage : Stage
            The stage of the experiment: Stage.TRAIN, Stage.VALID, Stage.TEST
        ckpt_prefix : str, None
            Prefix to use for SaveableDataLoader Checkpoint name. The Stage
            name is added to this to create the full key. Set to None to not
            save the DataLoader.
        **loader_kwargs : dict
            Additional keyword arguments to the DataLoader.
            E.g., batch_size, num_workers, pin_memory.
    """
    # TRAIN stage is handled specially.
    if flag_train:
        loader_kwargs = _train_loader_specifics(dataset, loader_kwargs)
    # This commented-out code block is useful when one can ensure
    # metric reporting is DDP-valid for VALID & EVAL datasets.
    # elif self.distributed_launch:
    #     loader_kwargs = sb.dataio.dataloader.distributed_loader_specifics(
    #         self.distributed_launch, self.rank, dataset, loader_kwargs
    #     )
    dataloader = sb.dataio.dataloader.make_dataloader(
        dataset, **loader_kwargs
    )

    return dataloader

def _train_loader_specifics(dataset, loader_kwargs):
    sampler = loader_kwargs.get("sampler", None)
    # Shuffling should really only matter for the train stage. Shuffling
    # will also lead to more padding in batches if the order was otherwise
    # sorted by length.
    shuffle = loader_kwargs.get("shuffle", False)
    if shuffle:
        if sampler is not None:
            raise ValueError(
                "Cannot specify both shuffle=True"
                "and a sampler in loader_kwargs"
            )
        sampler = ReproducibleRandomSampler(dataset)
        train_sampler = sampler
        loader_kwargs["sampler"] = train_sampler
        # Delete the shuffle flag, since you cannot specify both a sampler and
        # shuffling:
        del loader_kwargs["shuffle"]
    return loader_kwargs



if __name__ == "__main__":
    print(__doc__)
