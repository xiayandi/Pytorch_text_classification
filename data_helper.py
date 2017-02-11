#!/usr/bin/env python

"""data_helper: the read/write module for the classifiers. """

import h5py
from torch.utils.data import Dataset


__author__ = "Yandi Xia"


def _load_hdf5_file(hdf5_file):
    f = h5py.File(hdf5_file, "r")
    data = [f[key] for key in f.keys()]
    return tuple(data)


class HDF5Dataset(Dataset):
    def __init__(self, data_files):
        self.data_files = sorted(data_files)

    def __getitem__(self, index):
        return _load_hdf5_file(self.data_files[index])

    def __len__(self):
        return len(self.data_files)


