#!/usr/bin/env python

"""random_data_generator: """

import math
import h5py
import numpy as np

__author__ = "Yandi Xia"


def write_hdf5_file(features, labels, hdf5_output):
    """
    func:
            output features and labels into hdf5 file
    :param hdf5_output:
    :return:
    """
    f = h5py.File(hdf5_output, "w")
    dset = f.create_dataset("text", shape=features.shape, dtype="int64")
    dset[...] = features
    dset = f.create_dataset("label", shape=labels.shape, dtype="int64")
    dset[...] = labels
    f.close()


def generate_batch_of_random_data_files(vocab_size,
                                        label_size,
                                        max_len,
                                        num_instances,
                                        num_split,
                                        hdf5_output_prefix):
    """
    func:
            generate a batch of data files
    :param vocab_size:
            each word index is in range(0, vocab_size)
    :param label_size:
            each label index is in range(0, label_size)
    :param max_len:
            the shape of features is [num_instances, max_len]
    :param num_instances:
    :param num_split:
            how many files in the batch
    :param hdf5_output_prefix:
            the prefix that the files share
    :return:
            a list of file names
    """
    rng = np.random.RandomState(1234)
    features = rng.randint(0, vocab_size, (num_instances, max_len), dtype="int64")
    labels = rng.randint(0, label_size, (num_instances,), dtype="int64")

    data_size = int(math.ceil(num_instances / num_split))

    file_suffix = ".hdf5"

    file_list = []
    for i in xrange(num_split):
        file_name = hdf5_output_prefix + str(i) + file_suffix
        write_hdf5_file(features=features[i * data_size: (i + 1) * data_size, :],
                        labels=labels[i * data_size: (i + 1) * data_size],
                        hdf5_output=file_name)
        file_list.append(file_name)
    return file_list
