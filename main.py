#!/usr/bin/env python

"""main: """

from random_data_generator import generate_batch_of_random_data_files
from train_classifier import train

__author__ = "Yandi Xia"

vocab_size = 20518

label_size = 13

max_len = 88

num_train_instances = 755264
train_file_prefix = "train_"
num_train_files = 1

num_test_instances = 94308
test_file_prefix = "test_"
num_test_files = 1

# generate train hdf5 files
print "Generating random training data..."
train_file_list = generate_batch_of_random_data_files(vocab_size=vocab_size,
                                                      label_size=label_size,
                                                      max_len=max_len,
                                                      num_instances=num_train_instances,
                                                      num_split=num_train_files,
                                                      hdf5_output_prefix=train_file_prefix)
# generate test hdf5 files
print "Generating random testing data..."
test_file_list = generate_batch_of_random_data_files(vocab_size=vocab_size,
                                                     label_size=label_size,
                                                     max_len=max_len,
                                                     num_instances=num_train_instances,
                                                     num_split=num_test_files,
                                                     hdf5_output_prefix=test_file_prefix)

train(train_files=train_file_list,
      test_files=test_file_list,
      train_batch_size=128,
      eval_batch_size=128,
      model_file="model.gz",
      vocab_size=vocab_size,
      num_classes=label_size,
      n_epoch=10,
      print_every=50,
      eval_every=500)
