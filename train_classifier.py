#!/usr/bin/env python

"""train_classifier.py: provide an API for training models"""

import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader

from cnn import Net, CONFIG
from data_helper import HDF5Dataset

__author__ = "Yandi Xia"


def _get_variable(tensor, volatile=False):
    data = Variable(tensor, volatile=volatile)
    if torch.cuda.is_available():
        data = data.cuda()
    return data


def _train_loop(train_loader,
                test_loader,
                model,
                criterion,
                optimizer,
                n_epoch,
                print_every,
                eval_every,
                model_file):
    step = 0
    best_result = 0.
    best_step = 0
    best_epoch = 0
    model.train()
    print "Iteration starts..."
    for epoch in xrange(1, n_epoch + 1):
        print "Epoch#{}".format(epoch)
        for btch_dix, batch in enumerate(train_loader, 1):
            step += 1
            start_time = time.time()
            text, labels = batch
            text, labels = _get_variable(text), _get_variable(labels).view(-1)
            optimizer.zero_grad()
            outputs = model(text)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            duration = time.time() - start_time
            if step % print_every == 0:
                print "step: {}".format(step)
                print "batch: {}".format(btch_dix)
                print "loss: {}".format(loss.data[0])
                print "training speed: {}/sec".format(labels.size()[0] / duration)
            if step % eval_every == 0:
                start_time = time.time()
                model.eval()
                total = 0.
                correct = 0.
                for test_batch in test_loader:
                    text, labels = test_batch
                    text, labels = _get_variable(text, volatile=True), _get_variable(labels, volatile=True).view(-1)
                    outputs = model(text)
                    _, predicted = torch.max(outputs.data, dim=1)
                    total += labels.data.size()[0]
                    correct += (predicted == labels.data).sum()
                acc = correct / total
                model.train()
                duration = time.time() - start_time
                if best_result < acc:
                    best_result = acc
                    best_epoch = epoch
                    best_step = step
                    torch.save({
                        "epoch": epoch,
                        "step": step,
                        "state_dict": model.state_dict(),
                        "acc": acc
                    }, model_file)
                print "*" * 20 + "current acc: {}".format(acc) + " | best acc: {}".format(best_result)
                print "*" * 20 + "best epoch/step: {}/{}".format(best_epoch, best_step)
                print "*" * 20 + "testing speed: {}/sec".format(total / duration)
                print "*" * 20 + "current epoch: {}".format(epoch)


def train(train_files,
          test_files,
          train_batch_size,
          eval_batch_size,
          model_file,
          vocab_size,
          num_classes,
          n_epoch,
          print_every=50,
          eval_every=500):
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.backends.cudnn.benchmark = True
    print "Setting seed..."
    seed = 1234
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # setup CNN model
    CONFIG["vocab_size"] = vocab_size
    CONFIG["num_classes"] = num_classes
    model = Net()

    if torch.cuda.is_available():
        print "CUDA is available on this machine. Moving model to GPU..."
        model.cuda()
    print model

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    train_set = HDF5Dataset(train_files)
    test_set = HDF5Dataset(test_files)

    train_loader = DataLoader(dataset=train_set,
                              batch_size=train_batch_size,
                              shuffle=True,
                              num_workers=2)

    test_loader = DataLoader(dataset=test_set,
                             batch_size=eval_batch_size,
                             num_workers=2)

    _train_loop(train_loader=train_loader,
                test_loader=test_loader,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                n_epoch=n_epoch,
                print_every=print_every,
                eval_every=eval_every,
                model_file=model_file)
