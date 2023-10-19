#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import time
import copy
import uuid

import crypten
import random
import torch
import numpy as np

import crypten.communicator as comm
import torch.distributed as dist

from utils.dataset.dataset_linear_svm import dataset_svm
from utils.dataset.dataset_linear_svm import dataset_load_linear_svm
from crypten.common.rng import generate_random_ring_element
from crypten.common.tensor_types import is_float_tensor, is_int_tensor, is_tensor
from crypten.encoder import FixedPointEncoder
from examples.meters import AverageMeter


def train_linear_svm(features, labels, epochs=50, lr=0.5, print_time=False):
    # Initialize random weights
    w = features.new(torch.randn(1, features.size(0)))
    b = features.new(torch.randn(1))

    if print_time:
        pt_time = AverageMeter()
        end = time.time()

    for epoch in range(epochs):
        # Forward
        label_predictions = w.matmul(features).add(b).sign()

        # Compute accuracy
        correct = label_predictions.mul(labels)
        accuracy = correct.add(1).div(2).mean()
        if crypten.is_encrypted_tensor(accuracy):
            accuracy = accuracy.get_plain_text()

        # Print Accuracy once
        if crypten.communicator.get().get_rank() == 0:
            print(
                f"Epoch {epoch} --- Training Accuracy %.2f%%" % (accuracy.item() * 100)
            )

        # Backward
        loss_grad = -labels * (1 - correct) * 0.5  # Hinge loss
        b_grad = loss_grad.mean()
        w_grad = loss_grad.matmul(features.t()).div(loss_grad.size(1))

        # Update
        w -= w_grad * lr
        b -= b_grad * lr

        if print_time:
            iter_time = time.time() - end
            pt_time.add(iter_time)
            logging.info("    Time %.6f (%.6f)" % (iter_time, pt_time.value()))
            end = time.time()

    return w, b



def evaluate_linear_svm(features, labels, w, b):
    """Compute accuracy on a test set"""
    predictions = w.matmul(features).add(b).sign()
    correct = predictions.mul(labels)
    accuracy = correct.add(1).div(2).mean().get_plain_text()
    if crypten.communicator.get().get_rank() == 0:
        print("Test accuracy %.2f%%" % (accuracy.item() * 100))


def run_mpc_stitch(
    label_provider=0, ptype = crypten.mpc.arithmetic
):
    if ptype == crypten.mpc.binary:
        raise NotImplementedError("Binarytensor is not yet supported")
    crypten.init()
   
    world_size = comm.get().get_world_size()
    rank = comm.get().get_rank()
    usr_group = dist.new_group(range(world_size))

    # Set random seed for reproducibility
    # PS: not secure to use time as seed in pratice
    torch.manual_seed(time.time())

    # * User Party
    # Load the data set
    _dataset = dataset_load_linear_svm(dataset_path = "dataset", party=rank, with_label= (rank==label_provider))
    _intersect = np.load("./dataset/IdList_intersection.npy", allow_pickle=True).tolist()
    # print(_intersect)
    _indices = torch.tensor([ _dataset.ids.tolist().index(item) for item in _intersect])
    # print(_indices)
    _dataset.features = torch.index_select(_dataset.features,1,_indices)
    if rank == label_provider:
        _dataset.labels = torch.index_select(_dataset.labels,1,_indices)

    start = time.time()
    
    # Gather the feature's size of each party
    gather_list_feature_size = [torch.zeros(1).type(torch.int64) for _ in range(world_size)]
    dist.all_gather(tensor_list = gather_list_feature_size,tensor = torch.tensor([_dataset.features.size(0)]).type(torch.int64),group = usr_group)
    # print('===========Rank {} scatter feature size=============\n{}\n'.format(rank,gather_list_feature_size))

    
    # Each party send the random sharing (seeds)  to others 
    # PS:there should be 'all_to_all', but the backend 'gloo' do not support 
    # Scatter ramdom share (pesurandom generator's seed) 
    # ! the seed should be random
    # scatter_list_seeds = [(torch.arange(_dataset.features.size(1))*1.0 * (rank + 1) if i != rank else torch.zeros(_dataset.features.size(1)) ) for i in range(world_size-1)]
    scatter_list_seeds = [(torch.tensor([random.getrandbits(63) for _ in range(len(_intersect))]).type(torch.int64) if i != rank else torch.zeros(_dataset.features.size(1)).type(torch.int64) ) for i in range(world_size)]
    # print('===========Rank {} scatter=============\n{}\n'.format(rank,scatter_list_seeds))

    # Store the ramdom share from other parties
    gather_list_seeds = [torch.zeros(len(_intersect)).type(torch.int64) for i in range(world_size)]
    # print('===========Rank {} to collect seeds=============\n{}\n'.format(rank,gather_list_seeds))

    # Send and receive the seed  
    for i in range(world_size):
        dist.scatter(gather_list_seeds[i], scatter_list_seeds if i == rank else [], src = i ,group = usr_group,async_op=True)
    dist.barrier(group = usr_group)

    # Each party send the random sharing (data - PRG(seeds)) to TTP
    # encode the input tensor:
    encoder = FixedPointEncoder()
    _features_share = encoder.encode(_dataset.features)
    _labels_share = encoder.encode(_dataset.labels) if rank == label_provider else None
    # print(_features_share)
    # print(_labels_share)
    
    # Compute the random sharing (data - PRG(seeds))
    
    for i in range(_features_share.size(1)):
        for j in range(len(scatter_list_seeds)):
            if rank == j: 
                continue
            generator = torch.Generator()
            generator.manual_seed(scatter_list_seeds[j][i].int().item())
            # _share = generate_random_ring_element(size=(3,1), generator=generator)
            # print('===========Rank {} example{} party {}_share=============\n{}\n'.format(rank,i,j,_share))
            _share = generate_random_ring_element(_features_share[::,i].size(), generator=generator)
            _features_share[::,i] -= _share
            # print('===========Rank {} example{} party {}_share=============\n{}\n'.format(rank,i,j,_share))
            if rank == label_provider:
                _share = generate_random_ring_element(_labels_share[::,i].size(), generator=generator)
                _labels_share[::,i] -= _share
                # print('===========Rank {} example{} party {}_share (label)=============\n{}\n'.format(rank,i,j,_share))


   

    stitch_features_share = torch.zeros(0,(len(_intersect))).type(torch.int64)
    stitch_labels_share = torch.zeros(0,(len(_intersect))).type(torch.int64)
    
    # print(gather_list_feature_size)
    # gather_list_seeds_TTP
    # ! !!!!!
    # if rank !=0: return
    for k in range(world_size):
        # if k !=1: continue
        if k == rank:
            # print("rank{}:{}--{}".format(rank,stitch_features_share.shape,_features_share_TTP.shape))
            stitch_features_share=torch.cat((stitch_features_share,_features_share),dim=0)
            if k == label_provider:
                stitch_labels_share=torch.cat((stitch_labels_share,_labels_share),dim=0)
                # print("---------\nRank{}'s party{}'s _share:{}\n----------------".format(rank,k,stitch_labels_share))
            continue
        _examples_features_shares = torch.zeros(gather_list_feature_size[k].item(),0).type(torch.int64)
        label_size =  1 if k == label_provider else 0
        _examples_labels_shares = torch.zeros(label_size,0).type(torch.int64)
        for i in range(len(_intersect)):
            g1 = torch.Generator()
            g1.manual_seed(gather_list_seeds[k][i].int().item())
            _share = generate_random_ring_element((gather_list_feature_size[k].item(),1), generator=g1)
            _examples_features_shares=torch.cat((_examples_features_shares,_share),dim=1)

            if k == label_provider:
                _share = generate_random_ring_element((label_size,1), generator=g1)
                _examples_labels_shares=torch.cat((_examples_labels_shares,_share),dim=1)
            else:
                _examples_labels_shares=torch.cat((_examples_labels_shares,torch.zeros(label_size,1).type(torch.int64)),dim=1)
        stitch_features_share=torch.cat((stitch_features_share,_examples_features_shares),dim=0)

        stitch_labels_share=torch.cat((stitch_labels_share,_examples_labels_shares),dim=0)
    
    end_time = time.time()

    print("--rank:{}--all:{}--".format(rank,end_time-start))

    dist.reduce(stitch_features_share,dst=0,group=usr_group)
    if rank ==0:
        # print('===========Rank {} stitch_features=============\n{}\n'.format(rank,stitch_features_share)) 
        print('===========Rank {} stitch_features=============\n{}\n'.format(rank,stitch_features_share.shape)) 
    dist.reduce(stitch_labels_share,dst=0,group=usr_group)
    if rank ==0:
        # print('===========Rank {} stitch_features=============\n{}\n'.format(rank,stitch_features_share)) 
        print('===========Rank {} stitch_labels=============\n{}\n'.format(rank,stitch_labels_share.shape)) 


    dist.barrier(group = usr_group)
    if rank==1:
        import csv
        with open('./dataset/stat.csv'.format(i), 'a', newline='') as csvfile:
            writer  = csv.writer(csvfile)
            writer.writerow(['stitch',world_size-1,_dataset.features.size(1),stitch_features_share.size(0),stitch_features_share.size(1),0,0,0,end_time-start])
    return