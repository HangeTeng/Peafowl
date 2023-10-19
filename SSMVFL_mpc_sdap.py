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

# TODO: CLK code 
def CLK(ids):
    return ids

def intersect_shuffle(_lists):
    _intersect = list(set(_lists[0]).intersection(*_lists[1:]))
    random.seed(time.time())
    random.shuffle(_intersect)
    # print(_intersect)
    _shuffles = [ [ _list.index(item) for item in _intersect] for _list in _lists]
    # print(_shuffles)
    return _shuffles

def run_mpc_sdap(
    epochs=50, examples=50, features=100, lr=0.5, skip_plaintext=False, label_provider=0, ptype = crypten.mpc.arithmetic
):
    if ptype == crypten.mpc.binary:
        raise NotImplementedError("Binarytensor is not yet supported")
    crypten.init()
   
    world_size = comm.get().get_world_size()
    rank = comm.get().get_rank()
    usr_group = dist.new_group(range(world_size-1))
    all_group = dist.new_group(range(world_size))

    # Set random seed for reproducibility
    # PS: not secure to use time as seed in pratice
    torch.manual_seed(time.time())

    if rank != world_size-1:
        # * User Party
        # Load the data set
        _dataset = dataset_load_linear_svm(dataset_path = "dataset", party=rank, with_label= (rank==label_provider))

        # Gather the dataset's size of each party 
        gather_list_dataset_size = [torch.zeros(1).type(torch.int64) for _ in range(world_size-1)]
        dist.all_gather(tensor_list = gather_list_dataset_size,tensor = torch.tensor([_dataset.features.size(1)]).type(torch.int64),group = usr_group)
        # print('===========Rank {} scatter dataset size=============\n{}\n'.format(rank,gather_list_dataset_size))
        
        # Gather the feature's size of each party
        gather_list_feature_size = [torch.zeros(1).type(torch.int64) for _ in range(world_size-1)]
        dist.all_gather(tensor_list = gather_list_feature_size,tensor = torch.tensor([_dataset.features.size(0)]).type(torch.int64),group = usr_group)
        # print('===========Rank {} scatter feature size=============\n{}\n'.format(rank,gather_list_feature_size))

        
        # Each party send the random sharing (seeds)  to others 
        # PS:there should be 'all_to_all', but the backend 'gloo' do not support 
        # Scatter ramdom share (pesurandom generator's seed) 
        # ! the seed should be random
        scatter_list_seeds = [(torch.arange(_dataset.features.size(1))*1.0 * (rank + 1) if i != rank else torch.zeros(_dataset.features.size(1)) ) for i in range(world_size-1)]
        # scatter_list_seeds = [(torch.randn(_dataset.features.size(1)) if i != rank else torch.zeros(_dataset.features.size(1)) ) for i in range(world_size-1)]
        # print('===========Rank {} scatter=============\n{}\n'.format(rank,scatter_list_seeds))

        # Store the ramdom share from other parties
        gather_list_seeds = [torch.zeros(gather_list_dataset_size[i].item()) for i in range(world_size-1)]
        # print('===========Rank {} to collect seeds=============\n{}\n'.format(rank,gather_list_seeds))

        # Send and receive the seed  
        for i in range(world_size-1):
            dist.scatter(gather_list_seeds[i], scatter_list_seeds if i == rank else [], src = i ,group = usr_group,async_op=True)
        dist.barrier(group = usr_group)
        # print('===========Rank {} collected seeds=============\n{}\n'.format(rank,gather_list_seeds))

        # scatter_list_label_seeds = [(torch.ones(_dataset.label.size(0)) * (rank + 1) if i != rank else torch.zeros(_dataset.label.size(1)) ) for i in range(world_size-1)]
        # list_label_seeds = torch.zeros(_dataset.label.size(0))
        # dist.scatter(list_label_seeds, scatter_list_label_seeds if label_provider == rank else [], src = i ,group = usr_group,async_op=True)


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
        # print('===========Rank {}  _share=============\n{}\n'.format(rank,_features_share))
        # print(type(_features_share))
        # print(_labels_share.dtype)

        # Turn ID into CLK code
        _id_share = CLK(_dataset.ids)
        # print(_id_share)
        
        # _dataset_share = dataset_svm(ids=_id_share,features=_features_share,labels=_labels_share)

        # With TTP:
        # Send dataset size to TTP
        dist.gather(tensor = torch.tensor([_dataset.features.size(1)]).type(torch.int64),dst=world_size-1,group=all_group,async_op=True)
        # Send feature size to TTP
        dist.gather(tensor = torch.tensor([_dataset.features.size(0)]).type(torch.int64),dst=world_size-1,group=all_group,async_op=True)

        dist.barrier(group = all_group)

        
        # Send ids to TTP
        dist.gather_object(obj=_id_share,dst=world_size-1,group=all_group)

        # Send features to TTP
        dist.gather_object(obj = _features_share, dst=world_size-1, group=all_group)
        
        # Send labels to TTP
        if rank == label_provider:
            dist.send(tensor = _labels_share, dst=world_size-1, group=all_group)
        # ! why can't i use the isend and irecv 
    
        dist.barrier(group = all_group)

        # Receive shuffles
        _intersection_size = torch.zeros(1).type(torch.int64)
        # print('===========Rank {} to get _intersection_size=============\n{}\n'.format(rank,_intersection_size))
        dist.broadcast(tensor = _intersection_size,src=world_size -1,group=all_group)
        # dist.scatter(_intersection_size, src = world_size-1 ,group = all_group)
        # dist.irecv(tensor = _intersection_size, src = world_size-1, group=all_group)
        # dist.barrier(group = all_group)
        # print('===========Rank {} _intersection_size=============\n{}\n'.format(rank,_intersection_size.item()))

        _shuffles = torch.zeros(_intersection_size.item(),world_size-1).type(torch.int64)
        # print('===========Rank {} to get _shuffles=============\n{}\n'.format(rank,_shuffles))
        dist.scatter(_shuffles, src = world_size-1 ,group = all_group)
        # print('===========Rank {} _shuffles=============\n{}\n'.format(rank,_shuffles))

        # shuffle seed
        # print('===========Rank {} shuffle seeds=============\n{}\n'.format(rank,gather_list_seeds))
        for i in range(world_size-1):
            if i != rank:
                # print("{}--{}".format(gather_list_seeds[i],_shuffles[i]))
                gather_list_seeds[i] = torch.index_select(gather_list_seeds[i],0,_shuffles[:,i])
        # print('===========Rank {} shuffle seeds=============\n{}\n'.format(rank,gather_list_seeds))


        gather_list_seeds_TTP = [torch.zeros(_intersection_size.item()) for _ in range(world_size-1)]
        # print('===========Rank {} to collect seeds TTP=============\n{}\n'.format(rank,gather_list_seeds_TTP))

        # Send and receive the seed  
        for i in range(world_size-1):
            dist.scatter(gather_list_seeds_TTP[i], src = world_size-1 ,group = all_group)
                        #  , async_op=True)
        # print('===========Rank {} collect seeds TTP=============\n{}\n'.format(rank,gather_list_seeds_TTP))

        _features_share_TTP = torch.zeros(_features_share.size(0),(_intersection_size.item())).type(torch.int64)
        # print(_features_share_TTP)
        _labels_share_TTP = torch.zeros(1,(_intersection_size.item())).type(torch.int64)
        dist.recv(tensor = _features_share_TTP, src = world_size-1, group=all_group)
        if rank == label_provider:
            dist.recv(tensor = _labels_share_TTP, src = world_size-1, group=all_group)
            # print('===========Rank {} collect _labels_share_TTP=============\n{}\n'.format(rank,_labels_share_TTP))
        # print('===========Rank {} collect _features_share_TTP=============\n{}\n'.format(rank,_features_share_TTP))

        sdap_features_share = torch.zeros(0,(_intersection_size.item())).type(torch.int64)
        sdap_labels_share = torch.zeros(0,(_intersection_size.item())).type(torch.int64)
        
        # print(gather_list_feature_size)
        # gather_list_seeds_TTP
        # ! !!!!!
        # if rank !=0: return 
        for k in range(world_size-1):
            if k == rank:
                # print("rank{}:{}--{}".format(rank,sdap_features_share.shape,_features_share_TTP.shape))
                sdap_features_share=torch.cat((sdap_features_share,_features_share_TTP),dim=0)
                if k == label_provider:
                    sdap_labels_share=torch.cat((sdap_labels_share,_labels_share_TTP),dim=0)
                continue
            _examples_features_shares = torch.zeros(gather_list_feature_size[k].item(),0).type(torch.int64)
            label_size =  1 if k == label_provider else 0
            _examples_labels_shares = torch.zeros(label_size,0).type(torch.int64)
            for i in range(_intersection_size.item()):
                generator = torch.Generator()
                g2 = generator.manual_seed(gather_list_seeds[k][i].int().item())
                _share = generate_random_ring_element((gather_list_feature_size[k].item(),1), generator=g2)
                # print("{}--{}".format(_examples_features_shares.shape,_share.shape))
                _examples_features_shares=torch.cat((_examples_features_shares,_share),dim=1)

                if k == label_provider:
                    _share = generate_random_ring_element((label_size,1), generator=g2)
                    _examples_labels_shares=torch.cat((_examples_labels_shares,_share),dim=1)
                else:
                    _examples_labels_shares=torch.cat((_examples_labels_shares,torch.zeros(label_size,1).type(torch.int64)),dim=1)
            # print("rank{}:{}--{}".format(rank,sdap_features_share.shape,_examples_features_shares.shape))
            sdap_features_share=torch.cat((sdap_features_share,_examples_features_shares),dim=0)
            # print(rank)
            sdap_labels_share=torch.cat((sdap_labels_share,_examples_labels_shares),dim=0)

        # print('===========Rank {} sdap_features_share=============\n{}\n'.format(rank,sdap_features_share)) 
        # print('===========Rank {} sdap_labels_share=============\n{}\n'.format(rank,sdap_labels_share))

        #########################################################################################
        # sdap_features_share = torch.zeros(0,(_intersection_size.item())).type(torch.int64)
        # sdap_labels_share = torch.zeros(0,(_intersection_size.item())).type(torch.int64)
        
        # # print(gather_list_feature_size)
        # # gather_list_seeds_TTP
        # # ! !!!!!
        # # if rank !=0: return 
        # for k in range(world_size-1):
        #     if k == rank:
        #         # print("rank{}:{}--{}".format(rank,sdap_features_share.shape,_features_share_TTP.shape))
        #         sdap_features_share=torch.cat((sdap_features_share,_features_share_TTP),dim=0)
        #         if k == label_provider:
        #             sdap_labels_share=torch.cat((sdap_labels_share,_labels_share_TTP),dim=0)
        #         continue
        #     _examples_features_shares = torch.zeros(gather_list_feature_size[k].item(),0).type(torch.int64)
        #     label_size =  1 if k == label_provider else 0
        #     _examples_labels_shares = torch.zeros(label_size,0).type(torch.int64)
        #     for i in range(_intersection_size.item()):
        #         generator = torch.Generator()
        #         g1 = generator.manual_seed(gather_list_seeds_TTP[k][i].int().item())
        #         g2 = generator.manual_seed(gather_list_seeds[k][i].int().item())
        #         _share = generate_random_ring_element((gather_list_feature_size[k].item(),1), generator=g1) + generate_random_ring_element((gather_list_feature_size[k].item(),1), generator=g2)
        #         # print("{}--{}".format(_examples_features_shares.shape,_share.shape))
        #         _examples_features_shares=torch.cat((_examples_features_shares,_share),dim=1)

        #         if k == label_provider:
        #             _share = generate_random_ring_element((label_size,1), generator=g1) + generate_random_ring_element((label_size,1), generator=g2)
        #             _examples_labels_shares=torch.cat((_examples_labels_shares,_share),dim=1)
        #         else:
        #             _examples_labels_shares=torch.cat((_examples_labels_shares,torch.zeros(label_size,1).type(torch.int64)),dim=1)
        #     # print("rank{}:{}--{}".format(rank,sdap_features_share.shape,_examples_features_shares.shape))
        #     sdap_features_share=torch.cat((sdap_features_share,_examples_features_shares),dim=0)
        #     # print(rank)
        #     sdap_labels_share=torch.cat((sdap_labels_share,_examples_labels_shares),dim=0)

        # # print('===========Rank {} sdap_features_share=============\n{}\n'.format(rank,sdap_features_share)) 
        # # print('===========Rank {} sdap_labels_share=============\n{}\n'.format(rank,sdap_labels_share)) 
        #########################################################################################
        

        dist.reduce(sdap_features_share,dst=0,group=usr_group)
        if rank ==0:
            print('===========Rank {} sdap_features=============\n{}\n'.format(rank,sdap_features_share)) 


        dist.barrier(group = all_group)
        return
                
    else:
        # * The Third Party
        # Receive dataset size and feature size
        gather_list_dataset_size = [torch.zeros(1).type(torch.int64) for _ in range(world_size)]
        dist.gather(tensor = torch.zeros(1).type(torch.int64),gather_list=gather_list_dataset_size, dst=world_size-1,group=all_group,async_op=True)
        # print('===========Rank {} gather dataset size==/===========\n{}\n'.format(rank,gather_list_dataset_size))

        gather_list_feature_size = [torch.zeros(1).type(torch.int64) for _ in range(world_size)]
        dist.gather(tensor = torch.zeros(1).type(torch.int64),gather_list=gather_list_feature_size, dst=world_size-1,group=all_group,async_op=True)
        # print('===========Rank {} gather feature size=============\n{}\n'.format(rank,gather_list_feature_size))     

        dist.barrier(group = all_group)

        # Receive ids
        gather_list_ids = [[uuid.uuid4() for _ in range(gather_list_dataset_size[i].item())] for i in range(world_size)]
        # print('===========Rank {} gather ids=============\n{}\n'.format(rank,gather_list_ids)) 
        dist.gather_object(obj=[],object_gather_list=gather_list_ids, dst=world_size-1,group=all_group)
        gather_list_ids.pop()
        gather_list_ids = [ ids.tolist() for ids in gather_list_ids]
        # print('===========Rank {} gather ids=============\n{}\n'.format(rank,gather_list_ids)) 

        # Receive features
        gather_list_features_share = [torch.zeros(gather_list_feature_size[i].item(),gather_list_dataset_size[i].item()).type(torch.int64) for i in range(world_size)]
        # print(type(gather_list_features_share))
        # print('===========Rank {} gather features=============\n{}\n'.format(rank,gather_list_features_share[0].shape)) 
        dist.gather_object(obj = torch.zeros(0).type(torch.int64), object_gather_list= gather_list_features_share, dst=world_size-1, group=all_group)
        # print('===========Rank {} gather features=============\n{}\n'.format(rank,gather_list_features_share)) 

        # Receive labels
        _labels_share = torch.zeros(1,gather_list_dataset_size[label_provider]).type(torch.int64)
        dist.recv(tensor = _labels_share, src=label_provider, group=all_group)
        # print('===========Rank {} gather labels=============\n{}\n'.format(rank,_labels_share)) 

        dist.barrier(group = all_group)

        # * Compute the intersection (after shuffling)
        # The shuffles
        _shuffles = intersect_shuffle(gather_list_ids)
        # print(_shuffles)
        _shuffles = torch.tensor(_shuffles)
        for i in range(world_size-1):
            gather_list_features_share[i] = torch.index_select(gather_list_features_share[i],1,_shuffles[i])
            if i == label_provider:
                _labels_share = torch.index_select(_labels_share,1,_shuffles[i])
        # print('===========Rank {} shuffle feature=============\n{}\n'.format(rank,gather_list_features_share))  
        # print('===========Rank {} shuffle label=============\n{}\n'.format(rank,_labels_share))  

        scatter_list_shuffles = [copy.deepcopy(_shuffles) for _ in range(world_size)]  
        # print(scatter_list_shuffle)
        for i in range(world_size-1):
            scatter_list_shuffles[i][i] = 0
        # print(scatter_list_shuffles)

        # Send shuffles to usr
        # Send the shuffle size
       
        _intersection_size=torch.tensor([_shuffles.size(1)]).type(torch.int64)
        # print('===========Rank {} to get _intersection_size=============\n{}\n'.format(rank,_intersection_size.item()))
        dist.broadcast(tensor = _intersection_size,src=world_size-1,group=all_group)
        # scatter_list_intersection_size = [torch.tensor([_shuffles.size(1)]).type(torch.int64)]*(world_size)
        # dist.scatter(_intersection_size, scatter_list_intersection_size, src = world_size-1 ,group = all_group)
        # for i in range(world_size-1):
        #     dist.isend(tensor = _intersection_size, dst=i, group=all_group)
        # dist.barrier(group = all_group)
        # print('===========Rank {} _intersection_size=============\n{}\n'.format(rank,_intersection_size.item))
        
         
        dist.scatter(_shuffles, scatter_list_shuffles, src = world_size-1 ,group = all_group)

        # the zero sharing
        # ! random!!!
        scatter_list_zerosharing_seeds = [[(torch.arange(_intersection_size.item())*1.0 * (j  + 1) * (i + j + 1) if (j != i and j != world_size-1) else torch.zeros(_intersection_size.item()) ) for j in range(world_size)] for i in range(world_size-1)]
        # print(scatter_list_zerosharing_seeds)
        
        for i in range(world_size-1):
            dist.scatter(scatter_list_zerosharing_seeds[i][rank], scatter_list_zerosharing_seeds[i], src = world_size-1 ,group = all_group)
        
        # Compute the random sharing (share - PRG(zerosharing seeds))
        for k in range(world_size-1):
            for i in range(gather_list_features_share[k].size(1)):
                for j in range(len(scatter_list_zerosharing_seeds[k])):
                    if k == j: 
                        continue
                    generator = torch.Generator()
                    generator.manual_seed(scatter_list_zerosharing_seeds[k][j][i].int().item())
                    # _share = generate_random_ring_element(size=(3,1), generator=generator)
                    # print('===========Rank {} example{} party {}_share=============\n{}\n'.format(rank,i,j,_share))
                    _share = generate_random_ring_element(gather_list_features_share[k][::,i].size(), generator=generator)
                    gather_list_features_share[k][::,i] -= 0 # ! _share
                    print('===========Rank {} example{} party {}_share=============\n{}\n'.format(rank,i,j,_share))
                    if rank == label_provider:
                        _share = generate_random_ring_element(_labels_share[::,i].size(), generator=generator)
                        _labels_share[::,i] -= 0 # !_share
                        # print('===========Rank {} example{} party {}_share (label)=============\n{}\n'.format(rank,i,j,_share))
        # print('===========Rank {} shuffle feature=============\n{}\n'.format(rank,gather_list_features_share)) 

        for i in range(world_size-1):
            dist.send(tensor = gather_list_features_share[i], dst = i, group=all_group)
            if i == label_provider:
                dist.send(tensor = _labels_share[i], dst = i, group=all_group)
        dist.barrier(group = all_group)



        return 










        
        


        


            







    

    # # Initialize x, y, w, b
    # x = torch.randn(features, examples)
    # w_true = torch.randn(1, features)
    # b_true = torch.randn(1)
    # y = w_true.matmul(x) + b_true
    # y = y.sign()
    # print(x)
    # print(y)
    

    # if not skip_plaintext:
    #     logging.info("==================")
    #     logging.info("PyTorch Training")
    #     logging.info("==================")
    #     w_torch, b_torch = train_linear_svm(x, y, lr=lr, print_time=True)

    # # Encrypt features / labels
    # x = crypten.cryptensor(x)
    # print(x)
    # y = crypten.cryptensor(y)
    # print(y)

    # logging.info("==================")
    # logging.info("CrypTen Training")
    # logging.info("==================")
    # w, b = train_linear_svm(x, y, lr=lr, print_time=True)

    # if not skip_plaintext:
    #     logging.info("PyTorch Weights  :")
    #     logging.info(w_torch)
    # logging.info("CrypTen Weights:")
    # logging.info(w.get_plain_text())

    # if not skip_plaintext:
    #     logging.info("PyTorch Bias  :")
    #     logging.info(b_torch)
    # logging.info("CrypTen Bias:")
    # logging.info(b.get_plain_text())
