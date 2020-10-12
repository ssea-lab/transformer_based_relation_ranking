#!/usr/bin/env python
#-*- coding: utf-8 -*-
import torch
import torch.optim as optim
import torch.nn as nn
import time
import os, sys, glob
import numpy as np
from args import get_args
from model import RelationRanking
from relationRankingLoader import RelationRankingLoader

# please set the configuration in the file : args.py
args = get_args()
# set the random seed for reproducibility
torch.manual_seed(args.seed)

if not args.cuda:
    args.gpu = -1
if torch.cuda.is_available() and args.cuda:
    print("Note: You are using GPU " + str(args.gpu)+ " for training")
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(args.seed)
if torch.cuda.is_available() and not args.cuda:
    print("Warning: You have Cuda but do not use it. You are using CPU for training")
       
# 记录参数的配置
config = args
# with open(os.path.join(config.save_path, 'parameters.txt'), 'w') as f:
#     f.write(str(config))
os.makedirs(config.save_path, exist_ok=True)

# load training data
training_loader = RelationRankingLoader(args.train_file, args)
print('load train data, batch_num: %d\tbatch_size: %d'
      %(training_loader.batch_num, training_loader.batch_size))

# define the model
model = RelationRanking(config)
if config.resume_trained_weight:
    model.load_state_dict(torch.load(config.resume_trained_weight))
if args.cuda:
    model.cuda()
    print("move model to GPU")
#将预训练模型参数设为不可训练
for parameter in model.pretrained_model.parameters():
    parameter.requires_grad = False
# show model parameters
def get_parameter_info(net):
    for name, param in net.named_parameters():
        print(name, param.size())
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}
model_parameter_info = get_parameter_info(model)
print(model_parameter_info)

criterion = nn.MarginRankingLoss(args.loss_margin) # Max margin ranking loss function
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

# train the model
iterations = 0
start = time.time()
header = '  Time Epoch Iteration Progress    (%Epoch)   Loss       Accuracy  Dev/Accuracy'
log_template =     'train: '+' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:12.4f},{}'.split(','))
print(header)

for epoch in range(1, args.epochs+1):
    n_correct, n_total = 0, 0
    for batch_idx, batch in enumerate(training_loader.next_batch()):
        iterations += 1
        model.train()
        optimizer.zero_grad()
        pos_scores, neg_scores = model(batch)
        n_correct += (torch.sum(torch.gt(pos_scores, neg_scores), 1).data == neg_scores.size(1)).sum()
        n_total += pos_scores.size(0)
        train_acc =  float(n_correct) / float(n_total)
        ones = torch.autograd.Variable(torch.ones(pos_scores.size(0)*pos_scores.size(1)))
        if args.cuda:
            ones = ones.cuda(args.gpu)
        loss = criterion(pos_scores.contiguous().view(-1,1).squeeze(1), neg_scores.contiguous().view(-1,1).squeeze(1), ones)
        loss.backward()
        optimizer.step()

        # checkpoint model periodically
        if iterations % args.save_every == 0:
            model_weight_prefix = os.path.join(args.save_path, 'model_weight')
            model_weight_path = model_weight_prefix + \
                        '_iter_{}_acc_{:.4f}_loss_{:.6f}_modelweight.pt'.format(iterations, train_acc, loss.item())
            torch.save(model.state_dict(), model_weight_path)
            for f in glob.glob(model_weight_prefix + '*'):
                if f != model_weight_path:
                    os.remove(f)

        # print progress message
        elif iterations % args.log_every == 0:
            print(log_template.format(time.time()-start, epoch, iterations, 1+batch_idx, 
                                      training_loader.batch_num, 100. * (1+batch_idx)/training_loader.batch_num, 
                                      loss.item(), train_acc, ' '*12))

