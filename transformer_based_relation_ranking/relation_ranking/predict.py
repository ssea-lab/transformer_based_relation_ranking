import os, sys
import numpy as np
import torch
import pickle
from args import get_args
from model import RelationRanking
from relationRankingLoader import RelationRankingLoader


args = get_args()
torch.manual_seed(args.seed)
if not args.cuda:
    args.gpu = -1
if torch.cuda.is_available() and args.cuda:
    print("Note: You are using GPU for predicting")
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(args.seed)
if torch.cuda.is_available() and not args.cuda:
    print("Warning: You have Cuda but do not use it. You are using CPU for predicting")
if not args.trained_weight:
    print("ERROR: You need to provide a option 'trained_weight' path to load the model weight.")
    sys.exit(1)
# define the model and load weights
model = RelationRanking(args)
model.load_state_dict(torch.load(args.trained_weight))
for parameter in model.parameters():
    parameter.requires_grad = False
if args.cuda:
    model.cuda()
    print("move model to GPU")

def predict(data_file):
        # load batch data for predict
    testing_loader = RelationRankingLoader(data_file, args)
    print('load testing data, batch_num: %d\tbatch_size: %d'
    %(testing_loader.batch_num, testing_loader.batch_size))
    # model.eval()
    n_total = 0
    n_correct = 0
    for data_batch in testing_loader.next_batch(False):
        pos_scores, neg_scores = model(data_batch)
        n_correct += (torch.sum(torch.gt(pos_scores, neg_scores), 1).data == neg_scores.size(1)).sum()
        n_total += pos_scores.size(0)
    accuracy =  float(n_correct) / float(n_total)
    print("accuracy: %8.6f\tcorrect: %d\ttotal: %d" %(accuracy, n_correct, n_total))
    
predict(args.test_file)
