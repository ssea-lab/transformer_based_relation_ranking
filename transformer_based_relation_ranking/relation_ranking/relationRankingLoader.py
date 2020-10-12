#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pickle
import random
import re
import torch
from args import get_args
from tokenizerAndModel import get_tokenizer
args = get_args()

class RelationRankingLoader():
    def __init__(self, data_file, config):
        self.data_list = pickle.load(open(data_file, 'rb'))
        self.config = config
        self.batch_size = self.config.batch_size
        self.tokenizer = get_tokenizer(config.pretrained_weight)

        self.slice_index = [i for i in range(
            0, len(self.data_list), self.batch_size)]
        self.batch_num = len(self.slice_index)

    def get_separated_rel(self, relation):
        rel = ' '.join(re.split("/|_", relation)).strip()
        return rel

    def next_batch(self, shuffle=True):

        if shuffle:
            slice_index_copy = self.slice_index[:]
            random.shuffle(slice_index_copy)
            indices = slice_index_copy
        else:
            indices = self.slice_index

        for i in indices:
            data_slice = self.data_list[i:(i+self.batch_size)]
            questions = []
            pos_rels = []
            neg_rels = []

            for data in data_slice:
                question = data[0]
                relation = self.get_separated_rel(data[1])
                can_rels = [self.get_separated_rel(i) for i in data[2]]
                questions.append(question)
                pos_rels.append(relation)
                neg_rels.extend(can_rels)
            

            # questions: list of string, length: batch_size
            # pos_rels:  list of string, length: batch_size
            # neg_rels:  list of string, length: batch_size*neg_samples
            tokenized_questions = self.tokenizer(questions, padding=True, truncation=True, return_tensors="pt")
            tokenized_pos_rels = self.tokenizer(pos_rels, padding=True, truncation=True, return_tensors="pt")
            tokenized_neg_rels = self.tokenizer(neg_rels, padding=True, truncation=True, return_tensors="pt")
            device = torch.device('cuda:'+str(self.config.gpu))
            #将AlbertTokenizer分词得到的结果移动到指定gpu上
            tokenized_questions, tokenized_pos_rels, tokenized_neg_rels = tokenized_questions.to(device), tokenized_pos_rels.to(device), tokenized_neg_rels.to(device)
            
            yield tokenized_questions, tokenized_pos_rels, tokenized_neg_rels
