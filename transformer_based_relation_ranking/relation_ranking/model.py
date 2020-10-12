#!/usr/bin/env python
#-*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch import nn
from tokenizerAndModel import get_model
from transformers import PretrainedConfig
class RelationRanking(nn.Module):
    def __init__(self, config):
        super(RelationRanking, self).__init__()
        self.config = config
        self.pretrained_model = get_model(config.pretrained_weight)
        pretrained_model_config = PretrainedConfig()
        num_layers = pretrained_model_config.get_config_dict(config.pretrained_weight)[0]['num_hidden_layers']
        self.num_embedding_layers = num_layers + 1
        self.layer_weights = nn.parameter.Parameter(torch.ones(self.num_embedding_layers))


        self.questions_max_layer = nn.Sequential(
            nn.Linear(self.config.questions_maxlen, self.config.questions_maxlen//2),
            nn.Linear(self.config.questions_maxlen//2, 1)
        )
        self.rels_max_layer = nn.Sequential(
            nn.Linear(self.config.rels_maxlen, self.config.rels_maxlen//2),
            nn.Linear(self.config.rels_maxlen//2, 1)
        )
        # 此线性层是用来衡量双向relation权重的
        self.que_rel_weight = nn.Linear(2, 1)  
     
    def get_similarity_matrix(self,questions_tensor, rels_tensor):
        # the shape of questions_tensor is (batch_size, question_len, hidden_size)
        # the shape of rels_tensor is (batch_size, rels_len, hidden_size)
        questions_norm = torch.norm(questions_tensor, 2, dim=2, keepdim=True)
        questions_norm = questions_norm.expand_as(questions_tensor)
        questions_tensor = questions_tensor.div(questions_norm)
        rels_norm = torch.norm(rels_tensor, 2, dim=2, keepdim=True)
        rels_norm = rels_norm.expand_as(rels_tensor)
        rels_tensor = rels_tensor.div(rels_norm)
        sim_matrix = torch.bmm(questions_tensor, rels_tensor.transpose(1, 2))
        return sim_matrix

    def get_max_pooling(self, sim_matrix):
        ''' this function is used to calculate max_pooling for question and relation similarity matrix
        '''
        questions_max, _ = torch.max(sim_matrix, 2)  # the shape is [batch_size, questions_len]
        rels_max, _ = torch.max(sim_matrix, 1)  # the shape is [batch_size, rels_len]
        return questions_max, rels_max
    
    def weighted_sum_all_layers(self, tuple_of_hidden_states):
        assert len(tuple_of_hidden_states) == self.num_embedding_layers, 'number of layers does not match!'
        final_states = torch.zeros_like(tuple_of_hidden_states[0])
        for i in range(self.num_embedding_layers):
            final_states = final_states + tuple_of_hidden_states[i]*F.softmax(self.layer_weights)[i]
        return final_states

    def cal_score(self, questions, rels):
        #shape of questions :[batch_size, question_length, hidden_size]
        #shape of pos_rels :[batch_size, pos_rel_length, hidden_size]
        #shape of neg_rels :[batch_size*neg_samples, neg_rel_length, hidden_size]
        #对于正的关系对来讲，batch_size为原始的配置size，对于负的关系对来讲，batch_size为batch_size*neg_samples
        batch_size, questions_length, hidden_size = questions.size()
        _, rels_len, _ = rels.size()  
        #得到question和relations的相似度矩阵 shape of sim_matrix:[batch_size, questions_len, rels_len]
        sim_matrix = self.get_similarity_matrix(questions, rels)
        questions_max, rels_max = self.get_max_pooling(sim_matrix)

        #question和relations的长度与后面线性层的权重参数可能不一致，需要用0补齐或切片
        if questions_length < self.config.questions_maxlen:
            zero_padding = torch.zeros(batch_size, self.config.questions_maxlen - questions_length)
            zero_padding = zero_padding.cuda(self.config.gpu)
            questions_max = torch.cat((questions_max,zero_padding), 1)
        else:
            questions_max = questions_max[:, :self.config.questions_maxlen]

        if rels_len < self.config.rels_maxlen:
            zero_padding = torch.zeros(batch_size, self.config.rels_maxlen - rels_len)
            zero_padding = zero_padding.cuda(self.config.gpu)
            rels_max = torch.cat((rels_max,zero_padding), 1)
        else:
            rels_max = rels_max[:, :self.config.rels_maxlen]

        questions_max_score = self.questions_max_layer(questions_max)  # the shape of the questions_max_score is (batch_size, 1)
        rels_max_score = self.rels_max_layer(rels_max)  # the shape of the rels_max_score is (batch_size, 1)
        #计算关系最终的分数
        que_rel_scores = torch.cat((questions_max_score, rels_max_score), 1)
        relation_scores = self.que_rel_weight(que_rel_scores)
        return relation_scores  #shape of positive:[batch_size, 1]  shape of negative:[batch_size*neg_size, 1]

    def forward(self, data_batch):
        #shape of tokenized_questions['input_ids']:[batch_size, question_length]
        #shape of tokenized_pos_rels['input_ids']:[batch_size, pos_rel_length]
        #shape of tokenized_neg_rels['input_ids']:[batch_size*neg_samples, neg_rel_length]
        tokenized_questions, tokenized_pos_rels, tokenized_neg_rels = data_batch

        #将分词得到的数据输入到AlbertModel中
        #shape of bert_questions.last_hidden_state :[batch_size, question_length, hidden_size]
        #shape of bert_pos_rels.last_hidden_state :[batch_size, pos_rel_length, hidden_size]
        #shape of bert_neg_rels.last_hidden_state :[batch_size*neg_samples, neg_rel_length, hidden_size]
        model_questions = self.pretrained_model(**tokenized_questions, output_hidden_states=True)
        model_pos_rels = self.pretrained_model(**tokenized_pos_rels, output_hidden_states=True)
        model_neg_rels = self.pretrained_model(**tokenized_neg_rels, output_hidden_states=True)
        batch_size, questions_length, hidden_size = model_questions.last_hidden_state.size()

        #可能使用last_hidden_state和weighted_sum_layers_states两种情况
        if self.config.weighted_hidden_states:
            questions_hidden_states = self.weighted_sum_all_layers(model_questions.hidden_states)
            pos_rels_hidden_states = self.weighted_sum_all_layers(model_pos_rels.hidden_states)
            neg_rels_hidden_states = self.weighted_sum_all_layers(model_neg_rels.hidden_states)
        else:
            questions_hidden_states = model_questions.last_hidden_state
            pos_rels_hidden_states = model_pos_rels.last_hidden_state
            neg_rels_hidden_states = model_neg_rels.last_hidden_state

        pos_scores = self.cal_score(questions_hidden_states, pos_rels_hidden_states)
        pos_scores = pos_scores.expand(batch_size, self.config.neg_size)

        #处理负样本时需要将questions进行扩展
        neg_questions_hidden_state = torch.unsqueeze(questions_hidden_states, dim=1).expand(batch_size, self.config.neg_size, questions_length, hidden_size).contiguous().view(batch_size*self.config.neg_size, questions_length, hidden_size)
        neg_scores = self.cal_score(neg_questions_hidden_state, neg_rels_hidden_states)
        neg_scores = neg_scores.squeeze(1).contiguous().view(batch_size, self.config.neg_size)
        # shape of pos_scores, neg_scores: [batch_size, self.config.neg_size]

        return pos_scores, neg_scores

