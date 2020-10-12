from transformers import BertTokenizer, BertModel, PretrainedConfig
import torch
# # import torch.nn.functional as F
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
a = tokenizer(['hello world'])
print(a)
# model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
# model_config = PretrainedConfig()
# config_dict = model_config.get_config_dict('bert-base-uncased')
# print(config_dict)
# print(config_dict[0]['num_hidden_layers'])
# batch_sentences = ["Hello I'm a single sentence",
#                    "And another sentence",
#                    "And the very very last one"]
# batch = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
# # batch.to
# output = model(**batch)
# print(output.last_hidden_state)
# # print(batch['input_ids'])
# # print(batch)
#batch_size:2, questions_len:3, hidden_size:4, neg_samples:5
# a = torch.Tensor([
#                  [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]],
#                  [[4, 4, 4, 4], [5, 5, 5, 5], [6, 6, 6, 6]]
#                  ])
# b = torch.nn.parameter.Parameter(F.softmax(torch.randn(13)))
# print(a*b[0] )
# b = torch.unsqueeze(a, dim=1).expand(2, 5, 3, 4).squeeze(0)
# # c = torch.
# # print(a[:, :1])
# print(b)
# print(b.size())
# assert (1, 2) == (2, 2), 'sdsds'
# a = F.softmax(torch.nn.parameter.Parameter(torch.randn(13)))

# print(a)
# print(F.softmax(torch.randn(13)))
# print( torch.nn.parameter.Parameter(F.softmax(torch.randn(13))))
# print(torch.ones(13))
# import os
# with open(os.path.join('saved_checkpoints', 'parameters.txt'), 'w') as f:
#     f.write('----'*100)
#     f.write('----')
#     print('11')
# with open('/data/wangqingbin/kbka2/transformer_based_relation_ranking/relation_ranking/saved_checkpoints/param.txt', 'w') as f:
#     f.write('----'*100)
#     f.write('----')
#     print('11')
# a= torch.Tensor([[1, 2, 3],[4, 5, 6]])
# a = a.contiguous().view(-1, 1)
# a = a.contiguous().view(2, 3)
# print(a)