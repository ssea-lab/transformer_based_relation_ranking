import sys
from transformers import *
def get_tokenizer(pretrained_weight):
    #判断是哪个模型的预训练分词器权重
    weight_tokenizer = pretrained_weight.split('-')[0]
    if weight_tokenizer == 'bert':
        tokenizer = BertTokenizer.from_pretrained(pretrained_weight)
        return tokenizer
    elif weight_tokenizer == 'xlnet':
        tokenizer = XLNetTokenizer.from_pretrained(pretrained_weight)
        return tokenizer
    elif weight_tokenizer == 'albert':
        tokenizer = AlbertTokenizer.from_pretrained(pretrained_weight)
        return tokenizer
    elif weight_tokenizer == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(pretrained_weight)
        return tokenizer
    else:
        print('the pretrained tokenizer you input does not be supported now!')
        sys.exit(1)

def get_model(pretrained_weight):
    #判断是哪个模型的预训练权重
    weight_model = pretrained_weight.split('-')[0]
    if weight_model == 'bert':
        model = BertModel.from_pretrained(pretrained_weight, return_dict=True)
        return model
    elif weight_model == 'xlnet':
        model = XLNetModel.from_pretrained(pretrained_weight, return_dict=True)
        return model
    elif weight_model == 'albert':
        model = AlbertModel.from_pretrained(pretrained_weight, return_dict=True)
        return model
    elif weight_model == 'roberta':
        model = RobertaModel.from_pretrained(pretrained_weight, return_dict=True)
        return model
    else:
        print('the model you input does not be supported now!')
        sys.exit(1)




