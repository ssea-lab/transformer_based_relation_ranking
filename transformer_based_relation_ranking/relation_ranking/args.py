import os
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description='kbqa-FB model')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dropout_prob', type=float, default=0.2)
    parser.add_argument('--neg_size', type=int, default=50, help='negtive sampling number')
    parser.add_argument('--loss_margin', type=float, default=1)

    parser.add_argument('--rels_maxlen', type=int, default=20)
    parser.add_argument('--questions_maxlen', type=int, default=40, help='used to adjust the dynamic sequence length')

    #add for pretrained weight(目前使用bert-base-uncased， xlnet-base-cased，albert-base-v1，roberta-base)
    parser.add_argument('--pretrained_weight', type=str, default='bert-base-uncased')
    #用来确定是否使用各个层的hidden states的加权求和
    parser.add_argument('--weighted_hidden_states', action='store_true', default=False)


    parser.add_argument('--test', action='store_true', dest='test', help='get the testing set result')
    parser.add_argument('--dev', action='store_true', dest='dev', help='get the development set result')
    parser.add_argument('--log_every', type=int, default=200)
    parser.add_argument('--save_every', type=int, default=3000)
    parser.add_argument('--no_cuda', action='store_false', help='do not use CUDA', dest='cuda')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device to use') # use -1 for CPU
    parser.add_argument('--seed', type=int, default=1111, help='random seed for reproducing results')
    
    parser.add_argument('--train_file', type=str, default='/data/wangqingbin/kbka2/transformer_based_relation_ranking/data/train_list.pkl')
    parser.add_argument('--valid_file', type=str, default='/data/wangqingbin/kbka2/transformer_based_relation_ranking/data/valid_list.pkl')
    parser.add_argument('--test_file', type=str, default='/data/wangqingbin/kbka2/transformer_based_relation_ranking/data/test_list.pkl')
    parser.add_argument('--resume_trained_weight', type=str, default='')
    parser.add_argument('--save_path', type=str, default='/data/wangqingbin/kbka2/transformer_based_relation_ranking/relation_ranking/saved_checkpoints')

    # added for testing
    parser.add_argument('--trained_weight', type=str, default='/data/wangqingbin/kbka2/transformer_based_relation_ranking/relation_ranking/saved_checkpoints/bert_modelweight.pt')
    parser.add_argument('--results_path', type=str, default='/data/wangqingbin/kbka2/transformer_based_relation_ranking/relation_ranking/results')
    parser.add_argument('--write_res', default=True, help='write predict results to file or not')
    parser.add_argument('--write_score', action='store_true')
    parser.add_argument('--predict', default=True)
    args = parser.parse_args()
    return args
