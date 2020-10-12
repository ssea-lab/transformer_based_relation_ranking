import os
import pickle
import random
def cre_relation_dict(relations_file):
    rel_dict = {}
    line_num = 0
    with open(relations_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            line_num += 1
            rel_dict[line_num] = line.rstrip()
    print(len(rel_dict))
    pickle.dump(rel_dict, open('../data/rel_dict.pkl', 'wb'))

def lis_remove_element(lista, element):
    if element in lista:
        lista.remove(element)
    return lista


def get_standard_data(file_name):
    total_list = []
    idx = list(range(1, 6701))
    neg_sample = 50
    rel_dict = pickle.load(open('../data/rel_dict.pkl', 'rb'))
    with open(file_name, 'r',encoding='utf-8') as fin:
        file_type = file_name.split('/')[-1].split('.')[0]
        for line in fin:
            single_sample_list = []
            line_split = line.strip().split('\t')
            single_sample_list.append(line_split[2].replace('#head_entity#', 'X'))
            pos_idx = int(line_split[0])
            single_sample_list.append(pos_idx)

            if line_split[1] == 'noNegativeAnswer':
                neg_idx = random.sample(lis_remove_element(list(range(1, 6701)), pos_idx), neg_sample)
                single_sample_list.append(neg_idx)
            else:
                neg_idx = lis_remove_element([int(i) for i in line_split[1].split(' ')], pos_idx)
                if len(neg_idx) >= 50:
                    single_sample_list.append(neg_idx[:50])
                else:
                    neg_idx.extend(random.sample(lis_remove_element(list(range(1, 6701)), pos_idx), neg_sample - len(neg_idx)))
                    single_sample_list.append(neg_idx)
                
            
            single_sample_list[1] = rel_dict[single_sample_list[1]]
            single_sample_list[2] = [rel_dict[i] for i in single_sample_list[2]]
            assert len(single_sample_list[2]) == 50, 'the negative sample is not 50, %s'%len(single_sample_list[2])

            total_list.append(single_sample_list[:])
    
    print(len(total_list))
    print(total_list[0])
    pickle.dump(total_list, open('../data/%s_list.pkl'%file_type, 'wb'))




if __name__ == '__main__':
    # cre_relation_dict('../data/sq_relations/relation.2M.list')
    # get_standard_data('../data/sq_relations/valid.replace_ne.withpool')
    get_standard_data('../data/sq_relations/train.replace_ne.withpool')
    get_standard_data('../data/sq_relations/test.replace_ne.withpool')
