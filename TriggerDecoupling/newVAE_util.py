from torch.utils.data import DataLoader
from openbackdoor.victims import load_victim
from torch.utils.data import Dataset
from transformers import BertTokenizer
import torch
from openbackdoor.defenders.converter_defender import get_knn,get_efficient_knn
import csv
import json
from openbackdoor.utils import set_config,logger
from tqdm import tqdm

class Data(Dataset):
    def __init__(self,x,y):
        '''
        :param x: [x1,x2,...]
        :param y: [y1,y2,...]
        self.data [(x1,y1),...]
        '''
        self.data = list(zip(torch.tensor(x), torch.tensor(y)))
        # print('self.data',self.data)


    def __getitem__(self, idx):
        # item = {key: torch.tensor(val[idx]) for key, val in self.data.items()}
        return self.data[idx]
    def __len__(self):
        return len(self.data)

def get_inputIDS(file1,file2):
    all1 = []
    all2 = []
    with open(file1, 'r') as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            # for each in line:
            temp = [int(item) for item in line]
            all1.append(temp)
    with open(file2, 'r') as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            # for each in line:
            temp = [int(item) for item in line]
            all2.append(temp)
    return all1,all2

def get_iterators(opt,base_file = 'pretrained_datasets/'):
    # train_data = load_data('imdb/train.txt')

    x,y = get_inputIDS(base_file+'sst2_badnets_train_x.tsv',base_file+'sst2_badnets_train_y.tsv')

    train_data = Data(x,y)
    train_loader = DataLoader(train_data,batch_size=opt.batch_size,shuffle=True,num_workers=2)
    # test_data = load_data('imdb/test.txt')
    x,y = get_inputIDS(base_file+'sst2_badnets_test_x.tsv',base_file+'sst2_badnets_test_y.tsv')
    dev_data = Data(x,y)
    dev_loader = DataLoader(dev_data,batch_size=opt.batch_size,shuffle=False,num_workers=2)
    return train_loader,dev_loader
def get_sentence_list(file):
    type1_list = []
    type2_list = []
    with open(file,'r') as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            if line[1]=='0':
                type1_list.append(line[0])
            if line[1]=='1':
                type2_list.append(line[1])

    return type1_list,type2_list
'''
remove something from a tensor according to the index
'''
def del_tensor_ele(arr,index):
    arr1 = arr[0:index]
    arr2 = arr[index+1:]
    return torch.cat((arr1,arr2),dim=0)
def map_sentence_embed(sentence1_list,sentence2_list,max_length,backdoor_model_file,config,device):
    backdoored_model = load_victim(config["victim"])
    backdoored_model.to(device)
    state_dict = torch.load(backdoor_model_file,map_location={'cuda:0':device,'cuda:1':device})
    backdoored_model.load_state_dict(state_dict)
    raw_embed = backdoored_model.raw_embed
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    logger.info('start to convert the sentences (type1) to input_ids...')
    input_ids1 = tokenizer(sentence1_list, padding='max_length', max_length=max_length,truncation=True)['input_ids']
    logger.info('start to convert the sentences (type2) to input_ids...')
    input_ids2 = tokenizer(sentence2_list, padding='max_length', max_length=max_length, truncation=True)['input_ids']
    return raw_embed(torch.tensor(input_ids1).to(device)),raw_embed(torch.tensor(input_ids2).to(device)),input_ids1,input_ids2
def get_two_list(topk,file,max_length,backdoor_model_file,config,device):
    '''
    Args:
        topk:
    Returns:
        x[list]: [batch_size, seq_len]  input_ids
    '''
    x = []
    y = []
    type1,type2 = get_sentence_list(file)
    embeddings_list,embeddings_list2,input_ids1,input_ids2 = map_sentence_embed(type1,type2,max_length,backdoor_model_file,config,device)
    logger.info('start to get knn (type1)....')
    # for id in tqdm(range(len(embeddings_list))):
    for id in tqdm(range(len(input_ids1))):
        # if id>=from_id and id<to_id:
        remain_list = del_tensor_ele(embeddings_list,id)
        # remain_list.pop(id)
        remain_input_list= input_ids1.copy()
        remain_input_list.pop(id)
        # remain_list_tensor = torch.tensor(remain_list).to(device)
        # each_list = torch.tensor(embeddings_list[id]).to(device).unsqueeze(0)
        each_list = embeddings_list[id].unsqueeze(0)
        # print(remain_list_tensor.size()) [n-1, seq, embed]
        # print(each_list.size()) [1,seq,embed]
        n = remain_list.size()[0]
        remain_list_tensor = remain_list.view(n,-1)
        each_list = each_list.view(1,-1)
        value,index = get_knn(remain_list_tensor,each_list,topk=topk) # the index is [remain_list.size()[0], 1]
        for j in range(topk):
            x.append(input_ids1[id])
            y.append(remain_input_list[index[j][0].item()])
    logger.info('start to get knn (type2)....')
    for id in tqdm(range(len(input_ids2))):
        # if id >= from_id and id < to_id:
            # remain_list = embeddings_list2.copy()
            # remain_list.pop(id)
        remain_list = del_tensor_ele(embeddings_list2,id)
        remain_input_list= input_ids2.copy()
        remain_input_list.pop(id)
        # remain_list_tensor = torch.tensor(remain_list).to(device)
        each_list = embeddings_list2[id].unsqueeze(0)
        n = remain_list.size()[0]
        remain_list_tensor = remain_list.view(n,-1)
        each_list = each_list.view(1,-1)
        value,index = get_knn(remain_list_tensor,each_list,topk=topk) # the index is [remain_list.size()[0], 1]
        for j in range(topk):
            x.append(input_ids2[id])
            y.append(remain_input_list[index[j][0].item()])
    return x,y
def pretrain_data_generation(topk,file,max_length,backdoor_model_file,config,device,type):
    x,y = get_two_list(topk,file,max_length,backdoor_model_file,config,device)
    logger.info('write the x file...')
    with open(base_file+'sst2_badnets_'+type+'_x.tsv', 'w') as f:  #str(int(to_id/330))+
        writer = csv.writer(f, delimiter='\t')
        for line in tqdm(x):
            writer.writerow(line)
    logger.info('write the y file...')
    with open(base_file+'sst2_badnets_'+type+'_y.tsv', 'w') as f:  #str(int(to_id/330))+
        writer = csv.writer(f, delimiter='\t')
        for line in tqdm(y):
            writer.writerow(line)

if __name__ == '__main__':
    topk = 100
    opt = None
    max_length = 128
    backdoor_model_file = './models/dirty-badnets-0.1/1677651791/best.ckpt'
    config_path = './configs/my_config/converter_badnets_config.json'
    base_file = 'pretrained_datasets/'

    device = 'cuda:0'
    type = 'test'

    # parallel process the data
    # order = 1  #from 1 start
    # from_id = 330*(order-1)
    # to_id = 330*order

    file = 'datasets/SentimentAnalysis/SST-2/'+type+'.tsv'
    with open(config_path, 'r') as f:
        config = json.load(f)
    config = set_config(config)
    pretrain_data_generation(topk,file,max_length,backdoor_model_file,config,device,type)
    # get_iterators(opt)
