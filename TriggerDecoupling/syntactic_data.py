# from openbackdoor.utils import logger
import OpenAttack as oa
import os
import argparse
import csv
from tqdm import tqdm
'''
for SST-2 and HSOL, train/dev/test.tsv，for AGNews train/test csv
As for the format, SST-2 and HSOL are the same, while AGNews has no head, and label ranks the first
'''
parser = argparse.ArgumentParser()
parser.add_argument("--split_num", default=16, type=int)
parser.add_argument("--index",default=1,type=int)
args = parser.parse_args()
INDEX = args.index
SPLIT_NUM = args.split_num
MODE = 'test'
datasets_type = 'AGNews'
def transform(text,scpn):
    try:
        paraphrase = scpn.gen_paraphrase(text, [scpn.templates[-1]])[0].strip()
    except Exception:
        # logger.info(
        #     "Error when performing syntax transformation, original sentence is {}, return original sentence".format(
        #         text))
        paraphrase = text

    return paraphrase
def dataGeneration(frompath,topath):
    try:
        scpn = oa.attackers.SCPNAttacker()
    except:
        base_path = os.path.dirname(__file__)
        os.system('bash {}/openbackdoor/attackers/poisoners/utils/syntactic/download.sh'.format(base_path))
        scpn = oa.attackers.SCPNAttacker()
    with open(frompath,'r') as f1:
        with open(topath,'w',newline='') as f2:
            tsvwriter = csv.writer(f2,delimiter='\t')
            for id,line in enumerate(tqdm(f1.readlines())):
                line = line.rstrip().split('\t')
                # for text, label, poison_label in tqdm(data):
                tsvwriter.writerow([transform(line[0],scpn),line[1]])
                # poisoned.append((transform(line[0]), line[1]))
                # return poisoned



def splitDataset(frompath,split_count,topath,dataset_type=None):
    '''
    This fun is just for SST-2
    Args:
        frompath:
        topath:
    Returns:
    '''
    dataset = []
    if dataset_type=='AGNews':

        with open(frompath, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                label, headline, body = row
                text_a = headline.replace('\\', ' ')
                text_b = body.replace('\\', ' ')
                example = [text_a + " " + text_b, int(label) - 1]
                dataset.append(example)
    else:
        with open(frompath,'r') as f1:
            for id,line in enumerate(f1.readlines()):
                if id==0:
                    continue
                line = line.rstrip().split('\t')
                dataset.append(line)
    size = int(len(dataset)/split_count)

    for id in range(1,split_count+2):
        if id==split_count+1:
            sub_dataset = dataset[(id-1)*size:]
        else:
            sub_dataset = dataset[(id-1)*size:id*size]
        with open(topath+MODE+'-'+str(id)+'.tsv','w',newline='') as f2:
            tsvwriter = csv.writer(f2,delimiter='\t')
            for line in sub_dataset:
                tsvwriter.writerow(line)

def split_main(frompath,topath,dataset_type=None):

    splitDataset(frompath,SPLIT_NUM,topath,dataset_type)
def generate_main(frompath,topath):


    dataGeneration(frompath,topath)

def concatMode(final_path):
    dataset = []
    for index in range(1,SPLIT_NUM+2):
        path = 'myPoisoned/syntactic/'+datasets_type+'/'+MODE+'poisoned-'+str(index)+'.tsv'
        with open(path,'r') as f:
            for line in f.readlines():
                line = line.rstrip().split('\t')
                dataset.append(line)
    with open(final_path,'w',newline='') as f2:
        writer = csv.writer(f2,delimiter='\t')
        for line in dataset:
            writer.writerow(line)

def concatall(final_path,path1,path1to):
    dataset = []
    if datasets_type=='AGNews':
        suffix = '.csv'
        all_list = ['train','test']
    else:
        suffix = '.tsv'
        all_list = ['train','dev','test']
    for index in all_list:
        # path = 'myPoisoned/syntactic/'+datasets_type+'/'+index+'poisoned-all.tsv'
        # path2 = 'datasets/SentimentAnalysis/SST-2/'+index+'.tsv'
        path = path1 + index+'poisoned-all.tsv'
        path2 = path1to+index+suffix
        origin = []
        poison = []
        with open(path,'r') as f:
            for line in f.readlines():
                line = line.rstrip().split('\t')
                poison.append(line[0])
        if datasets_type=='AGNews':
            with open(path2, encoding='utf8') as f:
                reader = csv.reader(f, delimiter=',')
                for idx, row in enumerate(reader):
                    label, headline, body = row
                    text_a = headline.replace('\\', ' ')
                    text_b = body.replace('\\', ' ')
                    origin.append(text_a + " " + text_b)
                    # example = [text_a + " " + text_b, int(label) - 1]
                    # dataset.append(example)
        else:
            with open(path2,'r') as f2:
                for id,line in enumerate(f2.readlines()):
                    if id==0:
                        continue
                    line = line.rstrip().split('\t')
                    origin.append(line[0])
        # print(len(origin))
        # print(len(poison))
        assert len(origin) == len(poison)
        dataset += [list(item) for item in zip(origin,poison)]
        # for x,y in zip(origin,poison):
        #     if y=='literary purists may not be pleased , but as far as mainstream matinee-style entertainment goes , it does a bang-up job of pleasing the crowds .':
        #         print(x)
        #         exit()
        # exit()
    with open(final_path,'w',newline='') as f2:
        writer = csv.writer(f2,delimiter='\t')
        for line in dataset:
            writer.writerow(line)



if __name__ == '__main__':
    # frompath = 'datasets/SentimentAnalysis/SST-2/'+MODE+'.tsv'
    frompath = 'datasets/TextClassification/agnews/'+MODE+'.csv'
    # frompath = 'datasets/Toxic/hsol/' +MODE+'.tsv'
    topath = 'myPoisoned/syntactic/'+datasets_type+'/split-240/'

    split_main(frompath,topath,datasets_type)  #step1, given MODE,split_num split the train/dev/test(MODE) into sub file. each sub file is tied with INDEX
    exit()

    #frompath = 'myPoisoned/syntactic/'+datasets_type+'/'+MODE+'-'+str(INDEX)+'.tsv'
    #topath = 'myPoisoned/syntactic/'+datasets_type+'/'+MODE+'poisoned-'+str(INDEX)+'.tsv'
    #generate_main(frompath,topath)  #step2, given MODE and INDEX, generate the corresponding file whose sentences are poisoned
    #exit()

    #concatMode('myPoisoned/syntactic/'+MODE+'poisoned-all.tsv') # concat the MODE into a full file

    final_path = 'myPoisoned/syntactic/'+datasets_type+'/dict-all.tsv'
    path = 'myPoisoned/syntactic/'+datasets_type+'/'
    # path2 = 'datasets/SentimentAnalysis/SST-2/'
    path2 = 'datasets/TextClassification/agnews/'
    concatall(final_path,path,path2)  # concat all the MODE to one file
    # file = '/home/user/OpenBackdoor-main/datasets/TextClassification/agnews/test.csv'
    # temp()