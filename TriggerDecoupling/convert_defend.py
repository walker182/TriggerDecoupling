# Defend
import os
import json
import argparse
import torch

import openbackdoor as ob
from openbackdoor.data import load_dataset, get_dataloader, wrap_dataset
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
from openbackdoor.defenders import load_defender
from openbackdoor.utils import set_config, logger
from openbackdoor.utils.visualize import display_results
# from openbackdoor.trainers import Trainer
from openbackdoor.trainers import load_trainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs/my_config/warm_up/stylebkd_hsol.json')
    parser.add_argument('--basemodel',type=str,default='sst2_addsent')
    args = parser.parse_args()
    return args

finetuned_model = {
    'sst2_badnets':'/home/user/OpenBackdoor-main/fine-tuned/SST-2/badnets/mix-Base-0.01/1678685780/best.ckpt',
    'sst2_synbkd':'/home/user/OpenBackdoor-main/fine-tuned/SST-2/synbkd/mix-Base-0.01/1678259486/best.ckpt',
    'sst2_stylebkd':'/home/user/OpenBackdoor-main/fine-tuned/SST-2/stylebkd/mix-Base-0.01/1678325803/best.ckpt',
    'sst2_lws':'/home/user/OpenBackdoor-main/fine-tuned/SST-2/lws/mix-Base-0.01/1678365283/best.ckpt',
    'sst2_addsent':'/home/user/OpenBackdoor-main/fine-tuned/SST-2/addsent/mix-Base-0.01/1678260874/best.ckpt',
    'sst2_neuba':'/home/user/OpenBackdoor-main/fine-tuned/SST-2/neuba/mix-Base-0.01/1678677761/best.ckpt',
    'sst2_por':'/home/user/OpenBackdoor-main/fine-tuned/SST-2/por/mix-Base-0.01/1678684237/best.ckpt',
    'hsol_badnets': '/home/user/OpenBackdoor-main/fine-tuned/HSOL/badnets/mix-Base-0.01/1678261645/best.ckpt',
    'hsol_synbkd': '/home/user/OpenBackdoor-main/fine-tuned/HSOL/synbkd/mix-Base-0.01/1678306947/best.ckpt',
    'hsol_stylebkd':'/home/user/OpenBackdoor-main/fine-tuned/HSOL/stylebkd/mix-Base-0.01/1678327630/best.ckpt',
    'hsol_lws': '/home/user/OpenBackdoor-main/fine-tuned/HSOL/lws/mix-Base-0.01/1678373604/best.ckpt',
    'hsol_addsent': '/home/user/OpenBackdoor-main/fine-tuned/HSOL/addsent/1678686341/best.ckpt',
    'hsol_neuba': '/home/user/OpenBackdoor-main/fine-tuned/HSOL/neuba/3trigger/mix-Base-0.01/1678681996/best.ckpt',
    'hsol_por': '/home/user/OpenBackdoor-main/fine-tuned/HSOL/por/3trigger/mix-Base-0.01/1678681964/best.ckpt',
    'agnews_badnets': '/home/user/OpenBackdoor-main/fine-tuned/AGNews/badnets/mix-Base-0.01/1678264856/best.ckpt',
    'agnews_stylebkd': '/home/user/OpenBackdoor-main/fine-tuned/AGNews/stylebkd/mix-Base-0.01/1678649157/best.ckpt',
    'agnews_addsent': '/home/user/OpenBackdoor-main/fine-tuned/AGNews/addsent/mix-Base-0.01/1678264890/best.ckpt',
}
backdoorModel = {
    'hsol_stylebkd':'/home/user/OpenBackdoor-main/backdoored_models/HSOL/stylebkd/dirty-stylebkd-0.1/1678326660/best.ckpt',
    'agnews_stylebkd':'/home/user/OpenBackdoor-main/backdoored_models/AGNews/stylebkd/dirty-stylebkd-0.2/1678638884/best.ckpt',
    'sst2_badnets':'/home/user/OpenBackdoor-main/backdoored_models/SST-2/badnets/dirty-badnets-0.1/1677651791/best.ckpt',
    'sst2_addsent':'/home/user/OpenBackdoor-main/backdoored_models/SST-2/addsent/dirty-addsent-0.1/1678260555/best.ckpt',
    'sst2_synbkd':'/home/user/OpenBackdoor-main/backdoored_models/SST-2/synbkd/dirty-synbkd-0.1/1678259175/best.ckpt',
    'sst2_stylebkd':'/home/user/OpenBackdoor-main/backdoored_models/SST-2/stylebkd/dirty-stylebkd-0.1/1678292052/best.ckpt',
    'sst2_por':'/home/user/OpenBackdoor-main/POR-backdoor-model/pytorch_model.bin',
    'sst2_neuba':'/home/user/OpenBackdoor-main/NeuBA/pytorch_model.bin',
    'hsol_por':'/home/user/OpenBackdoor-main/POR-backdoor-model/pytorch_model.bin',
    'hsol_neuba':'/home/user/OpenBackdoor-main/NeuBA/pytorch_model.bin',
    'hsol_badnets':'/home/user/OpenBackdoor-main/backdoored_models/HSOL/badnets/dirty-badnets-0.1/1678261348/best.ckpt',
    'hsol_addsent':'/home/user/OpenBackdoor-main/backdoored_models/HSOL/addsent/dirty-addsent-0.1/1678261519/best.ckpt',
    'hsol_synbkd':'/home/user/OpenBackdoor-main/backdoored_models/HSOL/synbkd/dirty-synbkd-0.1/1678292074/best.ckpt',
    'agnews_badnets':'/home/user/OpenBackdoor-main/backdoored_models/AGNews/badnets/dirty-badnets-0.1/1678262419/best.ckpt',
    'agnews_addsent':'/home/user/OpenBackdoor-main/backdoored_models/AGNews/addsent/dirty-addsent-0.1/1678262457/best.ckpt',
}
def mainLoadExist(config,basemodel):
    # file = './models/clean-badnets-0.1/1677070294/best.ckpt'
    # file = './models/dirty-synbkd-0.1/1677649553/best.ckpt'
    # file = './models/dirty-badnets-0.1/1677651791/best.ckpt'
    # file = '/home/user/OpenBackdoor-main/fine-tuned/SST-2/badnets/mix-Base-0.01/1678685780/best.ckpt'
    # file = '/home/user/OpenBackdoor-main/fine-tuned/SST-2/synbkd/mix-Base-0.01/1678259486/best.ckpt'
    # file = '/home/user/OpenBackdoor-main/fine-tuned/SST-2/stylebkd/mix-Base-0.01/1678325803/best.ckpt'
    Pretrain = False
    if basemodel in ['sst2_por','sst2_neuba' ,'hsol_por','hsol_neuba']:
        Pretrain = True
    if config["defender"]["name"] == 'converter' and basemodel!='origin':
        file = backdoorModel[basemodel]
    elif config["defender"]["name"] == 'converter' and basemodel=='origin':
        print('load origin model')
        file = ''
    else:
        file = finetuned_model[basemodel]
    backdoored_model = load_victim(config["victim"])

    # choose attacker and initialize it with default parameters
    # print('1',config["victim"]["name"])

    attacker = load_attacker(config["attacker"])
    defender = load_defender(config["defender"])

    # choose target and poison dataset
    target_dataset = load_dataset(**config["target_dataset"])
    if defender.pre is True:  #We are not considering this settings!
        poison_dataset = load_dataset(**config["poison_dataset"])
        logger.info("Train backdoored model on {}".format(config["poison_dataset"]["name"]))
        backdoored_model = attacker.attack(backdoored_model, poison_dataset, defender)
        logger.info("Evaluate backdoored model on {}".format(config["target_dataset"]["name"]))
        results = attacker.eval(backdoored_model, target_dataset, defender)
    else:
        if Pretrain==False or config["defender"]["name"]=='gpt':
            if file!='':
                state_dict = torch.load(file,map_location={'cuda:0':'cuda:0','cuda:1':'cuda:0'})
                backdoored_model.load_state_dict(state_dict)
        if config["defender"]["name"] == 'converter':
            # Then retrain the backdoor model and train the converter

            print("train converter and retrain the model on {}".format(config["target_dataset"]["name"]))
            converterTrainer = load_trainer(config['convert_train'])
            backdoored_model,defender = converterTrainer.train(backdoored_model,target_dataset,defender=defender)
            # torch.save(backdoored_model,'/media/data/user/adv_defense/backdoormodel_agnews.pt')
            # torch.save(defender.converter,'/home/user/OpenBackdoor-main/visual_tensors/converter/converter.pt')

            # backdoored_model = torch.load('/media/data/user/adv_defense/backdoormodel_agnews.pt')
            # defender = torch.load('/media/data/user/adv_defense/converter_agnews.pt')
            if config['convert_train']['warm_up']==False:
                logger.info("Evaluate backdoored model on {}".format(config["target_dataset"]["name"]))
                #warm up here hurts me a lot!!!!
                results = attacker.eval(backdoored_model, target_dataset, defender,backdoored_model.raw_embed,warmup=False)
            # results = attacker.eval(backdoored_model, target_dataset, defender, backdoored_model.raw_embed,
            #                             warmup=False)
                # results = attacker.eval(backdoored_model, target_dataset, defender, defender.converter.vae.embedding.weight)
        else:
            logger.info("Evaluate backdoored model on {}".format(config["target_dataset"]["name"]))
            results = attacker.eval(backdoored_model, target_dataset, defender)

    # if config['convert_train']['warm_up'] == False:
    display_results(config, results)

def main(config):
    # choose a victim classification model
    victim = load_victim(config["victim"])
    # choose attacker and initialize it with default parameters
    attacker = load_attacker(config["attacker"])
    defender = load_defender(config["defender"])
    # choose target and poison dataset
    target_dataset = load_dataset(**config["target_dataset"])
    poison_dataset = load_dataset(**config["poison_dataset"])
    # target_dataset = attacker.poison(victim, target_dataset)
    # launch attacks
    logger.info("Train backdoored model on {}".format(config["poison_dataset"]["name"]))
    backdoored_model = attacker.attack(victim, poison_dataset)
    # Then retrain the backdoor model and train the converter
    # backdoored_model,defender = converter_trainer.train(backdoored_model,defender,target_dataset)
    print("train converter and retrain the model on {}".format(config["target_dataset"]["name"]))
    # converterTrainer = Trainer()
    # converterTrainer = attacker.poison_trainer
    converterTrainer = load_trainer(config['convert_train'])
    backdoored_model,defender = converterTrainer.train(backdoored_model,target_dataset,defender=defender)



    logger.info("Evaluate backdoored model on {}".format(config["target_dataset"]["name"]))
    results = attacker.eval(backdoored_model, target_dataset, defender)

    display_results(config, results)

    # Fine-tune on clean dataset
    '''
    print("Fine-tune model on {}".format(config["target_dataset"]["name"]))
    CleanTrainer = ob.BaseTrainer(config["train"])
    backdoored_model = CleanTrainer.train(backdoored_model, wrap_dataset(target_dataset, config["train"]["batch_size"]))
    '''



if __name__ == '__main__':
    args = parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    config = set_config(config)


    # main(config)
    basemodel = args.basemodel
    mainLoadExist(config,basemodel)
