'''
when obtained the backdoored model, users will fine-tune the model at first
so we can get the fine-tuned model from here, we can also obtain the original CACC and ASR
'''

# Defend

import json
import argparse
from openbackdoor.data import load_dataset
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
from openbackdoor.utils import set_config, logger
from openbackdoor.utils.visualize import display_results
import torch
from openbackdoor.trainers import load_trainer
# file = 'badpre_bert/bert-base-uncased-attacked-random_default_hyperparameter/'
# file = '/home/user/OpenBackdoor-main/models/SST-2/dirty-badnets-0.1/1677651791/best.ckpt'
# file = '/home/user/OpenBackdoor-main/backdoored_models/SST-2/stylebkd/dirty-stylebkd-0.1/1678292052/best.ckpt'
# file = '/home/user/OpenBackdoor-main/backdoored_models/SST-2/lws/dirty-badnets-0.1/1678362648/best.ckpt'
# file = '/home/user/OpenBackdoor-main/backdoored_models/HSOL/lws/dirty-badnets-0.1/1678371303/best.ckpt'
# file = '/home/user/OpenBackdoor-main/fine-tuned/SST-2/lws/mix-Base-0.01/1678365283/best.ckpt'
# file = '/home/user/OpenBackdoor-main/backdoored_models/AGNews/lws/dirty-badnets-0.1/1678545747/best.ckpt'
# file = '/home/user/OpenBackdoor-main/fine-tuned/AGNews/lws/mix-Base-0.01/1678586696/best.ckpt'
# file = '/home/user/OpenBackdoor-main/backdoored_models/AGNews/lws/dirty-badnets-0.1/1678545747/best.ckpt'
# file = '/home/user/OpenBackdoor-main/backdoored_models/AGNews/synbkd/mix-synbkd-0.4/1678609056/best.ckpt'
# file = '/home/user/OpenBackdoor-main/backdoored_models/AGNews/synbkd/mix-synbkd-0.5/1678630387/best.ckpt'
# file = '/home/user/OpenBackdoor-main/fine-tuned/SST-2/neuba/mix-Base-0.01/1678677761/best.ckpt'
# file = '/home/user/OpenBackdoor-main/fine-tuned/HSOL/neuba/mix-Base-0.01/1678677913/best.ckpt'
# file = '/home/user/OpenBackdoor-main/fine-tuned/HSOL/badnets/mix-Base-0.01/1678261645/best.ckpt'
# file = '/home/user/OpenBackdoor-main/backdoored_models/SST-2/badnets/dirty-badnets-0.1/1677651791/best.ckpt'
file = '/home/user/OpenBackdoor-main/backdoored_models/HSOL/addsent/dirty-addsent-0.1/1678261519/best.ckpt'
backdoor_type = 'train' #'train'
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs/my_config/attack/hsol_addsent.json')
    args = parser.parse_args()
    return args


def main(config):
    # choose a victim classification model
    backdoored_model = load_victim(config["victim"])
    attacker = load_attacker(config["attacker"])
    target_dataset = load_dataset(**config["target_dataset"])

    if backdoor_type=='train':
        state_dict = torch.load(file, map_location={'cuda:0': 'cuda:1'})
        backdoored_model.load_state_dict(state_dict)
    elif backdoor_type=='pretrain':
        pass
    else:
        raise NameError('there is no this backdoor type',backdoor_type)


    if config["clean-tune"]:
        logger.info("Fine-tune model on {}".format(config["target_dataset"]["name"]))
        CleanTrainer = load_trainer(config["train"])
        backdoored_model = CleanTrainer.train(backdoored_model, target_dataset)

    logger.info("Evaluate backdoored model on {}".format(config["target_dataset"]["name"]))
    results = attacker.eval(backdoored_model, target_dataset)

    display_results(config, results)


if __name__ == '__main__':
    args = parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    config = set_config(config)
    #    //  "input_type": "inputs_embeds"  write in the victim if converter!!!!
    main(config)
