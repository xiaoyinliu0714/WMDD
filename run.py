import os
#os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import argparse
import random
import numpy as np
import torch
import wandb
from train.trainer import Trainer
from configs.default import default_args

def setup_seed(seed):
     random.seed(seed)
     np.random.seed(seed)
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     torch.backends.cudnn.deterministic = True
     torch.backends.cudnn.benchmark = False

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='ENABL3S', help='Dataset is ENABL3S or DSADS?')
    parser.add_argument('--margin', type=float, default= 0.1, help='ratio of classifier discrepancy')
    parser.add_argument('--trade_off', type=float, default= 5.0, help='the trade-off between loss')
    
    parser.add_argument('--is_weight', default= True, help='whether to use weight')
    parser.add_argument('--is_adversarial', default= True, help='whether to use adversarial learning')
    parser.add_argument('--eval_only', default= False, help='evaluation only option')
    
    known_args, _ = parser.parse_known_args()
    for arg_key, default_value in default_args.items():
        parser.add_argument(f'--{arg_key}', default=default_value, type=type(default_value))
    args = parser.parse_args()

    if args.data_name == 'DSADS':
        args.total_subject = 8
    else:
        args.total_subject = 10
    
    # Using wandb to save data
    # 1. You should create two projects, named ENABL3S and DSADS.
    # 2. Your API key for logging in to the wandb library.
    WANDB_API_KEY  = "..."
    # 3. Your entity when logging wandb
    Entity = "..." 
    # 4. Use the following command to get only offline data
    # os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY

    setup_seed(args.seed)

    for target_subject in range(0, args.total_subject):
        args.checkpoint_dir = './result/'+args.data_name+'/target_subject_'+str(target_subject)
        print(target_subject)

        if not os.path.exists(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)
        train = Trainer(args=args, target_subject=target_subject)

        if not args.eval_only:
            wandb.init(project=args.data_name,entity=Entity,)
            train.train()
            train.save_model()
            wandb.finish()
        else:
            train.load_model()
            source_accuracy, target_accuracy = train.evaluate()
            print({'target accuracy': target_accuracy})