import sys
sys.path.append('...\WMDD')

import random
import torch
import argparse

import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns

from datasets.get_data import Get_data
from train.trainer import Trainer
from configs.default import default_args

def setup_seed(seed):
     random.seed(seed)
     np.random.seed(seed)
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     torch.backends.cudnn.deterministic = True
     torch.backends.cudnn.benchmark = False


def tnse(source_X, source_Y,target_X, target_Y,classes):
    feature = TSNE(n_components=2, learning_rate='auto',init='random', perplexity=30,
                   early_exaggeration =30,n_iter=2000).fit_transform(source_X) 
    feature1 = TSNE(n_components=2, learning_rate='auto',init='random', perplexity=30,
                    early_exaggeration = 30,n_iter=2000).fit_transform(target_X) 

    palet = sns.hls_palette(classes, l=0.5, s=0.9)
    palette = np.array(palet)
    volume = palette[source_Y.astype(np.int32)]
    volume1 = palette[target_Y.astype(np.int32)]
    return feature,feature1,volume,volume1


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='ENABL3S', help='Dataset is ENABL3S or DSADS?')
    parser.add_argument('--eval_only', default=False, help='evaluation only option')

    parser.add_argument('--margin', type=float, default= 0.1, help='ratio of classifier discrepancy')
    parser.add_argument('--trade_off', type=float, default= 5.0, help='the trade-off between loss')
    parser.add_argument('--dis_c_ratio', type=float, default= 5.0, help='discrpency loss')
    
    known_args, _ = parser.parse_known_args()
    for arg_key, default_value in default_args.items():
        parser.add_argument(f'--{arg_key}', default=default_value, type=type(default_value))
    args = parser.parse_args()

    if args.data_name == 'DSADS':
        args.total_subject = 8
        classes = 19
    else:
        args.total_subject = 10
        classes = 7

    setup_seed(args.seed)

    target_subject = 0
    args.checkpoint_dir = './result/'+args.data_name+'/target_subject_'+str(target_subject)
    train = Trainer(args=args, target_subject=target_subject)
    train.load_model()

    # Iuput feature
    data = Get_data(data_name=args.data_name, total_subject = args.total_subject,target_subject=target_subject)
    input_source_X, source_Y = data.input_source_data()
    input_target_X, target_Y = data.input_target_data()
    X_1,X_2,Y_1,Y_2 = tnse(input_source_X.reshape(len(input_source_X),-1), source_Y, input_target_X.reshape
                           (len(input_target_X),-1), target_Y,classes)
    
    data_1 = np.concatenate((X_1,source_Y.reshape(-1,1).astype(np.int32)), axis=1)
    data_2 = np.concatenate((X_2,target_Y.reshape(-1,1).astype(np.int32)), axis=1)
    
    np.savetxt( "EN_S_Input.csv", data_1, delimiter="," )
    np.savetxt( "EN_T_Input.csv", data_2, delimiter="," )

    # Adaption feature
    source_X = train.feature_output(input_source_X, source_Y)
    target_X = train.feature_output(input_target_X, target_Y)
    feature,feature1,volume,volume1 = tnse(source_X, source_Y,target_X, target_Y,classes)

    data_S = np.concatenate((feature,source_Y.reshape(-1,1).astype(np.int32)), axis=1)
    data_T = np.concatenate((feature1,target_Y.reshape(-1,1).astype(np.int32)), axis=1)

    np.savetxt( "EN_S_Adaption.csv", data_S, delimiter="," )
    np.savetxt( "EN_T_Adaption.csv", data_T, delimiter="," )