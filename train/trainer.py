import copy
import wandb
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from datasets.data import Datasets
from model.network import Feature, Predictor

torch.autograd.set_detect_anomaly(True)

device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)

class Trainer(object):
    def __init__(self, args, target_subject):
        self.args = args
        self.data_name = args.data_name
        self.batch_size = args.batch_size
        self.max_iter = args.max_iter
        self.eval_interval = args.eval_interval
        self.checkpoint_dir = args.checkpoint_dir
        self.lr = args.lr
        self.sensor_num = args.sensor_num
        self.total_subject = args.total_subject
        self.target_subject = target_subject

        self.num_F = args.num_F  
        self.num_C = args.num_C 
        self.num_C1 = (self.total_subject-1)

        idx_vec = list(range(self.total_subject))
        self.idx_target = [copy.deepcopy(idx_vec[self.target_subject])][0]
        idx_vec.pop(self.target_subject)
        self.idx_source = idx_vec
        self.data_iteror = iter([])

        self.data = Datasets(data_name=self.data_name, batch_size=self.batch_size,
                             is_one_hot=False, is_normalized=False, is_resize=True,
                             sensor_num=self.sensor_num, X_dim=4,total_subject = self.args.total_subject)

        self.net_dict = self.init_model()
        if args.eval_only:
            self.load_model()
        else:
            self.set_optimizer(which_opt=args.optimizer, lr=args.lr)

    def init_model(self):
        F_list = []
        for _ in range(self.num_F):
            F_list.append(Feature(dataset=self.data_name, sensor_num=self.sensor_num).to(device))
        C_list = []
        for _ in range(self.num_C):
            C_list.append(Predictor(dataset=self.data_name).to(device))
        C1_list = []
        for _ in range(self.num_C1):
            C1_list.append(Predictor(dataset=self.data_name).to(device))

        return {'F': F_list, 'C': C_list, 'C1': C1_list}

    def train_model(self):
        for key in self.net_dict.keys():
            for i in range(len(self.net_dict[key])):
                self.net_dict[key][i].train()

    def eval_model(self):
        for key in self.net_dict.keys():
            for i in range(len(self.net_dict[key])):
                self.net_dict[key][i].eval()

    def step_model(self, keys):
        for key in keys:
            for i in range(len(self.opt_dict[key])):
                self.opt_dict[key][i].step()

    def set_optimizer(self, which_opt='momentum', lr=0.001, momentum=0.9):
        self.opt_dict = {}
        for key in self.net_dict.keys():
            self.opt_dict.update({key: []})
            for i in range(len(self.net_dict[key])):
                if which_opt == 'momentum':
                    opt = optim.SGD(self.net_dict[key][i].parameters(),
                                    lr=lr, weight_decay=0.0005,
                                    momentum=momentum)
                elif which_opt == 'adam':
                    opt = optim.Adam(self.net_dict[key][i].parameters(),
                                     lr=lr, weight_decay=0.0005)
                else:
                    raise Exception("Unrecognized optimization method.")
                self.opt_dict[key].append(opt)

    def reset_grad(self,keys):
        for key in keys:
            for i in range(len(self.opt_dict[key])):
                self.opt_dict[key][i].zero_grad()

    def train(self):
        for iter_num in range(1, self.max_iter + 1):
            self.train_model()
            img_t,_ = self.data.get_samples(self.idx_target,False)
            img_t = Variable(img_t.to(device))
            source_x ={}
            source_y = {}

            # Update auxiliary classifiers f' network
            cnt = 0
            for idx in self.idx_source:
                img_s, label_s = self.data.get_samples(idx)
                img_s = Variable(img_s.to(device))
                label_s = Variable(label_s.long().to(device)).squeeze()
                source_x[idx],source_y[idx] = img_s, label_s

                loss_adv_src, loss_adv_src, loss_adv_tgt = self.calc_single_loss(img_s, label_s, img_t,cnt)
                loss = -1*(loss_adv_tgt - self.args.margin * loss_adv_src)
                loss.backward(retain_graph=True)
                self.opt_dict['C1'][cnt].step()   
                cnt = cnt + 1

            # Update classifiers f and feature network
            cnt = 0
            classifier_loss = torch.empty(len(self.idx_source))
            classifier_loss_adv_src = torch.empty(len(self.idx_source))
            classifier_loss_adv_tgt = torch.empty(len(self.idx_source))
            for idx in self.idx_source:
                img_s, label_s = source_x[idx],source_y[idx]
                classifier_loss[cnt], classifier_loss_adv_src[cnt], classifier_loss_adv_tgt[cnt] \
                    = self.calc_single_loss(img_s, label_s, img_t, cnt)
                cnt = cnt + 1

            transfer_loss = classifier_loss_adv_tgt - self.args.margin * classifier_loss_adv_src

            if self.args.is_weight:
                weight = F.softmax(transfer_loss, dim=0)
            else:
                weight = torch.ones(len(self.idx_source))/cnt

            loss_f_c = torch.matmul(weight, classifier_loss + transfer_loss)

            self.reset_grad(['F', 'C'])
            loss_f_c.backward(retain_graph=True)
            self.step_model(['F', 'C'])

            # Adversarial learning between classifiers and feature
            if self.args.is_adversarial:
                # Maximize the discrepancy of classifiers
                _, output_t_list, _, _ = self.calc_output_list(img_t)
                cnt = 0
                loss_src = 0
                for idx in self.idx_source:
                    img_s, label_s = source_x[idx],source_y[idx]
                    _,output_list, _, _ = self.calc_output_list(img_s)
                    loss_temp = self.calc_source_loss(output_list, label_s)
                    loss_src = loss_temp + loss_src
                    cnt = cnt + 1
                loss_c = loss_src/cnt - self.args.trade_off*self.calc_classifier_discrepancy_loss(output_t_list)
                self.reset_grad(['C'])
                loss_c.backward(retain_graph=True)
                self.step_model(['C'])

                # Minimize the discrepancy of classifiers by training feature extractor
                for _ in range(self.args.num_k):
                    _, output_t_list, _, _ = self.calc_output_list(img_t)
                    loss_f = self.calc_classifier_discrepancy_loss(output_t_list)
                    self.reset_grad(['F'])
                    loss_f.backward()
                    self.step_model(['F'])


            if iter_num % self.eval_interval == 0:
                source_accuracy, target_accuracy = self.evaluate()
                print({'source accuracy': source_accuracy, 'target accuracy': target_accuracy})
                wandb.log({'1_1_loss_f_c': loss_f_c, #'1_2_loss_c':loss_c, '1_3_loss_f':loss_f,
                           '2_1_classifier_loss': classifier_loss.mean(),
                           '3_1_transfer_loss': transfer_loss.mean(),
                           '4_1_weight': weight.mean(),
                           '5_1_classifier_loss_adv_src': classifier_loss_adv_src.mean(),
                           '6_1_classifier_loss_adv_tgt': classifier_loss_adv_tgt.mean(),
                           '2_2_min_classifier_loss': classifier_loss.min(),
                           '2_3_max_classifier_loss': classifier_loss.max(),
                           '3_2_min_transfer_loss': transfer_loss.min(),
                           '3_3_max_transfer_loss': transfer_loss.max(),
                           '4_2_min_weight': weight.min(), '4_3_max_weight': weight.max(),
                           '5_2_min_classifier_loss_adv_src': classifier_loss_adv_src.min(),
                           '6_2_min_classifier_loss_adv_tgt': classifier_loss_adv_tgt.min(),
                           '5_3_max_classifier_loss_adv_src': classifier_loss_adv_src.max(),
                           '6_3_max_classifier_loss_adv_tgt': classifier_loss_adv_tgt.max(),
                           '7_source accuracy': source_accuracy, '8_target accuracy': target_accuracy,
                           })

    def evaluate(self):
        self.eval_model()
        correct_source, size_source = 0, 0
        for idx in self.idx_source:
            correct, size = self.calc_correct_and_size(idx)
            correct_source = correct_source + correct
            size_source = size_source + size

        correct_target, size_target = self.calc_correct_and_size(self.idx_target)
        source_accuracy = float(correct_source) / float(size_source)
        target_accuracy = float(correct_target) / float(size_target)

        return source_accuracy, target_accuracy

    def calc_output_list(self, img):
        feat_list = [None for _ in range(self.num_F)]
        output_list = [None for _ in range(self.num_C)]
        output_list_adv = [None for _ in range(self.num_C1)]

        num_C_for_F = int(self.num_C / self.num_F)
        num_C1_for_F = int(self.num_C1 / self.num_F)

        for r in range(len(self.net_dict['F'])):
            feat_list[r] = self.net_dict['F'][r](img)

        for r in range(len(self.net_dict['F'])):
            for c in range(num_C_for_F):
                output_list[r * num_C_for_F + c] = self.net_dict['C'][r * num_C_for_F + c](feat_list[r])
            for d in range(num_C1_for_F):
                output_list_adv[r * num_C1_for_F + d] = self.net_dict['C1'][r * num_C1_for_F + d](feat_list[r])

        output_list_mean = torch.mean(torch.stack(output_list, dim=0), dim=0)

        return feat_list, output_list, output_list_adv, output_list_mean

    def calc_source_loss(self, output_s_list, label_s):
        criterion = nn.CrossEntropyLoss().to(device)
        loss = criterion(output_s_list[0], label_s)
        for output_s in output_s_list[1:]:
            loss = loss + criterion(output_s, label_s)
        return loss
    
    def calc_single_loss(self, img_s, label_s, img_t, idx):
        _, output_s_list, output_s_list_adv, _ = self.calc_output_list(img_s)
        _, output_t_list, output_t_list_adv, _ = self.calc_output_list(img_t)

        # Single source loss in f
        classifier_loss = self.calc_source_loss(output_s_list, label_s)

        # Classification results of f
        target_adv_src = torch.mode(torch.stack(output_s_list).max(dim=-1)[1], dim=0)[0]
        target_adv_tar = torch.mode(torch.stack(output_t_list).max(dim=-1)[1], dim=0)[0]

        # Single source loss in f'
        log_loss_src = shift_log(F.softmax(output_s_list_adv[idx], dim=1))
        classifier_loss_adv_src = F.nll_loss(log_loss_src, target_adv_src)
        
        # Target loss in f'
        log_loss_tgt = shift_log(1 - F.softmax(output_t_list_adv[idx], dim=1))
        classifier_loss_adv_tgt = -1*F.nll_loss(log_loss_tgt, target_adv_tar)

        return classifier_loss, classifier_loss_adv_src, classifier_loss_adv_tgt

    def calc_classifier_discrepancy_loss(self, output_list):
        loss = 0.0
        num_C_for_F = int(self.num_C/self.num_F)
        for r in range(self.num_F):
            mean_output_t = torch.mean(
                torch.stack(output_list[num_C_for_F * r:num_C_for_F * (r + 1)]), dim=0)
            for c in range(num_C_for_F):
                loss = loss + discrepancy(output_list[num_C_for_F * r + c], mean_output_t)
        return loss
    
    def calc_correct_and_size(self, idx):
        img, label = self.data.get_test_samples(idx)
        img, label = Variable(torch.Tensor(img).to(device)), Variable(torch.Tensor(label).long().to(device))
        _, output_list, _, _ = self.calc_output_list(img)
        output_vec = torch.stack(output_list)
        pred_ensemble = output_vec.data.max(dim=-1)[1]
        pred_ensemble = torch.mode(pred_ensemble, dim=0)[0]
        size = label.data.size()[0]
        correct = pred_ensemble.eq(label.data).cpu().sum()
        return correct, size

    def save_model(self):
        for key in self.net_dict.keys():
            for i in range(len(self.net_dict[key])):
                torch.save(self.net_dict[key][i], '{}/{}_{}.pt'.format(
                    self.checkpoint_dir, key, i))

    def load_model(self):
        for key in self.net_dict.keys():
            for i in range(len(self.net_dict[key])):
                if 'cpu' in device_name:
                    self.net_dict[key][i] = torch.load('{}/{}_{}.pt'.format(
                        self.args.checkpoint_dir, key, i),
                        map_location=lambda storage, loc: storage)
                else:
                    file_name = '{}/{}_{}.pt'.format(
                        self.args.checkpoint_dir, key, i)
                    self.net_dict[key][i] = torch.load(file_name)
    
    def feature_output(self,img,label):
        self.eval_model()
        img= Variable(torch.Tensor(img).to(device))
        _,_,_,feature = self.calc_output_list(img)
        feature = feature.detach().cpu().numpy().reshape(len(label),-1)
        return feature

def shift_log(x, offset = 1e-6):
    return torch.log(torch.clamp(x + offset, max=1.))

def discrepancy(out1, out2):
    return torch.mean(torch.abs(F.softmax(out1, dim=-1) - F.softmax(out2, dim=-1)))