# Copyright 2022 CircuitNet. All rights reserved.

import os
import json
import torch
import torch.optim as optim
from tqdm import tqdm

from datasets.build_dataset import build_dataset
from utils.losses import build_loss
from models.build_model import build_model
from utils.configs import Parser
from math import cos, pi
from datetime import datetime
import sys, os, subprocess


def checkpoint(model, epoch, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_out_path = f"./{save_path}/model_iters_{epoch}.pth"
    torch.save({'state_dict': model.state_dict()}, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

    # Only export onnx
    #============================================================
    # ===========================================================
    # Very Important: Saving Model will cause Accuracy decray!
    # ===========================================================
    # ===========================================================

    if False:
        print("====> to save onnx mode....")
        real_model = model
        real_model.eval()
        # dummy_input = torch.randn(2, 1, 24, 256, 256).cuda() #for IR_drop
        dummy_input = torch.randn(16, 3, 256, 256).cuda() #for congestion
        # dummy_input = torch.randn(16, 9, 256, 256).cuda()  # for DRC

        save_path = f"./{save_path}/model_iters_{epoch}.onnx"
        torch.onnx.export(real_model, dummy_input, save_path) #, verbose=False, opset_version=12)
        print("====> to save onnx mode, well done!!!")
        real_model.train()

class CosineRestartLr(object):
    def __init__(self,
                 base_lr,
                 periods,
                 restart_weights = [1],
                 min_lr = None,
                 min_lr_ratio = None):
        self.periods = periods
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        self.restart_weights = restart_weights
        super().__init__()

        self.cumulative_periods = [
            sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))
        ]

        self.base_lr = base_lr

    def annealing_cos(self, start: float,
                    end: float,
                    factor: float,
                    weight: float = 1.) -> float:
        cos_out = cos(pi * factor) + 1
        return end + 0.5 * weight * (start - end) * cos_out

    def get_position_from_periods(self, iteration: int, cumulative_periods):
        for i, period in enumerate(cumulative_periods):
            if iteration < period:
                return i
        raise ValueError(f'Current iteration {iteration} exceeds '
                        f'cumulative_periods {cumulative_periods}')


    def get_lr(self, iter_num, base_lr: float):
        target_lr = self.min_lr  # type:ignore

        idx = self.get_position_from_periods(iter_num, self.cumulative_periods)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_periods[idx - 1]
        current_periods = self.periods[idx]

        alpha = min((iter_num - nearest_restart) / current_periods, 1)
        return self.annealing_cos(base_lr, target_lr, alpha, current_weight)

    
    def _set_lr(self, optimizer, lr_groups):
        if isinstance(optimizer, dict):
            for k, optim in optimizer.items():
                for param_group, lr in zip(optim.param_groups, lr_groups[k]):
                    param_group['lr'] = lr
        else:
            for param_group, lr in zip(optimizer.param_groups,
                                        lr_groups):
                param_group['lr'] = lr

    def get_regular_lr(self, iter_num):
        return [self.get_lr(iter_num, _base_lr) for _base_lr in self.base_lr]  # iters

    def set_init_lr(self, optimizer):
        for group in optimizer.param_groups:  # type: ignore
            group.setdefault('initial_lr', group['lr'])
            self.base_lr = [group['initial_lr'] for group in optimizer.param_groups  # type: ignore
        ]


def train():
    argp = Parser()
    arg = argp.parser.parse_args()
    arg_dict = vars(arg)
    if arg.arg_file is not None:
        with open(arg.arg_file, 'rt') as f:
            arg_dict.update(json.load(f))

    if not os.path.exists(arg_dict['save_path']):
        os.makedirs(arg_dict['save_path'])
    with open(os.path.join(arg_dict['save_path'],  'arg.json'), 'wt') as f:
      json.dump(arg_dict, f, indent=4)

    arg_dict['ann_file'] = arg_dict['ann_file_train']
    arg_dict['test_mode'] = False


    now = datetime.now()
    print('===> Loading datasets')
    # Initialize dataset
    dataset = build_dataset(arg_dict)
    later = datetime.now()
    difference = int((later - now).total_seconds())
    now = datetime.now()
    print('===> Loading datasets takes: '+str(difference)+"(seconds)")

    print('===> Building model')
    # Initialize model parameters
    model = build_model(arg_dict)
    if not arg_dict['cpu']:
        model = model.cuda()
    
    # Build loss
    loss = build_loss(arg_dict)

    # Build Optimzer
    optimizer = optim.AdamW(model.parameters(), lr=arg_dict['lr'],  betas=(0.9, 0.999), weight_decay=arg_dict['weight_decay'])

    # Try Harley 2023/11/17
    # optimizer = optim.SGD(model.parameters(), lr=arg_dict['lr'], momentum=0.9, weight_decay=arg_dict['weight_decay'])

    # Build lr scheduler
    cosine_lr = CosineRestartLr(arg_dict['lr'], [arg_dict['max_iters']], [1], 1e-7)
    cosine_lr.set_init_lr(optimizer)

    epoch_loss = 0
    iter_num = 0
    print_freq = 100
    save_freq = 10000
    Show_freq = 1000

    lossList =[]

    while iter_num < arg_dict['max_iters']:
        with tqdm(total=print_freq) as bar:
            for feature, label, _ in dataset:        
                if arg_dict['cpu']:
                    input, target = feature, label
                else:
                    input, target = feature.cuda(), label.cuda()

                regular_lr = cosine_lr.get_regular_lr(iter_num)
                cosine_lr._set_lr(optimizer, regular_lr)

                prediction = model(input)

                optimizer.zero_grad()

                pixel_loss = loss(prediction, target)
                epoch_loss += pixel_loss.item()
                pixel_loss.backward()
                optimizer.step()

                iter_num += 1
                
                bar.update(1)

                if (iter_num % print_freq == 0 or iter_num==100):
                    break

        print("===> Iters({}/{}): Loss: {:.4f}".format(iter_num, arg_dict['max_iters'], epoch_loss / print_freq))
        oneValue = epoch_loss /print_freq
        lossList.append(oneValue)
        if(len(lossList)>10):
            lossList.pop(0)
            sumValue =0
            for kk in range(len(lossList)):
                sumValue += lossList[kk]
            print("===> Average Loss: {:.4f}".format(sumValue / len(lossList)))

        if ((iter_num % save_freq == 0) or (100==iter_num)):
            checkpoint(model, iter_num, arg_dict['save_path'])

        if ((iter_num % Show_freq == 0) or (100==iter_num)):
            later = datetime.now()
            difference = (later - now).total_seconds()
            difference_minutes = int(difference // 60)
            difference_seconds = int(difference % 60)
            print('===> So far taining takes: '
                  + str(difference_minutes) + "(minutes) and "
                  + str(difference_seconds) + "(seconds)")

            max_iters = arg_dict['max_iters']
            expect_difference = difference*(max_iters/iter_num)
            difference_minutes = int(expect_difference // 60)
            difference_seconds = int(expect_difference % 60)
            print('===> Expected Time: '
                  + str(difference_minutes) + "(minutes) and "
                  + str(difference_seconds) + "(seconds)")

        epoch_loss = 0



if __name__ == "__main__":

    '''
    --task Congestion --save_path ./Congestion_Models/
    --task DRC --save_path ./DRC_Models/
    '''
    train()
