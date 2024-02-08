# Copyright 2022 CircuitNet. All rights reserved.

import argparse
import os
import sys

sys.path.append(os.getcwd())

class Parser(object):
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--task', default='Congestion')
        self.parser.add_argument('--save_path', default='work_dir/Congestion_Models/')
        self.parser.add_argument('--pretrained', default=None)
        self.parser.add_argument('--max_iters', default=200000)
        self.parser.add_argument('--plot_roc', action='store_true')
        self.parser.add_argument('--arg_file', default=None)
        self.parser.add_argument('--cpu', action='store_true')
        self.get_remainder()
        
    def get_remainder(self):


        if self.parser.parse_args().task == 'Congestion':
            base_path = "C:/Circuit_Net/congestion_DataSet/"
            self.parser.add_argument('--dataroot', default=base_path+ 'congestion')
            self.parser.add_argument('--ann_file_train', default='./files/train.csv')
            self.parser.add_argument('--ann_file_test', default='./files/test.csv')
            self.parser.add_argument('--dataset_type', default='CongestionDataset')
            self.parser.add_argument('--batch_size', default=16)
            self.parser.add_argument('--aug_pipeline', default=['Flip'])
            self.parser.add_argument('--model_type', default='Congestion_Prediction_Net')
            self.parser.add_argument('--in_channels', default=3)
            self.parser.add_argument('--out_channels', default=1)
            self.parser.add_argument('--lr', default=2e-4)
            ###self.parser.add_argument('--weight_decay', default=0)
            self.parser.add_argument('--weight_decay', default=1e-4) #===
            self.parser.add_argument('--loss_type', default='MSELoss')
            # self.parser.add_argument('--eval-metric', default=['NRMS', 'SSIM', 'EMD'])
            self.parser.add_argument('--eval-metric', default=['NRMS', 'SSIM'])

        elif self.parser.parse_args().task == 'DRC':
            base_path = "C:/Circuit_Net/DRC_DataSet/"
            self.parser.add_argument('--dataroot', default=base_path + 'DRC')
            self.parser.add_argument('--ann_file_train', default='./files/train.csv')
            self.parser.add_argument('--ann_file_test', default='./files/test.csv')
            self.parser.add_argument('--dataset_type', default='DRCDataset')
            self.parser.add_argument('--batch_size', default=16)
            self.parser.add_argument('--aug_pipeline', default=['Flip'])
            self.parser.add_argument('--model_type', default='DRC_Prediction_Net')
            self.parser.add_argument('--in_channels', default=9)
            self.parser.add_argument('--out_channels', default=1)
            self.parser.add_argument('--lr', default=2e-4)
            self.parser.add_argument('--weight_decay', default=1e-4)
            self.parser.add_argument('--loss_type', default='MSELoss')
            self.parser.add_argument('--eval-metric', default=['NRMS', 'SSIM'])
            self.parser.add_argument('--threshold', default=0.1)
        else:
            raise ValueError
