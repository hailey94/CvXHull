import os
import copy
import random
import argparse
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

import data_loader
import utility
import models
import tool
import coverage
import constants
from itertools import product
import numpy as np

class Model(nn.Module):
    def __init__(self, in_channels=1, name=None):
        super(Model, self).__init__()    
        # change the name of pre-trained model based on your selection
        if name == 'resnet18':
            self.net = torchvision.models.resnet18(pretrained=False)
        elif name == 'wide_resnet50_2':
            self.net = torchvision.models.wide_resnet50_2(pretrained=False)
            
        self.net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = self.net.fc.in_features
        self.net.fc = nn.Linear(num_ftrs, 10)

    def forward(self, x):
        return self.net(x)
    
parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, default='./test_folder')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--num_per_class', type=float, default=5000)
parser.add_argument('--hyper', type=float, default=None)
args = parser.parse_args()
# 

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
modelinfo = ['lenet']#,'resnet18', 'wide_resnet50_2']
datas = [ 'MNISTC'] #'MNIST', 'SVHN', 'FMNIST','FMNISTC', 
criterions = ['NLC', 'NBC', 'LSA2', 'DSA2','FDplus2']
for data, model in product(datas, modelinfo):
    args.dataset = data
    args.model = model

    if args.dataset == 'SVHN':
        args.image_size = 32
        args.num_class = 10
        args.nc = 3
    elif args.dataset == 'FMNIST':
        args.image_size = 28
        args.num_class = 10
        args.nc = 1
    elif args.dataset == 'MNIST':
        args.image_size = 28
        args.num_class = 10
        args.nc = 1
        if args.model=='wide_resnet50_2':
            continue
    elif args.dataset == 'FMNISTC':
        args.image_size = 28
        args.num_class = 10
        args.nc = 1
    elif args.dataset == 'MNISTC':
        args.image_size = 28
        args.num_class = 10
        args.nc = 1
        if args.model=='wide_resnet50_2':
            continue
    
    if args.model == 'lenet':
        model = getattr(models, args.model)(pretrained=False)
    else:
        model = Model(in_channels=args.nc, name=args.model).to(DEVICE)
    
    if args.dataset == 'FMNISTC':
        path = os.path.join(constants.PRETRAINED_MODELS, ('FMNIST/%s.pt' % (args.model)))
    elif args.dataset == 'MNISTC':
        path = os.path.join(constants.PRETRAINED_MODELS, ('MNIST/%s.pt' % (args.model)))
    else:
        path = os.path.join(constants.PRETRAINED_MODELS, ('%s/%s.pt' % (args.dataset, args.model)))
    model.load_state_dict(torch.load(path))
    model.to(DEVICE)
    model.eval()

    input_size = (1, args.nc, args.image_size, args.image_size)
    random_data = torch.randn(input_size).to(DEVICE)
    layer_size_dict = tool.get_layer_output_sizes(model, random_data)

    num_neuron = 0
    for layer_name in layer_size_dict.keys():
        num_neuron += layer_size_dict[layer_name][0]

    print(args.model)
    print('Total %d layers: ' % len(layer_size_dict.keys()))
    print('Total %d neurons: ' % num_neuron)

    TOTAL_CLASS_NUM, train_loader, test_loader, seed_loader = data_loader.get_loader(args)
    
    for criterion in criterions:
        args.criterion = criterion
        args.exp_name = ('%s-%s-%s-%s' % (args.dataset, args.model, args.criterion, args.hyper))
        USE_SA = args.criterion in ['LSA2', 'DSA2']

        if USE_SA:
            criterion = getattr(coverage, args.criterion)(model, layer_size_dict, min_var=1e-5, num_class=TOTAL_CLASS_NUM)
        elif args.criterion == 'FDplus2':
            criterion = getattr(coverage, args.criterion)(model, layer_size_dict, alpha=0.0499, eps=0.95, num_class=TOTAL_CLASS_NUM)
        else:
            criterion = getattr(coverage, args.criterion)(model, layer_size_dict, hyper=args.hyper)

        criterion.build(train_loader)

        criterion.per_sample(test_loader)
        perSample = pd.DataFrame(criterion.sample_cov, columns=['{}'.format(args.criterion)])
        ids = pd.DataFrame(criterion.ids, columns=['ids'])

        dict_sample = pd.DataFrame(pd.concat([ids, perSample], axis=1))
        dict_sample['total_build(sec)'] = criterion.build_time
        dict_sample['total_inference(sec)'] = criterion.inference_time
        dict_sample.to_csv('%s/%s.csv' % (args.output_dir, args.exp_name), index=False)

        labels = np.expand_dims(criterion.collect_label(test_loader), axis=1)
        predictions = np.expand_dims(criterion.collect_predictions(test_loader), axis=1)

        df = pd.DataFrame(np.concatenate([labels, predictions], axis=1), columns=['Ground Truth', 'Predicted class'])
        df = pd.DataFrame(pd.concat([df, perSample], axis=1))

        no_buggy_instances = (df['Ground Truth'] != df['Predicted class']).sum()

        # DeepGini
        if 'FDplus' in args.criterion:
            df.sort_values(by=['{}'.format(args.criterion)], ascending=True, inplace=True)
        else:
            df.sort_values(by=['{}'.format(args.criterion)], ascending=False, inplace=True)

        gini = []
        for k in tqdm(range(no_buggy_instances)):
            bdgt = (k+1)
            TPF = sum(df.iloc[:bdgt,0] != df.iloc[:bdgt,1])/bdgt # min = bdgt
            gini.append(TPF)

        print('{} Eval: '.format(args.criterion), round(100*sum(gini)/no_buggy_instances,3))