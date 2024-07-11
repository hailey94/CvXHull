import torch
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import time
import os

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from functools import reduce
from tqdm import tqdm
from dch import DCH

def multiply(arr):
    return reduce(lambda x, y: x * y, arr)

def duplication_check(train_img, test_img):
    dup_list = []
    for idx, elt1 in tqdm(enumerate(test_img)):
        sames = torch.sum((elt1 == train_img), dim=[i for i in range(len(train_img.shape)-1, 0, -1)])
        max_idx, max_value = torch.argmax(sames), sames[torch.argmax(sames)]
        shape_rev = [i for i in reversed(train_img.shape)][:-1]
        resolution = multiply(shape_rev)
        if max_value == resolution:
            dup_list.append(idx)
    return dup_list

def delete_duplicates(arr, indices):
    mask = torch.ones(arr.shape[0], dtype=torch.bool)
    mask[indices] = False
    return arr[mask]

def calculate_epsilon(train_img, test_img):
    closest = []
    for idx, elt1 in tqdm(enumerate(test_img)):
        pairwise_dist = torch.cdist(train_img, torch.unsqueeze(elt1, dim=0), p=2)
        pairwise_dist = pairwise_dist[pairwise_dist != 0.0]
        closest.append(torch.min(pairwise_dist).item())
        del elt1, pairwise_dist 

    closest = np.array(closest)
    return np.min(closest), np.mean(closest), np.max(closest)


class PerLabelDataset(Dataset):
    def __init__(self, targets, imgs, transform=None, target_transform=None):
        self.targets = targets
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        image = self.imgs[idx]
        label = self.targets[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def per_class_dch(args, train, test, device, save_dir, max_sparsity=20, stopping_criterion = 1, transform_=None, feature_level=False): # 1 = max, 2 = mean
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        
    if isinstance(train, list):
        train_imgs, train_labels = train[0].detach().cpu().numpy(), train[1].detach().cpu().numpy()
    else:
        train_imgs, train_labels = train.data, np.array(train.targets)
    
    if isinstance(test, list):
        test_imgs, test_labels = test[0].detach().cpu().numpy(), test[1].detach().cpu().numpy()
    else:
        test_imgs, test_labels = test.data, np.array(test.targets)

    labels = np.unique(train_labels)

    for label in labels:
        label_ids = np.where(train_labels == label)[0]
        train_label_imgs, train_label_targets = train_imgs[label_ids], train_labels[label_ids]

        label_ids = np.where(test_labels == label)[0]
        test_label_imgs, test_label_targets = test_imgs[label_ids], test_labels[label_ids]
        
        try:
            train_label_imgs = torch.from_numpy(train_label_imgs).to(device)
            test_label_imgs = torch.from_numpy(test_label_imgs).to(device)
        except TypeError:
            train_label_imgs = train_label_imgs.to(device)
            test_label_imgs = test_label_imgs.to(device)

        duplication_idx_in_test = duplication_check(train_label_imgs, test_label_imgs)
        if len(duplication_idx_in_test) > 0:
            test_label_imgs, test_label_targets = delete_duplicates(test_label_imgs, duplication_idx_in_test), delete_duplicates(test_label_targets, duplication_idx_in_test)
            print('\t(Optional). After duplicate removal')
            print('\tShape of Train & Test',test_label_imgs.shape, test_label_targets.shape)

        train = PerLabelDataset(train_label_targets, train_label_imgs.detach().cpu().numpy(), transform=transform_)
        test = PerLabelDataset(test_label_targets, test_label_imgs.detach().cpu().numpy(), transform=transform_)

        train_loader = DataLoader(train, shuffle=False, num_workers=2, batch_size=train_label_targets.shape[0])
        test_loader = DataLoader(test, shuffle=False, num_workers=2, batch_size=test_label_targets.shape[0])
        
        for idx, ((train_label_imgs, train_label_targets), (test_label_imgs, test_label_targets)) in enumerate(zip(train_loader, test_loader)):
            train_label_imgs = train_label_imgs.reshape((train_label_imgs.shape[0],-1)).to(device)
            test_label_imgs = test_label_imgs.reshape((test_label_imgs.shape[0],-1)).to(device)
            
            if idx == 0:
                tmp1 = test_label_imgs
                tmp2 = train_label_imgs
                
            else:
                tmp1 = torch.cat([tmp1, test_label_imgs], dim=0)
                tmp2 = torch.cat([tmp2, train_label_imgs], dim=0)
                
        test_label_imgs = tmp1
        train_label_imgs = tmp2

        train_label_imgs = train_label_imgs / torch.norm(train_label_imgs , dim=1).unsqueeze(1)
        test_label_imgs = test_label_imgs / torch.norm(test_label_imgs , dim=1).unsqueeze(1)
            
        epsilon = calculate_epsilon(args.d, train_label_imgs, test_label_imgs)
        del test_label_imgs
        print('======> Starting Calculate Train Set Convex hull of Label {}'.format(label))
        print('epsilon = ', epsilon)
        approx = DCH(train_label_imgs, epsilon, max_sparsity, stopping_criterion, cuda=torch.cuda.is_available())
        c_hull_t = approx.calculate_chull()
        sol_U = approx.U.t()
        print('sol_U combined shape =', sol_U.shape)
        if feature_level:
            np.save(os.path.join(save_dir, 'feature-{}-{}.npy'.format(int(label), epsilon)), sol_U.detach().cpu().numpy())
        else:
            np.save(os.path.join(save_dir, '{}-{}.npy'.format(int(label), epsilon)), sol_U.detach().cpu().numpy())
        torch.cuda.empty_cache()


    
def combined_dch(args, frac_dir, test, d, device, save_dir, max_sparsity=20, stopping_criterion = 1, transform_=None, feature_level=False): # 1 = max, 2 = mean
    
    if feature_level:
        file_list = [i for i in os.listdir(frac_dir) if ('.npy' in i) and ('feature' in i)]
    else:
        file_list = [i for i in os.listdir(frac_dir) if ('.npy' in i) and ('feature' not in i)]
    
    if isinstance(test, list):
        test_imgs, test_labels = test[0].detach().cpu().numpy(), test[1].detach().cpu().numpy()
    else:
        test_imgs, test_labels = test.data, np.array(test.targets)

    labels = np.unique(test_labels)
    cvx_hull = np.zeros([1, d])
    
    for file in file_list:
        cvx_hull = np.concatenate([cvx_hull, np.load(os.path.join(save_dir, file))], axis=0)
    
    cvx_hull = cvx_hull[1:]
    
    try:
        cvx_hull = torch.from_numpy(cvx_hull).to(device)
        test_imgs = torch.from_numpy(test_imgs).to(device)            
    except TypeError:
        cvx_hull = cvx_hull.to(device)
        test_imgs = test_imgs.to(device)            
    
    print('cvx hull shape = ', cvx_hull.shape, 'test img shape = ', test_imgs.shape)
    duplication_idx_in_test = duplication_check(cvx_hull, test_imgs.reshape((test_imgs.shape[0],-1)))

    if len(duplication_idx_in_test) > 0:
        test_imgs, test_labels= delete_duplicates(test_imgs, duplication_idx_in_test), delete_duplicates(test_labels, duplication_idx_in_test)
        print('\t(Optional). After duplicate removal')
        print('\tShape of Train & Test',cvx_hull.shape, test_imgs.shape, test_labels.shape)
    
    test = PerLabelDataset(test_labels, test_imgs.detach().cpu().numpy(), transform=transform_)
    test_loader = DataLoader(test, shuffle=False, num_workers=2, batch_size=test_labels.shape[0])
    
    for idx, (test_batch_imgs, test_batch_targets) in enumerate(test_loader):
        test_batch_imgs = test_batch_imgs.reshape((test_batch_imgs.shape[0],-1)).to(device)        
        if idx == 0:
            tmp = test_batch_imgs
        else:
            tmp = torch.cat([tmp, test_batch_imgs], dim=0)

    test_imgs = tmp
    test_imgs = test_imgs / torch.norm(test_imgs , dim=1).unsqueeze(1)

    epsilon = calculate_epsilon(args.d, cvx_hull, test_imgs.type(torch.DoubleTensor).to(device))
    del test_imgs, test_labels, test
    print('======> Starting Calculate Combined Train Set Convex hull')
    print('epsilon = ', epsilon)
    
    cvx_hull = cvx_hull.type(torch.FloatTensor).to(device)
    approx = DCH(cvx_hull, epsilon, max_sparsity, stopping_criterion, cuda=torch.cuda.is_available())
    print('cvx_hull.shape',cvx_hull.shape)
    
    c_hull_t = approx.calculate_chull()
    sol_U = approx.U.t()
    print('sol_U shape =', sol_U.shape)
    if feature_level:
        np.save(os.path.join(save_dir, 'Combined-feature-{}.npy'.format(epsilon)), sol_U.detach().cpu().numpy())
    else:
        np.save(os.path.join(save_dir, 'Combined-{}.npy'.format(epsilon)), sol_U.detach().cpu().numpy())
    torch.cuda.empty_cache()

    

    
def combined_augment_dch(data_name, file_list, test, device, save_dir, max_sparsity=20, stopping_criterion = 1, transform_=None, feature_level=False): # 1 = max, 2 = mean
    
    for i, file in enumerate(file_list):
        if i == 0:
            cvx_hull = file
        else:
            cvx_hull = np.concatenate([cvx_hull, file], axis=0)
    
    cvx_hull = torch.from_numpy(cvx_hull).to(device)
    
    if isinstance(test, list):
        test_imgs, test_labels = test[0].detach().cpu().numpy(), test[1].detach().cpu().numpy()
    else:
        test_imgs, test_labels = test.data, np.array(test.targets)
 
    test_imgs = torch.from_numpy(test_imgs).to(device)
    
    print('cvx hull shape = ', cvx_hull.shape, 'test img shape = ', test_imgs.shape)
    duplication_idx_in_test = duplication_check(cvx_hull, test_imgs.reshape((test_imgs.shape[0],-1)))

    if len(duplication_idx_in_test) > 0:
        test_imgs, test_labels= delete_duplicates(test_imgs, duplication_idx_in_test), delete_duplicates(test_labels, duplication_idx_in_test)
        print('\t(Optional). After duplicate removal')
        print('\tShape of Train & Test',cvx_hull.shape, test_imgs.shape, test_labels.shape)
    
    test = PerLabelDataset(test_labels, test_imgs.detach().cpu().numpy(), transform=transform_)
    test_loader = DataLoader(test, shuffle=False, num_workers=2, batch_size=test_labels.shape[0])
    
    for idx, (test_batch_imgs, test_batch_targets) in enumerate(test_loader):
        test_batch_imgs = test_batch_imgs.reshape((test_batch_imgs.shape[0],-1)).to(device)
        
        test_batch_imgs = test_batch_imgs / torch.norm(test_batch_imgs , dim=1).unsqueeze(1)
        if idx == 0:
            tmp = test_batch_imgs
        else:
            tmp = torch.cat([tmp, test_batch_imgs], dim=0)

    test_imgs = tmp
    epsilon = calculate_epsilon(data_name, cvx_hull, test_imgs.type(torch.DoubleTensor).to(device))
    del test_imgs, test_labels, test
    print('======> Starting Calculate Combined Train Set Convex hull')
    print('epsilon = ', epsilon)
    
    cvx_hull = cvx_hull.type(torch.FloatTensor).to(device)
    approx = DCH(cvx_hull, epsilon, max_sparsity, stopping_criterion, cuda=torch.cuda.is_available())
    print('cvx_hull.shape',cvx_hull.shape)
    
    c_hull_t = approx.calculate_chull()
    sol_U = approx.U.t()
    print('sol_U shape =', sol_U.shape)
    if feature_level:
        np.save(os.path.join(save_dir, 'Augmented-feature-{}.npy'.format(epsilon)), sol_U.detach().cpu().numpy())
    else:
        np.save(os.path.join(save_dir, 'Augmented-{}.npy'.format(epsilon)), sol_U.detach().cpu().numpy())
    torch.cuda.empty_cache()
    
            
def closure_matrix_calc(test, class_names, args, device, file_dir, save_dir='./', transform_=None, feature_level=False):
    if feature_level:
        file_list = [i for i in os.listdir(file_dir) if ('.npy' in i) and ('feature' in i)]
    else:
        file_list = [i for i in os.listdir(file_dir) if ('.npy' in i) and ('feature' not in i)]

    closure_matrix = np.zeros([len(file_list), len(file_list)])
    
    if isinstance(test, list):
        test_imgs, test_labels = test[0].detach().cpu().numpy(), test[1].detach().cpu().numpy()
    else:
        test_imgs, test_labels = test.data, np.array(test.targets)

    for file in file_list:
        label = int(file.split('-')[-2])
        epsilon = float(file.split('-')[-1].split('.npy')[0])
        tag = class_names[label]

        cvx_hull = np.load(os.path.join(save_dir, file))
        cvx_hull = torch.from_numpy(cvx_hull).to(device)
        for comp_label, comp_tag in enumerate(class_names):

            label_ids = np.where(test_labels == comp_label)[0]

            test_d = PerLabelDataset(test_labels[label_ids], test_imgs[label_ids], transform=transform_)
            test_loader = DataLoader(test_d, shuffle=False, num_workers=2, batch_size=test_labels[label_ids].shape[0])
            assert len(test_loader) == 1

            for (test_label_imgs, test_label_targets) in test_loader:
                test_label_imgs = test_label_imgs.reshape((test_label_imgs.shape[0],-1)).to(device)
                test_label_imgs = test_label_imgs / torch.norm(test_label_imgs , dim=1).unsqueeze(1)
        
                approx = DCH(cvx_hull.t(), epsilon, args.max_sparsity, args.stopping_criterion, cuda=torch.cuda.is_available())
                D,_ , _ = approx.compute_dist_to_chull(cvx_hull.t(), test_label_imgs.t(), 0)
                out_nums = D[D > epsilon].shape[0]
                closure_ratio = 1 - (out_nums/test_label_imgs.shape[0])
                closure_matrix[label, comp_label] = closure_ratio
                torch.cuda.empty_cache()

    closure_matrix = pd.DataFrame(closure_matrix)
    closure_matrix.columns = class_names
    closure_matrix.index = class_names
    if feature_level:
        closure_matrix.to_csv(os.path.join(save_dir, 'ClosureRatio-feature-{}.csv'.format(args.d)))
    else:
        closure_matrix.to_csv(os.path.join(save_dir, 'ClosureRatio-{}.csv'.format(args.d)))
    
    return closure_matrix


def per_class_dch_woNorm(args, train, test, device, save_dir, max_sparsity=20, stopping_criterion = 1, transform_=None, feature_level=False): # 1 = max, 2 = mean
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        
    if isinstance(train, list):
        train_imgs, train_labels = train[0].detach().cpu().numpy(), train[1].detach().cpu().numpy()
    else:
        train_imgs, train_labels = train.data, np.array(train.targets)
    
    if isinstance(test, list):
        test_imgs, test_labels = test[0].detach().cpu().numpy(), test[1].detach().cpu().numpy()
    else:
        test_imgs, test_labels = test.data, np.array(test.targets)

    labels = np.unique(train_labels)
    
    train_setup = []
    for label in labels:
        label_ids = np.where(train_labels == label)[0]
        train_label_imgs, train_label_targets = train_imgs[label_ids], train_labels[label_ids]

        label_ids = np.where(test_labels == label)[0]
        test_label_imgs, test_label_targets = test_imgs[label_ids], test_labels[label_ids]
        
        try:
            train_label_imgs = torch.from_numpy(train_label_imgs).to(device)
            test_label_imgs = torch.from_numpy(test_label_imgs).to(device)
        except TypeError:
            train_label_imgs = train_label_imgs.to(device)
            test_label_imgs = test_label_imgs.to(device)

        duplication_idx_in_test = duplication_check(train_label_imgs, test_label_imgs)
        if len(duplication_idx_in_test) > 0:
            test_label_imgs, test_label_targets = delete_duplicates(test_label_imgs, duplication_idx_in_test), delete_duplicates(test_label_targets, duplication_idx_in_test)
            print('\t(Optional). After duplicate removal')
            print('\tShape of Train & Test',train_label_imgs.shape, train_label_targets.shape, test_label_imgs.shape, test_label_targets.shape)

        train = PerLabelDataset(train_label_targets, train_label_imgs.detach().cpu().numpy(), transform=transform_)
        test = PerLabelDataset(test_label_targets, test_label_imgs.detach().cpu().numpy(), transform=transform_)

        train_loader = DataLoader(train, shuffle=False, num_workers=2, batch_size=train_label_targets.shape[0])
        test_loader = DataLoader(test, shuffle=False, num_workers=2, batch_size=test_label_targets.shape[0])

        for idx, ((train_label_imgs, train_label_targets), (test_label_imgs, test_label_targets)) in enumerate(zip(train_loader, test_loader)):
            train_label_imgs = train_label_imgs.reshape((train_label_imgs.shape[0],-1)).to(device)
            test_label_imgs = test_label_imgs.reshape((test_label_imgs.shape[0],-1)).to(device)
            
            if idx == 0:
                tmp1 = test_label_imgs
                tmp2 = train_label_imgs
                
            else:
                tmp1 = torch.cat([tmp1, test_label_imgs], dim=0)
                tmp2 = torch.cat([tmp2, train_label_imgs], dim=0)
                
        test_label_imgs = tmp1
        train_label_imgs = tmp2
            
        train_label_imgs = (train_label_imgs - torch.min(train_label_imgs)) / (torch.max(train_label_imgs) - torch.min(train_label_imgs))
        test_label_imgs = (test_label_imgs - torch.min(train_label_imgs)) / (torch.max(train_label_imgs) - torch.min(train_label_imgs))
        
        train_setup.append([torch.min(train_label_imgs), torch.max(train_label_imgs)])
        
        epsilon = calculate_epsilon(args.d, train_label_imgs, test_label_imgs)
        del test_label_imgs
        print('======> Starting Calculate Train Set Convex hull of Label {}'.format(label))
        print('epsilon = ', epsilon)
        approx = DCH(train_label_imgs, epsilon, max_sparsity, stopping_criterion, cuda=torch.cuda.is_available())
        c_hull_t = approx.calculate_chull()
        sol_U = approx.U.t()
        print('sol_U combined shape =', sol_U.shape)
        if feature_level:
            np.save(os.path.join(save_dir, 'feature-{}-{}.npy'.format(int(label), epsilon)), sol_U.detach().cpu().numpy())
        else:
            np.save(os.path.join(save_dir, '{}-{}.npy'.format(int(label), epsilon)), sol_U.detach().cpu().numpy())
        torch.cuda.empty_cache()
    
    train_mins, train_maxs = [i[0] for i in train_setup], [i[1] for i in train_setup]
    return train_mins, train_maxs
    
def combined_dch_woNorm(args, frac_dir, train_min, train_max, test, d, device, save_dir, max_sparsity=20, stopping_criterion = 1, transform_=None, feature_level=False): # 1 = max, 2 = mean
    
    if feature_level:
        file_list = [i for i in os.listdir(frac_dir) if ('.npy' in i) and ('feature' in i)]
    else:
        file_list = [i for i in os.listdir(frac_dir) if ('.npy' in i) and ('feature' not in i)]
    
    if isinstance(test, list):
        test_imgs, test_labels = test[0].detach().cpu().numpy(), test[1].detach().cpu().numpy()
    else:
        test_imgs, test_labels = test.data, np.array(test.targets)

    labels = np.unique(test_labels)
    cvx_hull = np.zeros([1, d])
    
    print(file_list)
    for file in file_list:
        cvx_hull = np.concatenate([cvx_hull, np.load(os.path.join(save_dir, file))], axis=0)
    
    cvx_hull = cvx_hull[1:]
    
    try:
        cvx_hull = torch.from_numpy(cvx_hull).to(device)
        test_imgs = torch.from_numpy(test_imgs).to(device)            
    except TypeError:
        cvx_hull = cvx_hull.to(device)
        test_imgs = test_imgs.to(device)            
    
    print('cvx hull shape = ', cvx_hull.shape, 'test img shape = ', test_imgs.shape)
    duplication_idx_in_test = duplication_check(cvx_hull, test_imgs.reshape((test_imgs.shape[0],-1)))

    if len(duplication_idx_in_test) > 0:
        test_imgs, test_labels= delete_duplicates(test_imgs, duplication_idx_in_test), delete_duplicates(test_labels, duplication_idx_in_test)
        print('\t(Optional). After duplicate removal')
        print('\tShape of Train & Test',cvx_hull.shape, test_imgs.shape, test_labels.shape)
    
    test = PerLabelDataset(test_labels, test_imgs.detach().cpu().numpy(), transform=transform_)
    test_loader = DataLoader(test, shuffle=False, num_workers=2, batch_size=test_labels.shape[0])
    
    for idx, (test_batch_imgs, test_batch_targets) in enumerate(test_loader):
        test_batch_imgs = test_batch_imgs.reshape((test_batch_imgs.shape[0],-1)).to(device)
        if idx == 0:
            tmp = test_batch_imgs
        else:
            tmp = torch.cat([tmp, test_batch_imgs], dim=0)

    test_imgs = tmp
    test_imgs = (test_imgs - train_min) / (train_max - train_min)
    
    epsilon = calculate_epsilon(args.d, cvx_hull, test_imgs.type(torch.DoubleTensor).to(device))
    del test_imgs, test_labels, test
    print('======> Starting Calculate Combined Train Set Convex hull')
    print('epsilon = ', epsilon)
    
    cvx_hull = cvx_hull.type(torch.FloatTensor).to(device)
    approx = DCH(cvx_hull, epsilon, max_sparsity, stopping_criterion, cuda=torch.cuda.is_available())
    print('cvx_hull.shape',cvx_hull.shape)
    
    c_hull_t = approx.calculate_chull()
    sol_U = approx.U.t()
    print('sol_U shape =', sol_U.shape)
    if feature_level:
        np.save(os.path.join(save_dir, 'Combined-feature-{}.npy'.format(epsilon)), sol_U.detach().cpu().numpy())
    else:
        np.save(os.path.join(save_dir, 'Combined-{}.npy'.format(epsilon)), sol_U.detach().cpu().numpy())
    torch.cuda.empty_cache()

    
    
def per_class_dch_woNorm2(args, train, test, device, save_dir, max_sparsity=20, stopping_criterion = 1, feature_level=False): # 1 = max, 2 = mean
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        
    if isinstance(train, list):
        train_imgs, train_labels = train[0], train[1]
    else:
        train_imgs, train_labels = train.data, np.array(train.targets)
    
    if isinstance(test, list):
        test_imgs, test_labels = test[0], test[1]
    else:
        test_imgs, test_labels = test.data, np.array(test.targets)

    labels = torch.unique(train_labels)
    
    train_setup = []
    for label in labels:
        train_label_imgs, train_label_targets = train_imgs[train_labels == label], train_labels[train_labels == label]
        test_label_imgs, test_label_targets = test_imgs[test_labels == label], test_labels[test_labels == label]
        
        
        duplication_idx_in_test = duplication_check(train_label_imgs, test_label_imgs)
        if len(duplication_idx_in_test) > 0:
            test_label_imgs, test_label_targets = delete_duplicates(test_label_imgs, duplication_idx_in_test), delete_duplicates(test_label_targets, duplication_idx_in_test)
            print('\t(Optional). After duplicate removal')
            print('\tShape of Train & Test',train_label_imgs.shape, train_label_targets.shape, test_label_imgs.shape, test_label_targets.shape)
        
        epsilon = calculate_epsilon(args.d, train_label_imgs, test_label_imgs)
        del test_label_imgs
        print('======> Starting Calculate Train Set Convex hull of Label {}'.format(label))
        print('epsilon = ', epsilon)
        approx = DCH(train_label_imgs, epsilon, max_sparsity, stopping_criterion, cuda=torch.cuda.is_available())
        c_hull_t = approx.calculate_chull()
        sol_U = approx.U.t()
        print('sol_U combined shape =', sol_U.shape)
        if feature_level:
            np.save(os.path.join(save_dir, 'feature-{}-{}.npy'.format(int(label), epsilon)), sol_U.detach().cpu().numpy())
        else:
            np.save(os.path.join(save_dir, '{}-{}.npy'.format(int(label), epsilon)), sol_U.detach().cpu().numpy())
        torch.cuda.empty_cache()
    
    
def combined_dch_woNorm2(args, frac_dir, test, d, device, save_dir, max_sparsity=20, stopping_criterion = 1, feature_level=False): # 1 = max, 2 = mean
    
    if feature_level:
        file_list = [i for i in os.listdir(frac_dir) if ('.npy' in i) and ('feature' in i)]
    else:
        file_list = [i for i in os.listdir(frac_dir) if ('.npy' in i) and ('feature' not in i)]
    
    if isinstance(test, list):
        test_imgs, test_labels = test[0], test[1]
    else:
        test_imgs, test_labels = test.data, np.array(test.targets)

    cvx_hull = np.zeros([1, d])
    
    for file in file_list:
        cvx_hull = np.concatenate([cvx_hull, np.load(os.path.join(save_dir, file))], axis=0)
    
    cvx_hull = cvx_hull[1:]
    
    try:
        cvx_hull = torch.from_numpy(cvx_hull).to(device)
    except TypeError:
        cvx_hull = cvx_hull.to(device)
    
    print('cvx hull shape = ', cvx_hull.shape, 'test img shape = ', test_imgs.shape)
    duplication_idx_in_test = duplication_check(cvx_hull, test_imgs.reshape((test_imgs.shape[0],-1)))

    if len(duplication_idx_in_test) > 0:
        test_imgs, test_labels= delete_duplicates(test_imgs, duplication_idx_in_test), delete_duplicates(test_labels, duplication_idx_in_test)
        print('\t(Optional). After duplicate removal')
        print('\tShape of Train & Test',cvx_hull.shape, test_imgs.shape, test_labels.shape)
    
    epsilon = calculate_epsilon(args.d, cvx_hull, test_imgs.type(torch.DoubleTensor).to(device)) #
    del test_imgs, test_labels, test
    print('======> Starting Calculate Combined Train Set Convex hull')
    print('epsilon = ', epsilon)
    
    cvx_hull = cvx_hull.type(torch.FloatTensor).to(device)
    approx = DCH(cvx_hull, epsilon, max_sparsity, stopping_criterion, cuda=torch.cuda.is_available())
    print('cvx_hull.shape',cvx_hull.shape)
    
    c_hull_t = approx.calculate_chull()
    sol_U = approx.U.t()
    print('sol_U shape =', sol_U.shape)
    if feature_level:
        np.save(os.path.join(save_dir, 'Combined-feature-{}.npy'.format(epsilon)), sol_U.detach().cpu().numpy())
    else:
        np.save(os.path.join(save_dir, 'Combined-{}.npy'.format(epsilon)), sol_U.detach().cpu().numpy())
    torch.cuda.empty_cache()

    
def per_class_dch_Standardization(args, train, test, device, save_dir, max_sparsity=20, stopping_criterion = 1, feature_level=False): # 1 = max, 2 = mean
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        
    if isinstance(train, list):
        train_imgs, train_labels = train[0], train[1]
    else:
        train_imgs, train_labels = train.data, np.array(train.targets)
    
    if isinstance(test, list):
        test_imgs, test_labels = test[0], test[1]
    else:
        test_imgs, test_labels = test.data, np.array(test.targets)

    labels = torch.unique(train_labels)
    
    train_setup = []
    for label in labels:
        train_label_imgs, train_label_targets = train_imgs[train_labels == label], train_labels[train_labels == label]
        test_label_imgs, test_label_targets = test_imgs[test_labels == label], test_labels[test_labels == label]
        
        
        duplication_idx_in_test = duplication_check(train_label_imgs, test_label_imgs)
        if len(duplication_idx_in_test) > 0:
            test_label_imgs, test_label_targets = delete_duplicates(test_label_imgs, duplication_idx_in_test), delete_duplicates(test_label_targets, duplication_idx_in_test)
            print('\t(Optional). After duplicate removal')
            print('\tShape of Train & Test', test_label_imgs.shape, test_label_targets.shape)
        
        epsilon = calculate_epsilon(args.d, train_label_imgs, test_label_imgs)
        del test_label_imgs
        print('======> Starting Calculate Train Set Convex hull of Label {}'.format(label))
        print('epsilon = ', epsilon)
        approx = DCH(train_label_imgs, epsilon, max_sparsity, stopping_criterion, cuda=torch.cuda.is_available())
        c_hull_t = approx.calculate_chull()
        sol_U = approx.U.t()
        print('sol_U combined shape =', sol_U.shape)
        if feature_level:
            np.save(os.path.join(save_dir, 'feature-{}-{}.npy'.format(int(label), epsilon)), sol_U.detach().cpu().numpy())
        else:
            np.save(os.path.join(save_dir, '{}-{}.npy'.format(int(label), epsilon)), sol_U.detach().cpu().numpy())
        torch.cuda.empty_cache()
    
    
def combined_dch_Standardization(args, frac_dir, test, d, device, save_dir, max_sparsity=20, stopping_criterion = 1, feature_level=False): # 1 = max, 2 = mean
    
    if feature_level:
        file_list = [i for i in os.listdir(frac_dir) if ('.npy' in i) and ('feature' in i)]
    else:
        file_list = [i for i in os.listdir(frac_dir) if ('.npy' in i) and ('feature' not in i)]
    
    if isinstance(test, list):
        test_imgs, test_labels = test[0], test[1]
    else:
        test_imgs, test_labels = test.data, np.array(test.targets)

    cvx_hull = np.zeros([1, d])
    
    for file in file_list:
        cvx_hull = np.concatenate([cvx_hull, np.load(os.path.join(save_dir, file))], axis=0)
    
    cvx_hull = cvx_hull[1:]
    
    try:
        cvx_hull = torch.from_numpy(cvx_hull).to(device)
    except TypeError:
        cvx_hull = cvx_hull.to(device)
    
    print('cvx hull shape = ', cvx_hull.shape, 'test img shape = ', test_imgs.shape)
    duplication_idx_in_test = duplication_check(cvx_hull, test_imgs.reshape((test_imgs.shape[0],-1)))

    if len(duplication_idx_in_test) > 0:
        test_imgs, test_labels= delete_duplicates(test_imgs, duplication_idx_in_test), delete_duplicates(test_labels, duplication_idx_in_test)
        print('\t(Optional). After duplicate removal')
        print('\tShape of Train & Test',cvx_hull.shape, test_imgs.shape, test_labels.shape)
    
    epsilon = calculate_epsilon(args.d, cvx_hull, test_imgs.type(torch.DoubleTensor).to(device)) #
    del test_imgs, test_labels, test
    print('======> Starting Calculate Combined Train Set Convex hull')
    print('epsilon = ', epsilon)
    
    cvx_hull = cvx_hull.type(torch.FloatTensor).to(device)
    approx = DCH(cvx_hull, epsilon, max_sparsity, stopping_criterion, cuda=torch.cuda.is_available())
    print('cvx_hull.shape',cvx_hull.shape)
    
    c_hull_t = approx.calculate_chull()
    sol_U = approx.U.t()
    print('sol_U shape =', sol_U.shape)
    if feature_level:
        np.save(os.path.join(save_dir, 'Combined-feature-{}.npy'.format(epsilon)), sol_U.detach().cpu().numpy())
    else:
        np.save(os.path.join(save_dir, 'Combined-{}.npy'.format(epsilon)), sol_U.detach().cpu().numpy())
    torch.cuda.empty_cache()
