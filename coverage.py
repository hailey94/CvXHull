from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyflann import FLANN
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity

import tool
import time

class Coverage:
    def __init__(self, model, layer_size_dict, hyper=None, **kwargs):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.model.to(self.device)
        self.layer_size_dict = layer_size_dict
        self.init_variable(hyper, **kwargs)
        self.sample_cov = None
        self.ids = None
        self.softmax = torch.nn.Softmax(dim=1)
        self.build_time = 0
        self.inference_time = 0
        self.labels = None
        self.predictions = None
        
    def init_variable(self):
        raise NotImplementedError
        

    def sample_coverage(self):
        raise NotImplementedError

    def build(self, data_loader):
        print('Building is not needed.')
        
    def collect_label(self, data_loader, corruption=False):
        if corruption:
            for num, (_, labels, _ , _) in tqdm(enumerate(data_loader)):
                labels = labels.detach().cpu().numpy()
                if num == 0:
                    lbs = labels
                else:
                    lbs =np.concatenate([lbs, labels])
            
        else:
            for num, (_, labels) in tqdm(enumerate(data_loader)):
                labels = labels.detach().cpu().numpy()
                if num == 0:
                    lbs = labels
                else:
                    lbs =np.concatenate([lbs, labels])
        return lbs
    
    def collect_predictions(self, data_loader, corruption=False):
        if corruption:
            for num, (data, _, _, _) in tqdm(enumerate(data_loader)):
                data = data.to(self.device)
                final_output = self.model(data)
                predictions = final_output.argmax(dim=1, keepdim=True).squeeze().detach().cpu().numpy()
                if num == 0:
                    lbs = predictions
                else:
                    lbs = np.concatenate([lbs, predictions])

        else:
            for num, (data, _) in tqdm(enumerate(data_loader)):
                data = data.to(self.device)
                final_output = self.model(data)
                predictions = final_output.argmax(dim=1, keepdim=True).squeeze().detach().cpu().numpy()
                if num == 0:
                    lbs = predictions
                else:
                    lbs = np.concatenate([lbs, predictions])
        return lbs

    def per_sample(self, data_loader):
        start = time.time()
        for num, (data, *_) in tqdm(enumerate(data_loader)):
            if isinstance(data, tuple):
                data = (data[0].to(self.device), data[1].to(self.device))
            else:
                data = data.to(self.device)
                
            coverages = self.sample_coverage(data).detach().cpu().numpy()
            # print(coverages, coverages.shape)
            if num == 0:
                self.sample_cov = coverages
            else:
                self.sample_cov =np.concatenate([self.sample_cov, coverages])
            del data
        self.inference_time = time.time() - start

class NLC2(Coverage):
    def init_variable(self, hyper=None):
        assert hyper is None, 'NLC has no hyper-parameter'
        self.estimator_dict = {}
        self.current = 1
        for (layer_name, layer_size) in self.layer_size_dict.items():
            self.estimator_dict[layer_name] = tool.Estimator(feature_num=layer_size[0])
    
    def assess(self, data_loader):
        start = time.time()
        for data, *_ in tqdm(data_loader):
            if isinstance(data, tuple):
                data = (data[0].to(self.device), data[1].to(self.device))
            else:
                data = data.to(self.device)
            self.step(data)
        self.build_time = time.time() - start
    
    def per_sample(self, data_loader):
        self.sample_cov = []
        start = time.time()
        for num, (data, *_) in tqdm(enumerate(data_loader)):
            if isinstance(data, tuple):
                data = (data[0].to(self.device), data[1].to(self.device))
            else:
                data = data.to(self.device)                
            cove_set = self.calculate(data)
            gain = self.gain(cove_set)
            if gain is not None:
                self.sample_cov.append(gain[0].detach().cpu().numpy())
            else:
                self.sample_cov.append(0)
        self.inference_time = time.time() - start
            
    def step(self, data): #origin
        cove_set = self.calculate(data)
        gain = self.gain(cove_set)
        if gain is not None:
            self.update(cove_set, gain)
            
    
    def calculate(self, data):
        stat_dict = {}
        layer_output_dict = tool.get_layer_output(self.model, data)
        for (layer_name, layer_output) in layer_output_dict.items():
            info_dict = self.estimator_dict[layer_name].calculate(layer_output.to(self.device))
            stat_dict[layer_name] = (info_dict['Ave'], info_dict['CoVariance'], info_dict['Amount'])
        return stat_dict

    def update(self, stat_dict, gain=None):
        if gain is None:    
            for i, layer_name in enumerate(stat_dict.keys()):
                (new_Ave, new_CoVariance, new_Amount) = stat_dict[layer_name]
                self.estimator_dict[layer_name].Ave = new_Ave
                self.estimator_dict[layer_name].CoVariance = new_CoVariance
                self.estimator_dict[layer_name].Amount = new_Amount
            self.current = self.coverage(self.estimator_dict)
        else:
            (delta, layer_to_update) = gain
            for layer_name in layer_to_update:
                (new_Ave, new_CoVariance, new_Amount) = stat_dict[layer_name]
                self.estimator_dict[layer_name].Ave = new_Ave
                self.estimator_dict[layer_name].CoVariance = new_CoVariance
                self.estimator_dict[layer_name].Amount = new_Amount
            self.current += delta

    def coverage(self, stat_dict):
        val = 0
        for i, layer_name in enumerate(stat_dict.keys()):
            # Ave = stat_dict[layer_name].Ave
            CoVariance = stat_dict[layer_name].CoVariance
            # Amount = stat_dict[layer_name].Amount
            val += self.norm(CoVariance)
        return val

    def gain(self, stat_new):
        total = 0
        layer_to_update = []
        for i, layer_name in enumerate(stat_new.keys()):
            (new_Ave, new_CoVar, new_Amt) = stat_new[layer_name]
            value = self.norm(new_CoVar) - self.norm(self.estimator_dict[layer_name].CoVariance)
            if value > 0:
                layer_to_update.append(layer_name)
                total += value
        if total > 0:
            return (total, layer_to_update)
        else:
            return None

    def norm(self, vec, mode='L1', reduction='mean'):
        m = vec.size(0)
        assert mode in ['L1', 'L2']
        assert reduction in ['mean', 'sum']
        if mode == 'L1':
            total = vec.abs().sum()
        elif mode == 'L2':
            total = vec.pow(2).sum().sqrt()
        if reduction == 'mean':
            return total / m
        elif reduction == 'sum':
            return total

            
class NLC(Coverage):
    def init_variable(self, hyper=None):
        assert hyper is None, 'NLC has no hyper-parameter'
        self.estimator_dict = {}
        self.current = 1
        for (layer_name, layer_size) in self.layer_size_dict.items():
            self.estimator_dict[layer_name] = tool.Estimator(feature_num=layer_size[0])
    
    def batch_cov(self,data):
        N, D = data.size()
        mean = data.mean(dim=1).unsqueeze(1)
        diffs = (data - mean)
        prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(N, D, D)
        return prods  # (N, D, D)
    
    def sample_coverage(self, data):
        stat_dict = {}
        layer_output_dict = tool.get_layer_output(self.model, data)
        cove = torch.zeros(data.shape[0]).to(self.device)
        for (layer_name, layer_output) in layer_output_dict.items():
            prods = self.batch_cov(layer_output)
            cove += prods.abs().sum() / (prods.size(1)*prods.size(1))
        return cove


    def norm(self, vec, mode='L1', reduction='mean'):
        m = vec.squeeze().size(0)
        # print('m = ', m)
        assert mode in ['L1', 'L2']
        assert reduction in ['mean', 'sum']
        if mode == 'L1':
            total = vec.abs().sum()
        elif mode == 'L2':
            total = vec.pow(2).sum().sqrt()
        if reduction == 'mean':
            return total / (m)
        elif reduction == 'sum':
            return total

class FDplus2(Coverage):
    def init_variable(self, hyper, alpha, eps, num_class):
        self.epsilon, self.alpha = eps, alpha
        self.num_class = num_class
        self.name = 'FDplus2'
        self.bounds = {}
        self.status_dict = {}
        self.avg_dict = {}
        self.var_dict = {}
        self.data_count = 0
        for (layer_name, layer_size) in self.layer_size_dict.items():
            self.bounds[layer_name] = {i:np.zeros([layer_size[0]]) for i in range(self.num_class)}            
    
    def build(self, data_loader):
        print('collect status...')
        start = time.time()
        self.status_collenction(data_loader)
        self.calculate_bounds()
        self.build_time = time.time() - start
            
    def calculate_bounds(self):
        for (layer_name, averages), (_, variances) in zip(self.avg_dict.items(), self.var_dict.items()):
            activ_info = self.status_dict[layer_name]

            for l in range(self.num_class):
                avg_l = averages[l]
                var_l = torch.sqrt(variances[l])
                lb = avg_l - 1.65 * var_l
                ub = avg_l + 1.65 * var_l                
                active = (activ_info[l] == 1).nonzero().cpu().detach()
                inactive = (activ_info[l] == -1).nonzero().cpu().detach()
                assert len(set(active).intersection(set(inactive))) == 0
                self.bounds[layer_name][l][active] = lb[active].cpu().detach()
                self.bounds[layer_name][l][inactive] = ub[inactive].cpu().detach()

    
    def status_collenction(self, data_loader):
        start = time.time()
        status_dict = {}
        avg_dict = {}
        var_dict = {}
        
        for (layer_name, layer_size) in self.layer_size_dict.items():
            num_neuron = layer_size[0]            
            status_dict[layer_name] = torch.zeros([self.num_class, num_neuron]).to(self.device)
            self.avg_dict[layer_name] = torch.zeros([self.num_class, num_neuron]).to(self.device)
            self.var_dict[layer_name] = torch.zeros([self.num_class, num_neuron]).to(self.device)
            
        per_label_instances = torch.zeros([self.num_class, 1]).to(self.device)
        for data, label in tqdm(data_loader):
            data = data.to(self.device)  
            batch_size = data.size(0)
            layer_output_dict = tool.get_layer_output(self.model, data, get_max=True)
            label_info = torch.unique(label)
            for num, (layer_name, layer_output) in enumerate(layer_output_dict.items()):
                self.data_count += batch_size
                for l in label_info:
                    label_ids = (label == l).nonzero()
                    status_dict[layer_name][l] += torch.sum(layer_output[label_ids] > 0, dim=0).squeeze()
                    if num == 0:
                        per_label_instances[l] += len(label_ids)
                    self.avg_dict[layer_name][l] = ((per_label_instances[l] - len(label_ids)) * self.avg_dict[layer_name][l] + layer_output[label_ids].sum(0)) / per_label_instances[l]
                    self.var_dict[layer_name][l] = (per_label_instances[l] - len(label_ids)) * self.var_dict[layer_name][l] / per_label_instances[l] \
            + (per_label_instances[l] - len(label_ids)) * ((layer_output - self.avg_dict[layer_name][l]) ** 2).sum(0) / per_label_instances[l] ** 2
                    
        
        for layer_name, values in status_dict.items():
            ratio = values/per_label_instances
            status = torch.zeros(ratio.shape).to(self.device)
            status[ratio >= self.epsilon] = 1
            status[ratio < 1 - self.epsilon] = -1
            self.status_dict[layer_name] = status
        del status_dict
            
    def sample_coverage(self, data):
        
        layer_output_dict, final_output = tool.get_layer_output(self.model, data, get_final=True, get_max=True)
        predictions = self.softmax(final_output).argmax(dim=1, keepdim=True).squeeze()
        del final_output
        N = 0
        Sp = torch.zeros(data.shape[0]).to(self.device)
        Sn = torch.zeros(data.shape[0]).to(self.device)
        
        for (layer_name, layer_output) in layer_output_dict.items():
            lmax, lmin = layer_output.max(), layer_output.min()
            layer_status = self.status_dict[layer_name][predictions]
            bounds = np.array([self.bounds[layer_name][i.item()] for i in predictions])
            bounds = torch.from_numpy(bounds).to(self.device)
            
            #### for triggered #####
            mask = torch.zeros(layer_status.shape).to(self.device)
            mask[layer_status == 1] = 1 # active condition
            triggered = layer_output * mask # active neuron's output only
            
            mask_bound = torch.zeros(layer_status.shape).to(self.device)
            mask_bound[layer_status == 1] = 1 # lb for active condition
            lbs = bounds * mask_bound
            
            mask_bound = torch.zeros(layer_status.shape).to(self.device)
            mask_bound[layer_status != 1] = lmax+1000000 # lb for active condition
            lbs = lbs + mask_bound
            
            Sp += (triggered > lbs).sum(dim=1)
            
            #### for inhibited #####
            mask = torch.zeros(layer_status.shape).to(self.device)
            mask[layer_status == -1] = 1 # active condition
            inhibited = layer_output * mask # active neuron's output only
            
            mask_bound = torch.zeros(layer_status.shape).to(self.device)
            mask_bound[layer_status == -1] = 1 # lb for active condition
            ubs = bounds * mask_bound
            mask_bound = torch.zeros(layer_status.shape).to(self.device)
            mask_bound[layer_status != -1] = lmin + 1000000 # lb for active condition
            ubs = ubs - mask_bound
            
            Sn += (inhibited < ubs).sum(dim=1)
                               
            # print(active)
            N += layer_status.shape[1]
        return (Sp + Sn)/N

    

        
class KMNC(Coverage):
    def init_variable(self, hyper):
        assert hyper is not None
        self.k = hyper
        self.name = 'KMNC'
        self.range_dict = {}
        for (layer_name, layer_size) in self.layer_size_dict.items():
            num_neuron = layer_size[0]
            self.range_dict[layer_name] = [torch.ones(num_neuron).to(self.device) * 10000, torch.ones(num_neuron).to(self.device) * -10000]
        
        self.current = 0

    def build(self, data_loader):
        print('Building range...')
        start = time.time()
        for data, *_ in tqdm(data_loader):
            if isinstance(data, tuple):
                data = (data[0].to(self.device), data[1].to(self.device))
            else:
                data = data.to(self.device)
            self.set_range(data)
        self.build_time = time.time() - start

    def set_range(self, data):
        layer_output_dict = tool.get_layer_output(self.model, data)
        for (layer_name, layer_output) in layer_output_dict.items():
            cur_max, _ = layer_output.max(0) # per neuron maximum
            cur_min, _ = layer_output.min(0) # per neuron minimum
            is_less = cur_min < self.range_dict[layer_name][0]
            is_greater = cur_max > self.range_dict[layer_name][1]
            # smaller -> change to cur_min, else: remained as range_dict
            self.range_dict[layer_name][0] = is_less * cur_min + ~is_less * self.range_dict[layer_name][0] 
            # bigger -> change to cur_max, else: remained as range_dict
            self.range_dict[layer_name][1] = is_greater * cur_max + ~is_greater * self.range_dict[layer_name][1]
    
            

class NBC(KMNC):
    def init_variable(self, hyper=None):
        assert hyper is None
        self.name = 'NBC'
        self.range_dict = {}
        coverage_lower_dict = {}
        coverage_upper_dict = {}
        for (layer_name, layer_size) in self.layer_size_dict.items():
            num_neuron = layer_size[0]
            coverage_lower_dict[layer_name] = torch.zeros(num_neuron).type(torch.BoolTensor).to(self.device)
            coverage_upper_dict[layer_name] = torch.zeros(num_neuron).type(torch.BoolTensor).to(self.device)
            self.range_dict[layer_name] = [torch.ones(num_neuron).to(self.device) * 10000, torch.ones(num_neuron).to(self.device) * -10000]
        self.coverage_dict = {
            'lower': coverage_lower_dict,
            'upper': coverage_upper_dict
        }
        self.current = 0

    def calculate(self, data):
        lower_cove_dict = {}
        upper_cove_dict = {}
        layer_output_dict = tool.get_layer_output(self.model, data)
        for (layer_name, layer_output) in layer_output_dict.items():
            [l_bound, u_bound] = self.range_dict[layer_name]
            
            lower_covered = (layer_output < l_bound).sum(0) > 0
            upper_covered = (layer_output > u_bound).sum(0) > 0

            lower_cove_dict[layer_name] = lower_covered | self.coverage_dict['lower'][layer_name]
            upper_cove_dict[layer_name] = upper_covered | self.coverage_dict['upper'][layer_name]
        
        return {
            'lower': lower_cove_dict,
            'upper': upper_cove_dict
        }
    
    def sample_coverage(self, data):
        lower_coverages = torch.zeros(data.shape[0]).to(self.device)
        upper_coverages = torch.zeros(data.shape[0]).to(self.device)
        num_total = 0
        
        layer_output_dict = tool.get_layer_output(self.model, data)
        for (layer_name, layer_output) in layer_output_dict.items():
            [l_bound, u_bound] = self.range_dict[layer_name]
            
            lower_coverages += (layer_output < l_bound).sum(1)
            lower_coverages += (layer_output > u_bound).sum(1)
            num_total += layer_output.size(1)

        return (lower_coverages + lower_coverages)/ (2*num_total)


    

class SurpriseAdequacy2(Coverage):
    def init_variable(self, hyper, min_var, num_class):
        self.name = self.get_name()
        assert self.name in ['LSA2', 'DSA2']
        self.threshold = hyper
        self.min_var = min_var
        self.num_class = num_class
        self.data_count = 0
        self.current = 0
        self.kde_cache = {}
        self.SA_cache = {}
        self.mask_index_dict = torch.ones(self.num_class).type(torch.LongTensor).to(self.device)
        self.mean_dict = torch.zeros(self.num_class).to(self.device)
        self.var_dict = torch.zeros(self.num_class).to(self.device)

    def get_name(self):
        raise NotImplementedError


    def build(self, data_loader):
        print('Building Mean & Var...')
        start = time.time()
        for i, (data, label) in enumerate(tqdm(data_loader)):
            if isinstance(data, tuple):
                data = (data[0].to(self.device), data[1].to(self.device))
            else:
                data = data.to(self.device)
            self.set_meam_var(data, label) #per layer mean & var calculation
        self.set_mask() # exclude the neurons with activation variance less than t
        print('Building SA...')
        for i, (data, label) in enumerate(tqdm(data_loader)):
            if isinstance(data, tuple):
                data = (data[0].to(self.device), data[1].to(self.device))
            else:
                data = data.to(self.device)
            label = label.to(self.device)
            self.build_SA(data, self.name)
        self.to_numpy(self.name)
        if self.name == 'LSA2':
            self.set_kde()
        self.build_time = time.time() - start
        self.var_dict = None
        self.mean_dict = None
        torch.cuda.empty_cache()
        

    def per_sample(self, data_loader):
        start = time.time()
        for num, (data, label) in tqdm(enumerate(data_loader)):
            data = data.to(self.device)
            label = label.to(self.device)
            coverages = self.sample_coverage(data).detach().cpu().numpy()
            if num == 0:
                self.sample_cov = coverages
            else:
                self.sample_cov =np.concatenate([self.sample_cov, coverages])
            del data, label
        self.inference_time = time.time() - start
        self.mask_index_dict = None
        self.kde_cache = None
        self.SA_cache = None
        torch.cuda.empty_cache()

    def set_meam_var(self, data, label):
        batch_size = label.size(0)
        _, final_output = tool.get_layer_output(self.model, data, get_final=True)
        self.data_count += batch_size
        self.mean_dict = ((self.data_count - batch_size) * self.mean_dict + final_output.sum(0)) / self.data_count
        self.var_dict = (self.data_count - batch_size) * self.var_dict / self.data_count \
            + (self.data_count - batch_size) * ((final_output - self.mean_dict) ** 2).sum(0) / self.data_count ** 2

    def set_mask(self):
        feature_num = 0
        self.mask_index_dict = (self.var_dict >= self.min_var).nonzero()
        feature_num += self.mask_index_dict.size(0)
        print('feature_num: ', feature_num)

    def build_SA(self, data_batch, name):
        SA_batch = []
        batch_size = data_batch.size(0)
        _, final_output = tool.get_layer_output(self.model, data_batch, get_final=True)
        predictions = self.softmax(final_output).argmax(dim=1, keepdim=True).squeeze()
        
        SA_batch.append(final_output[:, self.mask_index_dict].view(batch_size, -1)) #flatten
        SA_batch = torch.cat(SA_batch, 1) # [batch_size, num_neuron]
        SA_batch = SA_batch[~torch.any(SA_batch.isnan(), dim=1)]
        SA_batch = SA_batch[~torch.any(SA_batch.isinf(), dim=1)]
        #save Train feature values in SA_cache with idx
        for i, label in enumerate(predictions):
            if 'LSA' in name:
                if int(label.cpu()) in self.SA_cache.keys():
                    self.SA_cache[int(label.cpu())] += [SA_batch[i].detach().cpu().numpy()]
                else:
                    self.SA_cache[int(label.cpu())] = [SA_batch[i].detach().cpu().numpy()]
            elif 'DSA' in name:
                if int(label.cpu())in self.SA_cache.keys():
                    self.SA_cache[int(label.cpu())] = torch.cat([self.SA_cache[int(label.cpu())], SA_batch[i].unsqueeze(0)], dim=0)
                else:
                    self.SA_cache[int(label.cpu())] = SA_batch[i].unsqueeze(0)
        
    def to_numpy(self, name):
        if 'LSA' in name:
            for k in self.SA_cache.keys():
                self.SA_cache[k] = np.stack(self.SA_cache[k], 0)

    def set_kde(self):
        raise NotImplementedError




class LSA2(SurpriseAdequacy2):
    def get_name(self):
        return 'LSA2'

    def set_kde(self):
        for k in self.SA_cache.keys():
            if self.num_class <= 1:
                self.kde_cache[k] = gaussian_kde(self.SA_cache[k].T)
            else:
                self.kde_cache[k] = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(self.SA_cache[k])
            # The original LSC uses the `gaussian_kde` function, however, we note that this function
            # frequently crashes due to numerical issues, especially for large `num_class`.
            
    def sample_coverage(self, data_batch):
        SA_batch = []
        batch_size = data_batch.size(0)
        cove_set = torch.zeros(batch_size).to(self.device)
        _, final_output = tool.get_layer_output(self.model, data_batch, get_final=True)

        SA_batch.append(final_output[:, self.mask_index_dict].view(batch_size, -1))
        SA_batch = torch.cat(SA_batch, 1) # [batch_size, num_neuron]
        predictions = self.softmax(SA_batch).argmax(dim=1, keepdim=True).squeeze()
        SA_batch = SA_batch.detach().cpu().numpy()
        
        for i, label in enumerate(predictions):
            SA = SA_batch[i]
            if self.num_class <= 1:
                lsa = -self.kde_cache[int(label.cpu())].logpdf(np.expand_dims(SA, 1)).item()
            else:
                lsa = -self.kde_cache[int(label.cpu())].score_samples(np.expand_dims(SA, 0)).item()
            cove_set[i] = lsa
        # print(cove_set)
        return cove_set
    

class FDplus(Coverage):
    def init_variable(self, hyper, alpha, eps, num_class):
        self.epsilon, self.alpha = eps, alpha
        self.num_class = num_class
        self.name = 'FDplus'
        self.bounds = {}
        self.status_dict = {}        
        for (layer_name, layer_size) in self.layer_size_dict.items():
            self.bounds[layer_name] = {i:np.zeros([layer_size[0]]) for i in range(self.num_class)}
        
    
    def build(self, data_loader):
        print('collect status...')
        start = time.time()
        self.status_collenction(data_loader)
        status_dict = self.collect_bounds(data_loader)
        self.calculate_bounds(status_dict)
        self.build_time = time.time() - start
        print(self.build_time)
        
    def collect_bounds(self, data_loader):
        status_dict = {}        
        for (layer_name, layer_size) in self.layer_size_dict.items():
            status_dict[layer_name] = {i:torch.zeros([1, layer_size[0]]) for i in range(self.num_class)}

        for data, label, ids in tqdm(data_loader):
            data = data.to(self.device)
            layer_output_dict = tool.get_layer_output(self.model, data, get_max=True)
            label_info = torch.unique(label)
            for num, (layer_name, layer_output) in enumerate(layer_output_dict.items()):
                for l in label_info:
                    label_ids = (label == l).nonzero()
                    label_layer_output = layer_output[label_ids]
                    label_layer_output = label_layer_output.squeeze(1).cpu() # (# label, num_neuron)
                    status_dict[layer_name][l.item()] = torch.cat([status_dict[layer_name][l.item()], \
                                                                        label_layer_output], dim=0)                            
        return status_dict
    
    def calculate_bounds(self, status_dict):
        for layer_name, values in status_dict.items():
            activ_info = self.status_dict[layer_name]
            for c, fs in values.items():
                fs = fs[1:].numpy()
                lb = np.percentile(fs, 100*(self.alpha), axis=0)
                ub = np.percentile(fs, 100*(1-self.alpha), axis=0)
                data_num = fs.shape[0]
                
                active = (activ_info[c] == 1).nonzero().cpu().detach()
                inactive = (activ_info[c] == -1).nonzero().cpu().detach()
                assert len(set(active).intersection(set(inactive))) == 0
                self.bounds[layer_name][c][active] = lb[active]
                self.bounds[layer_name][c][inactive] = ub[inactive]
    
    def status_collenction(self, data_loader):
        start = time.time()
        status_dict = {}
        for (layer_name, layer_size) in self.layer_size_dict.items():
            num_neuron = layer_size[0]
            status_dict[layer_name] = torch.zeros([self.num_class, num_neuron]).to(self.device)
            
        per_label_instances = torch.zeros([self.num_class, 1]).to(self.device)
        for data, label, ids in tqdm(data_loader):
            data = data.to(self.device)            
            layer_output_dict = tool.get_layer_output(self.model, data, get_max=True)
            label_info = torch.unique(label)
            for num, (layer_name, layer_output) in enumerate(layer_output_dict.items()):
                for l in label_info:
                    label_ids = (label == l).nonzero()
                    status_dict[layer_name][l] += torch.sum(layer_output[label_ids] > 0, dim=0).squeeze()
                    
                    if num == 0:
                        per_label_instances[l] += len(label_ids)
        
        for layer_name, values in status_dict.items():
            ratio = values/per_label_instances
            status = torch.zeros(ratio.shape).to(self.device)
            status[ratio >= self.epsilon] = 1
            status[ratio < 1 - self.epsilon] = -1
            self.status_dict[layer_name] = status
        del status_dict
            
    def sample_coverage(self, data):
        
        layer_output_dict, final_output = tool.get_layer_output(self.model, data, get_final=True, get_max=True)
        predictions = self.softmax(final_output).argmax(dim=1, keepdim=True).squeeze()
        del final_output
        N = 0
        Sp = torch.zeros(data.shape[0]).to(self.device)
        Sn = torch.zeros(data.shape[0]).to(self.device)
        
        for (layer_name, layer_output) in layer_output_dict.items():
            lmax, lmin = layer_output.max(), layer_output.min()
            layer_status = self.status_dict[layer_name][predictions]
            bounds = np.array([self.bounds[layer_name][i.item()] for i in predictions])
            bounds = torch.from_numpy(bounds).to(self.device)
            
            #### for triggered #####
            mask = torch.zeros(layer_status.shape).to(self.device)
            mask[layer_status == 1] = 1 # active condition
            triggered = layer_output * mask # active neuron's output only
            
            mask_bound = torch.zeros(layer_status.shape).to(self.device)
            mask_bound[layer_status == 1] = 1 # lb for active condition
            lbs = bounds * mask_bound
            
            mask_bound = torch.zeros(layer_status.shape).to(self.device)
            mask_bound[layer_status != 1] = lmax+1000000 # lb for active condition
            lbs = lbs + mask_bound
            
            Sp += (triggered > lbs).sum(dim=1)
            
            #### for inhibited #####
            mask = torch.zeros(layer_status.shape).to(self.device)
            mask[layer_status == -1] = 1 # active condition
            inhibited = layer_output * mask # active neuron's output only
            
            mask_bound = torch.zeros(layer_status.shape).to(self.device)
            mask_bound[layer_status == -1] = 1 # lb for active condition
            ubs = bounds * mask_bound
            mask_bound = torch.zeros(layer_status.shape).to(self.device)
            mask_bound[layer_status != -1] = lmin + 1000000 # lb for active condition
            ubs = ubs - mask_bound
            
            Sn += (inhibited < ubs).sum(dim=1)
                               
            # print(active)
            N += layer_status.shape[1]
        return (Sp + Sn)/N

    
class DSA2(SurpriseAdequacy2):
    def get_name(self):
        return 'DSA2'
    
    def sample_coverage(self, data_batch):
        SA_batch = []
        batch_size = data_batch.size(0)
        cove_set = torch.zeros(batch_size).to(self.device)
        _, final_output = tool.get_layer_output(self.model, data_batch, get_final=True)
        
        SA_batch.append(final_output[:, self.mask_index_dict].view(batch_size, -1))
        SA_batch = torch.cat(SA_batch, 1) # [batch_size, num_neuron]
        predictions = self.softmax(SA_batch).argmax(dim=1, keepdim=True).squeeze()
        
        for i, label in enumerate(predictions):
            SA = SA_batch[i]

            dist_a_list = torch.linalg.norm(SA - self.SA_cache[int(label.cpu())], dim=1)
            idx_a = torch.argmin(dist_a_list, 0).item() #extract index of minimum distance of same label train feature's index 
 
            (SA_a, dist_a) = (self.SA_cache[int(label.cpu())][idx_a], dist_a_list[idx_a]) # that train feature value, distance

            dist_b_list = torch.tensor([]).to(self.device)
            for j in range(self.num_class):
                if ( j != int(label.cpu()) ) and ( j in self.SA_cache.keys() ):
                    dist_b_list = torch.cat([dist_b_list, torch.linalg.norm(SA - self.SA_cache[j],dim=1)])                    
            dist_b = torch.min(dist_b_list)
            dsa = dist_a / dist_b if dist_b > 0 else 1e-6
            cove_set[i] = dsa.item()
        return cove_set

        
if __name__ == '__main__':
    pass

