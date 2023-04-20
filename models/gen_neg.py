import random
from signal import Sigmasks
from tempfile import tempdir
import torch
import torch.nn.functional as F
import numpy as np

def mix(inputs, beta):
    b, t, c = inputs.shape
    lam = torch.from_numpy(np.random.beta(beta,1,(b,1,1))).float().to(inputs.device)
    # print('lam mean',lam.mean().item())
    ids1 = torch.randperm(b)
    inputs1 = inputs[ids1]    
    return lam*inputs+ (1-lam)*inputs1

def adjacent_shuffle4(x):
    # [T,C]
    tmp = torch.chunk(x, 4, dim=0)
    order = [0,1,2,3]
    ind1 = random.randint(0,3)
    ind2 = (ind1 + random.randint(0,2) + 1) % 4 
    order[ind1], order[ind2] = order[ind2], order[ind1]
    x_new = torch.cat((tmp[order[0]], tmp[order[1]], tmp[order[2]], tmp[order[3]]),0)
    return x_new

def adjacent_shuffle8(x):
    # [T,C]
    tmp = torch.chunk(x, 8, dim=0)
    num = len(tmp)
    order = [i for i in range(num)]
    ind1 = random.randint(0,3)
    ind2 = (ind1 + random.randint(0,2) + 1) % 4
    order[ind1], order[ind2] = order[ind2], order[ind1]

    ind3 = random.randint(0,num-5)
    ind4 = (ind3 + random.randint(0,num-6) + 1) % (num-4)
    order[4+ind3], order[4+ind4] = order[4+ind4], order[4+ind3]
    tmp_new = [tmp[j] for j in order ]
    x_new = torch.cat(tmp_new, 0)    
    return x_new


p = 0.3
sigma = 0.3

def jitter(x):
    if random.random() > p:
        return x
    return x + (torch.randn(x.shape).to(x.device) * sigma)

def scale( x):
    if random.random() > p:
        return x
    return x * (torch.randn(x.size(-1)).to(x.device) * sigma + 1)

def shift( x):
    if random.random() > p:
        return x
    return x + (torch.randn(x.size(-1)).to(x.device) * sigma)

def transform(x):
        return jitter(shift(scale(x)))


def gen_neg_view(inputs, option): 
    b, t, c = inputs.shape

    if option == 'mix2':
        return mix(inputs,2)
    elif option == 'mix4':
        return mix(inputs, 4)
    else:
        new_in = []
        for i in range(b):
            one_sample = inputs[i,:,:]
            if option == 'shuffle4':
                new_in.append(adjacent_shuffle4(one_sample))
            elif option == 'shuffle8':
                new_in.append(adjacent_shuffle8(one_sample))
            elif option == 'transform' : 
                new_in.append(transform(one_sample))
            elif option == 'jitter' : 
                new_in.append(jitter(one_sample))
            elif option == 'scale' :
                new_in.append(scale(one_sample))
            else:
                print('Wrong option to generate hard negatives!')
    return torch.stack(new_in)
