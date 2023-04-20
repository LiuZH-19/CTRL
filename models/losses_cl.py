from distutils.dep_util import newer_group
from operator import ne
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from models.gen_neg import mix


def contrastive_loss(z1, z2, z_neg = None, alpha=0.5, temporal_unit=0, debiase = False, threshold = 0.98, topk = 0.4, temp_H = True, ins_H = True):
    B, T = z1.size(0), z1.size(1)
    temperature = 1
    ins_loss = torch.tensor(0., device=z1.device)
    temp_loss = torch.tensor(0., device=z1.device)
    d_ins = 0
    d_temp = 0

    if (alpha != 0) and (not ins_H):
        z1_ins = F.max_pool1d(z1.transpose(1, 2), kernel_size=T).transpose(1, 2)
        z2_ins = F.max_pool1d(z2.transpose(1, 2), kernel_size=T).transpose(1, 2)
        ins_loss += instance_contrastive_loss(z1_ins, z2_ins, debiase, temperature=temperature, z_neg = z_neg, threshold = threshold, topk = topk)
        d_ins +=1
    if  (1-alpha != 0) and (not temp_H):
        temp_loss += temporal_contrastive_loss(z1, z2,debiase, temperature=temperature)
        d_temp += 1

    
    while z1.size(1) > 1 and (ins_H or temp_H):
        if alpha != 0 and ins_H : 
            ins_loss += instance_contrastive_loss(z1, z2, debiase, temperature=temperature, z_neg = z_neg, threshold = threshold, topk = topk)
            d_ins +=  1
        if d_temp >= temporal_unit: 
            if 1 - alpha != 0 and temp_H:
                temp_loss += temporal_contrastive_loss(z1, z2, debiase,temperature=temperature) 
                d_temp +=  1
    
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)  
        if z_neg is not None:
            z_neg = F.max_pool1d(z_neg.transpose(1, 2), kernel_size=2).transpose(1, 2) 
    if z1.size(1) == 1:
        d_temp +=  1
        if alpha != 0 and ins_H:
            ins_loss +=  instance_contrastive_loss(z1, z2, debiase, temperature=temperature, z_neg = z_neg, threshold = threshold, topk = topk)
            d_ins +=  1
    if d_ins != 0:    
        ins_loss /= d_ins
    if d_temp != 0:
        temp_loss /= d_temp
    print(f'temporal_loss: {temp_loss}, instance_loss:{ins_loss}')
    return alpha * ins_loss + (1-alpha) * temp_loss, ins_loss, temp_loss


def instance_contrastive_loss(z1, z2, debiase = False, temperature = None, z_neg = None,  threshold =None, topk = None):
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B

    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
            
    if temperature is not None:
        logits /= temperature
    
    #hard_neg
    if z_neg is not None:
        #z_neg [B, T, C]
        neg_ins = z_neg.transpose(0, 1) 
        logits_neg =  torch.matmul(z, neg_ins.transpose(1, 2)) # [T, 2B, B]
        
        if temperature is not None:
            logits_neg /= temperature

        logits = torch.cat([logits, logits_neg], -1)


    i = torch.arange(B, device=z1.device)

    mask_score = torch.tensor(logits)          
    pos_mask = torch.full(mask_score.shape, False, dtype=torch.bool, device=mask_score.device)
    pos_mask[:, i, B + i - 1] =True
    pos_mask[:, B + i, i] = True
    pos_score = torch.masked_select(mask_score, pos_mask)
    neg_score = torch.masked_select(mask_score, ~pos_mask)
    
    if debiase and B > 35:
        pos_sim = mask_score*(pos_mask.float())
        pos_sim =  torch.sum(pos_sim, dim =-1, keepdim=True)
        assert pos_sim.std().item()== pos_score.std().item(), f'pos_sim is wrong:{pos_sim.std().iterm}  {pos_score.std().item()}, {pos_sim.shape}'
        mask_logits1 = mask_score > threshold * pos_sim   
        top_P = topk
        
        if mask_logits1.float().mean() <  top_P-0.1:
            mask_logits = mask_logits1
        else:
            k =int(mask_score.size(-1)* top_P)
            values, indices = mask_score.topk( k, dim=-1, largest=True) #[T, 2B,K]
            kth_value = values.min(-1).values.unsqueeze(-1) #[T,2B,1]
            kth_value = kth_value.repeat(1,1, mask_score.size(-1))
            mask_logits2 = mask_score > kth_value
            mask_logits = mask_logits1 * mask_logits2           
        mask_logits[:, i, B + i - 1] = False
        mask_logits[:, B + i, i] = False   
        logits[mask_logits]=-float('inf')         

    logits = -F.log_softmax(logits, dim=-1)
    
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss


def temporal_contrastive_loss(z1, z2, debiase = False, temperature =None):
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
   
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]

    if temperature is not None:
        logits /= temperature

    t = torch.arange(T, device=z1.device)

    mask_score = torch.tensor(logits) 

    if debiase and T > 35: 
        top_P = 0.04      
        k =int(mask_score.size(-1)*top_P)
        values, indices = mask_score.topk( k, dim=-1, largest=True) #[T, 2B,K]
    
        kth_value = values.min(-1).values.unsqueeze(-1) #[T,2B,1]
        kth_value = kth_value.repeat(1,1, mask_score.size(-1))
            
        mask_logits = mask_score > kth_value
        mask_logits[:, t, T + t - 1] = False
        mask_logits[:, T + t, t] = False      
        logits[mask_logits]=-float('inf')
    
    logits = -F.log_softmax(logits, dim=-1)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    return loss
