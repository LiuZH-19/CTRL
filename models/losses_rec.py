import torch
from torch import nn
import torch.nn.functional as F

def masked_mae_loss(y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.BoolTensor):
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    mask = torch.where(torch.isnan(y_pred), torch.zeros_like(mask), mask) 
    mask = torch.where(torch.isnan(y_true), torch.zeros_like(mask), mask) 
    loss = torch.abs(y_pred-y_true)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def get_align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=-1).pow(alpha).mean()
    
def get_uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


def masked_mse_loss(y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.BoolTensor):
    raw_result = masked_mse_loss2(y_pred, y_true, mask)
    mask = mask.float()
    mask /= torch.mean((mask))
    print('masked_mse_loss mask mean:', torch.mean((mask)))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    mask = torch.where(torch.isnan(y_pred), torch.zeros_like(mask), mask) 
    mask = torch.where(torch.isnan(y_true), torch.zeros_like(mask), mask) 
    loss = (y_pred-y_true)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    result = torch.mean(loss)
    assert abs(result- raw_result)< 0.01, f"result = {result}; raw_result= {raw_result}"
    return result

def masked_rmse_loss(y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.BoolTensor):
    return torch.sqrt(masked_mse_loss(y_pred, y_true, mask))




