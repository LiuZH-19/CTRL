import numpy as np
import time
from . import _eval_protocols as eval_protocols
import pandas as pd
import torchcde
import torch

def generate_samples(model, missing_data, data, seq_len, missing_mask = None):
    N, T, F = missing_data.shape
    features =[]
    labels = []
    # print('T', T, 'SEQ_LEN', seq_len)
    for i in range(0, T-seq_len):
        x_i = missing_data[:, i: i+seq_len]
        y_i = data[:, i: i+seq_len]
        f_i = model.encode(
                x_i,
                batch_size=256
                )
        assert f_i.shape[0]==y_i.shape[0] and f_i.shape[1]==y_i.shape[1], f"f_i.shape:{f_i.shape}, y_i.shape:{y_i.shape}"
        features.append(f_i)
        labels.append(y_i)
    features = np.stack(features, axis = 1)#[N, n_sample, seq_len, F]
    features = features.reshape(-1, features.shape[-1]) 
    labels = np.stack(labels, axis = 1).reshape(-1, F)
    if missing_mask is not None:
        masks = np.stack([missing_mask[:, j: j+seq_len] for j in range(0, T-seq_len)], axis =1). reshape(-1)
        assert masks.shape[0] == features.shape[0]
        return features[masks], labels[masks]
    else:
        return features, labels


def cal_metrics(pred, target):
    return {
        'MSE': ((pred - target) ** 2).mean(),
        'MAE': np.abs(pred - target).mean(),
    }

    
def eval_imputation(model, data, missing_data, missing_mask, train_slice, valid_slice, test_slice, scaler, lens):
    
    train_miss_data = missing_data[:, train_slice]
    valid_miss_data =  missing_data[:, valid_slice]
    test_miss_data =  missing_data[:, test_slice]
    
    train_data = data[:, train_slice]
    valid_data = data[:, valid_slice]
    test_data = data[:, test_slice]

    train_mask = missing_mask[:, train_slice]
    valid_mask = missing_mask[:, valid_slice]
    

    train_features, train_labels = generate_samples(model, train_miss_data, train_data, lens, train_mask )
    valid_features, valid_labels = generate_samples(model, valid_miss_data, valid_data, lens, valid_mask )
    test_features, test_labels = generate_samples(model, test_miss_data, test_data, lens)
    
    t = time.time()
    lr = eval_protocols.fit_ridge(train_features, train_labels, valid_features, valid_labels)
    lr_train_time = time.time() - t
    
    t = time.time()
    test_pred = lr.predict(test_features)
    lr_infer_time= time.time() - t
   
    test_mask = missing_mask[:, test_slice]
    test_mask = np.stack([test_mask[:, j: j+lens] for j in range(0, test_data.shape[1]-lens)], axis =1)


    ori_shape = test_data.shape[0], -1, lens, test_data.shape[-1]
    test_pred = test_pred.reshape(ori_shape) 
    test_labels = test_labels.reshape(ori_shape)
    
    
    

    if test_data.shape[0] > 1:
        test_pred_inv = scaler.inverse_transform(test_pred.swapaxes(0, 3)).swapaxes(0, 3)
        test_labels_inv = scaler.inverse_transform(test_labels.swapaxes(0, 3)).swapaxes(0, 3)
        
    else:
        test_pred_inv = scaler.inverse_transform(test_pred)
        test_labels_inv = scaler.inverse_transform(test_labels)
        

    out_log = {
        'norm': test_pred,
        'raw': test_pred_inv,
        'norm_gt': test_labels,
        'raw_gt': test_labels_inv
    }
    print('test mask ratio:', test_mask.mean())
    norm_metrics = cal_metrics(test_pred[test_mask], test_labels[test_mask])
    raw_metrics = cal_metrics(test_pred_inv[test_mask], test_labels_inv[test_mask])

    eval_res = {
        'norm': norm_metrics,
        'raw': raw_metrics,
        'lr_infer_time': lr_infer_time
    }
    return out_log, eval_res
